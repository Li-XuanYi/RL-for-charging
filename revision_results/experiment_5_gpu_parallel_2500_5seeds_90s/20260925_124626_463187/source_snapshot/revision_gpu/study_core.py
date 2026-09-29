"""Shared protocol for E3-E6. All expensive work requires an explicit run call."""
from collections import deque
import copy
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
import paths
from common import dump,write_csv,sha256,inside
from control import Pack,SafetyController,ACTIONS
from study_control import ContinuousScreen,RulePolicy
from study_metrics import metrics,plot_case
from portable import project_path
from resources import profile_training


DEFAULTS=dict(seeds=[7,17,27,37,47],decision_s=90.,horizon_s=5400.,hold_s=270.,
    training_steps=60000,checkpoint_every_steps=10000,update_every_collected_steps=50,
    batch_size=32,max_sequence_length=80,learning_rate=2e-4,gamma_per_90s=.99,
    replay_episodes=800,target_sync_every_episodes=50,epsilon_start=.5,epsilon_end=.05,
    epsilon_decay_fraction=.8,cpu_threads=4,rollout_batch_steps=2048,ppo_epochs=10,
    minibatch_size=256,gae_lambda=.95,clip_ratio=.2,entropy_coef=.01,tau=.005,
    alpha=.2,gradient_steps=1,sac_batch_size=256,sac_gradient_steps=50,
    validation_count=8,test_count=24,validation_seed=390901,
    test_seed=390902,bootstrap_seed=390903,bootstrap_replicates=2000,
    voltage_margin_V=.03,temperature_margin_K=1.,soc_max=.95,nominal_ambient_K=298.15,
    plant_sample_dt_s=1.,numerical_failure_penalty=2000.,
    rule_gain_A_per_soc=20.,rule_base_A=3.,cc_current_A=7.5,cv_gain_A_per_V=20.,
    rule_terminal_soc_target=.93,rule_terminal_taper_A_per_soc=75.,rule_terminal_min_request_A=.5,
    compress_trajectories=True,replicate_training_triplet=False,device='cpu',
    reward_mode='raw',reward_weights={'time':.75,'balance':50.,'voltage':20.,'temperature':2.},
    reward_normalizers={'time':1.,'balance':.1,'voltage':.1,'temperature':10.})


def resolved_cfg(cfg):
    result=copy.deepcopy(DEFAULTS)
    for key,value in cfg.items():
        if key in ('reward_weights','reward_normalizers'):
            result[key].update(value)
        else:result[key]=copy.deepcopy(value)
    result['gamma']=result['gamma_per_90s']**(result['decision_s']/90.)
    for key in ['training_steps','checkpoint_every_steps','update_every_collected_steps','batch_size',
                'max_sequence_length','replay_episodes','target_sync_every_episodes','rollout_batch_steps',
                'bootstrap_replicates','cpu_threads']:
        if not isinstance(result[key],int) or isinstance(result[key],bool) or result[key]<1:
            raise ValueError(f'{key} must be a positive integer')
    if not result['seeds'] or len(set(result['seeds']))!=len(result['seeds']):
        raise ValueError('Distinct nonempty seeds required')
    if any(not isinstance(s,int) or isinstance(s,bool) or not 0<=s<2**32 for s in result['seeds']):
        raise ValueError('Seeds must be uint32-compatible integers')
    if result['validation_seed']==result['test_seed']:
        raise ValueError('Validation and test scenario seeds must differ')
    dt=float(result['decision_s'])
    if not math.isfinite(dt) or dt<=0 or result['horizon_s']<=0 or not float(result['horizon_s']/dt).is_integer() or not float(result['hold_s']/dt).is_integer():
        raise ValueError('Decision period must divide both physical horizon and holding duration')
    if result['hold_s']<270 or not .9<result['soc_max']<1 or not 0<result['plant_sample_dt_s']<=1:
        raise ValueError('Protocol requires hold>=270s, .9<SOCmax<1, dense sample<=1s')
    if not 0<=result['epsilon_end']<=result['epsilon_start']<=1 or not 0<result['epsilon_decay_fraction']<=1:
        raise ValueError('Invalid exploration schedule')
    if result['reward_mode'] not in ('raw','normalized_unit'):
        raise ValueError('Unknown reward_mode')
    for key in ('learning_rate','numerical_failure_penalty','voltage_margin_V','temperature_margin_K'):
        if not math.isfinite(float(result[key])) or result[key]<=0:raise ValueError(f'{key} must be finite and positive')
    if not 0<result['gamma_per_90s']<=1:raise ValueError('Invalid physical-time discount factor')
    for key in ('validation_count','test_count'):
        if not isinstance(result[key],int) or isinstance(result[key],bool) or result[key]<1:raise ValueError(f'{key} must be a positive integer')
    if any(not math.isfinite(float(v)) or v<0 for v in result['reward_weights'].values()):
        raise ValueError('Reward magnitudes must be finite/nonnegative')
    if any(not math.isfinite(float(v)) or v<=0 for v in result['reward_normalizers'].values()):
        raise ValueError('Reward normalizers must be finite/positive')
    return result


def code_fingerprint():
    files=sorted(paths.HERE.glob('*.py'))
    files += [paths.BASE/name for name in ['control.py','model.py','original_parameters.json']]
    return {p.relative_to(paths.ROOT).as_posix():sha256(p) for p in files}


def _completed(root,number,cfg):
    base=root/'revision_results'/f'experiment_{number}'
    tag=cfg.get('upstream_result_tags',{}).get(str(number),'')
    if tag:
        if not tag.isascii() or not all(c.isalnum() or c=='_' for c in tag):raise ValueError('Invalid upstream tag')
        base=root/'revision_results'/f'experiment_{number}_{tag}'
    explicit=cfg.get('upstream_results',{}).get(str(number))
    if explicit:folder=inside(project_path(explicit),base)
    else:
        complete=base/'LATEST_COMPLETED.txt';started=base/'LATEST_STARTED.txt'
        if not complete.exists():raise FileNotFoundError(f'Complete experiment {number} first; {complete} is missing.')
        if started.exists() and started.read_text(encoding='utf-8-sig').strip()!=complete.read_text(encoding='utf-8-sig').strip():
            raise ValueError(f'Latest experiment {number} did not finish; inspect it or explicitly select an older completed run in upstream_results.')
        folder=inside(project_path(complete.read_text(encoding='utf-8-sig').strip()),base)
    status=json.loads((folder/'status.json').read_text(encoding='utf-8-sig'))
    if status.get('status')!='completed':raise ValueError(f'Experiment {number} has not completed')
    return folder


def load_chain(root,out,cfg,required=(1,2)):
    subset_assessment=None
    upstreams={int(number):_completed(root,int(number),cfg) for number in required
               if not (int(number)==2 and cfg.get('upstream_e2_slice_manifest'))}
    if cfg.get('upstream_e2_slice_manifest'):
        from e2_slice import audit
        upstreams[2],subset_assessment=audit(root,out,cfg)
    one=upstreams[1]
    h=json.loads((one/'handoff.json').read_text(encoding='utf-8-sig'))
    ready=h.get('readiness',{})
    if ready.get('status')!='READY_FOR_EXPLORATORY_SIMULATION' or not ready.get('numerical_precision_passed') or not ready.get('model_screening_passed'):
        raise ValueError('E1 handoff did not pass numerical/model screening')
    for name,digest in h['artifact_hashes'].items():
        if sha256(inside(one/name,one))!=digest:raise ValueError('Modified E1 artifact: '+name)
    for name in ('model.py','original_parameters.json'):
        if sha256(paths.BASE/name)!=h['code_hashes'][name]:raise ValueError('E1 model changed; repeat E1/E2 before new experiments')
    e2=upstreams[2]
    if sha256(e2/'source_snapshot'/'control.py')!=sha256(paths.BASE/'control.py'):
        raise ValueError('E2 safety/control implementation changed; restore its source or repeat E2')
    source=json.loads((e2/'model_handoff_used.json').read_text(encoding='utf-8-sig'))
    if project_path(source['source'])!=one.resolve() or source['sha256']!=sha256(one/'handoff.json'):
        raise ValueError('Selected E1 and E2 do not share the same calibration')
    assessment=subset_assessment or json.loads((e2/'scientific_assessment.json').read_text(encoding='utf-8-sig'))
    if not assessment.get('any_safe_task_observed',False):
        raise ValueError('E2 has no observed safe completed task. Resolve model/control failures before formal E3-E6 runs.')
    current_fingerprint=code_fingerprint()
    for number,folder in upstreams.items():
        if number<3:continue
        lineage=json.loads((folder/'lineage.json').read_text(encoding='utf-8-sig'))
        if lineage['e1_handoff_sha256']!=sha256(one/'handoff.json'):
            raise ValueError(f'Experiment {number} used a different calibrated model')
        if lineage['code_fingerprint']!=current_fingerprint:
            raise ValueError(f'Experiment {number} used a different control/algorithm implementation. Recompute affected studies or restore its source snapshot.')
        for dependency,path in lineage['upstreams'].items():
            dep=int(dependency)
            if dep in upstreams and project_path(path)!=upstreams[dep].resolve():
                raise ValueError(f'Lineage mismatch: experiment {number} used another experiment {dep} run')
    base_cfg=json.loads((e2/'config.json').read_text(encoding='utf-8-sig'))
    base_cfg.update(cfg);base_cfg=resolved_cfg(base_cfg)
    torch.set_num_threads(base_cfg['cpu_threads'])
    selected_device=torch.device(base_cfg['device'])
    if selected_device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; this run will not silently fall back to CPU')
    device_info={'requested_device':str(selected_device),'torch':torch.__version__,
                 'torch_cuda':torch.version.cuda,'cpu_threads':torch.get_num_threads(),
                 'plant_and_safety_device':'cpu','mixed_precision':False,
                 'parallel_gpu_slots':json.loads(os.environ.get('REVISION_GPU_SLOTS','[]')),
                 'gpu_selection':json.loads(os.environ.get('REVISION_GPU_SELECTION','[]'))}
    if selected_device.type=='cuda':
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        props=torch.cuda.get_device_properties(selected_device)
        device_info.update(gpu_name=props.name,total_memory_bytes=props.total_memory,
                           compute_capability=list(torch.cuda.get_device_capability(selected_device)))
    dump(out/'device_info.json',device_info)
    print('LEARNING DEVICE:',base_cfg['device'],device_info.get('gpu_name','CPU'),flush=True)
    lineage={'upstreams':{str(k):str(v) for k,v in upstreams.items()},
        'e1_handoff_sha256':sha256(one/'handoff.json'),'code_fingerprint':current_fingerprint,
        'e2_readiness':'at least one safe task observed; this is a computation gate, not proof of convergence or universal safety',
        'scope':'numerical-only, with inherited E1 model and metadata limitations',
        'e2_evidence_kind':'completed_90s_subset' if subset_assessment else 'complete_run',
        'exploratory':bool(cfg.get('exploratory',False))}
    dump(out/'lineage.json',lineage);dump(out/'effective_config.json',base_cfg)
    dump(out/'e1_model_used.json',h)
    registry=json.loads((upstreams[3]/'model_registry.json').read_text(encoding='utf-8-sig')) if 3 in upstreams else []
    return {'vectors':np.asarray(h['parameter_vectors'],dtype=float),'base_cfg':base_cfg,
            'upstreams':upstreams,'registry':registry,'handoff':h}


def make_policy(spec,n_agents,seed,cfg):
    effective=resolved_cfg({**cfg,**{k:v for k,v in spec.items() if k not in ('id','method')}})
    method=spec['method']
    if method in ('cccv','cccv_continuous','soc_rule'):
        return RulePolicy(method,n_agents,seed,effective)
    if method in ('mappo','mappo_continuous','sac'):
        from actor_agents import ActorPolicy
        return ActorPolicy(method,n_agents,seed,effective)
    from value_agents import ValuePolicy
    return ValuePolicy(method,n_agents,seed,effective)


def load_policy(entry,device=None):
    spec=copy.deepcopy(entry['spec']);config=copy.deepcopy(entry['config'])
    if device is not None:config['device']=device;spec['device']=device
    policy=make_policy(spec,entry['n_agents'],entry['seed'],config)
    if entry['checkpoint']:
        path=project_path(entry['checkpoint'])
        if sha256(path)!=entry['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch: '+str(path))
        policy.load(path)
    return policy


def make_cases(vectors,cfg,seed,count,prefix):
    rng=np.random.default_rng(seed);n=len(vectors)
    cases=[]
    for i in range(count):
        initial=rng.uniform(.1,.75,n)
        cases.append({'case_id':f'{prefix}_{i:03d}','initial_soc':initial.tolist(),
            'plant_vectors':np.asarray(vectors).tolist(),'test_family':'random_nominal',
            'ambient_K':cfg.get('nominal_ambient_K',298.15),'observation_seed':int(seed)+i+500000})
    return cases


def _reward(terms,cfg):
    scales={'time':.75,'balance':50.,'voltage':20.,'temperature':2.}
    total=sum(float(terms.get(key,0.)) for key in ('numerical_failure','protective_stop','failed_screen','observer_failure'))
    for key,scale in scales.items():
        divisor=cfg['reward_normalizers'][key] if cfg['reward_mode']=='normalized_unit' else 1.
        total+=float(terms.get(key,0.))/scale*cfg['reward_weights'][key]/divisor
    return float(total)


def _mask(observed,soc_max=.95):
    mask=np.ones((len(observed),16),dtype=bool)
    for i,(soc,v,temp) in enumerate(observed):
        if soc>=soc_max:mask[i,1:]=False
        elif v>=4.2 or temp>=309:mask[i,ACTIONS>1.]=False
    return mask


def _initial_rows(pack,phase):
    return [dict(time_s=float(pack.time),cell=i+1,phase=phase,soc=float(s[0]),voltage_V=float(s[1]),
                 temperature_K=float(s[2]),current_A=0.,decision_index=-1) for i,s in enumerate(pack.physical())]


def _dense(result,phase,index):
    rows=[]
    for i,s in enumerate(result['stats']):
        for t,z,v,k in zip(s['time_s'],s['soc'],s['voltage_V'],s['temperature_K']):
            rows.append(dict(time_s=float(t),cell=i+1,phase=phase,soc=float(z),voltage_V=float(v),
                             temperature_K=float(k),current_A=float(s['applied_current_A']),decision_index=index))
    return sorted(rows,key=lambda r:(r['time_s'],r['cell']))


def rollout(policy,vectors,case,cfg,training=False,epsilon=0.,max_actions=None,hold=True):
    begun=time.perf_counter();dt=cfg['decision_s'];n=len(vectors)
    pack=Pack(case['plant_vectors'],case['initial_soc'],ambient_K=case.get('ambient_K',cfg['nominal_ambient_K']),decision_s=dt,
        model_options={'rtol':1e-7,'atol':1e-9,'sample_dt_s':cfg['plant_sample_dt_s'],
            'numerical_failure_penalty':-cfg['numerical_failure_penalty'],'physical_interlock':True,'physical_soc_max':cfg['soc_max']})
    screen_class=ContinuousScreen if policy.action_kind=='continuous' else SafetyController
    screen=screen_class(vectors,case['initial_soc'],ambient_K=case.get('ambient_K',cfg['nominal_ambient_K']),decision_s=dt,
        voltage_margin_V=cfg['voltage_margin_V'],temperature_margin_K=cfg['temperature_margin_K'],soc_max=cfg['soc_max'])
    policy.reset();rows=_initial_rows(pack,'charge');decisions=[];goal=None;failure=''
    noise_seed=case.get('observation_seed',int(hashlib.sha256(case['case_id'].encode()).hexdigest()[:8],16))
    obs_rng=np.random.default_rng(noise_seed);observation_history=[]
    noise=np.asarray(case.get('observation_noise',[0.,0.,0.]),dtype=float)
    bias=np.asarray(case.get('observation_bias',[0.,0.,0.]),dtype=float)
    delay=int(case.get('observation_delay_steps',0))
    if noise.shape!=(3,) or bias.shape!=(3,) or np.any(noise<0) or delay<0:
        raise ValueError('Invalid observation disturbance')
    def observe():
        measured=pack.physical()+bias+obs_rng.normal(0.,noise,(n,3))
        observation_history.append(measured.copy())
        observed=observation_history[max(0,len(observation_history)-1-delay)]
        return observed,((observed-[.5,3.5,308.])/[.5,1.,11.]).astype(np.float32),_mask(observed,cfg['soc_max'])
    measured,obs,mask=observe()
    ep=dict(obs=[obs],masks=[mask],actions=[],executed_actions=[],rewards=[],done=[],info=[])
    policy_s=safety_s=plant_s=0.;setup_s=time.perf_counter()-begun
    limit=int(round(cfg['horizon_s']/dt))
    if max_actions is not None:limit=min(limit,int(max_actions))
    for k in range(limit):
        decision_observation=measured.copy()
        tick=time.perf_counter();requested,info=policy.act(obs,mask,training=training,epsilon=epsilon);policy_s+=time.perf_counter()-tick
        requested=np.asarray(requested,dtype=float)
        if requested.shape!=(n,) or not np.isfinite(requested).all() or np.any(requested<0) or np.any(requested>7.5+1e-7):
            raise ValueError('Policy returned an invalid current vector')
        limited=requested.copy()
        if pack.time>=case.get('power_drop_at_s',float('inf')):
            cap=7.5*float(case.get('current_cap_factor_after',1.))
            if not 0<=cap<=7.5:raise ValueError('Invalid current cap')
            limited=np.minimum(limited,cap)
            if policy.action_kind=='discrete':limited=np.floor(limited/.5+1e-10)*.5
        tick=time.perf_counter();applied,details=screen.filter(limited,measured);safety_s+=time.perf_counter()-tick
        result=None
        if details['infeasible']:
            actual=np.zeros(n);duration=0.;failure='safety_screen_infeasible'
            terms={'failed_screen':-cfg['numerical_failure_penalty']}
        else:
            tick=time.perf_counter();result=pack.step(applied);plant_s+=time.perf_counter()-tick
            rows.extend(_dense(result,'charge',k));duration=result['interval_duration_s']
            terms=dict(result['reward_terms']);actual=np.array([r['applied_current_A'] for r in result['stats']])
            if result['failure']:failure=result['failure_reason']
            if duration>0:
                tick=time.perf_counter();advanced=screen.advance(actual,duration);safety_s+=time.perf_counter()-tick
                if advanced['failure'] and not failure:
                    failure='nominal_observer_failure: '+advanced['failure_reason'];terms['observer_failure']=-cfg['numerical_failure_penalty']
            if result['goal'] and not failure:goal=float(pack.time)
        reward=_reward(terms,cfg);terminal=bool(failure or goal is not None)
        policy.set_executed(actual)
        ep['actions'].append(requested);ep['executed_actions'].append(actual);ep['rewards'].append(reward)
        ep['done'].append(terminal);ep['info'].append(info)
        measured,nextobs,nextmask=observe()
        ep['obs'].append(nextobs);ep['masks'].append(nextmask)
        decisions.append(dict(decision_index=k,phase='charge',time_s=pack.time,duration_s=duration,
            requested_current_A=requested.tolist(),limited_current_A=limited.tolist(),executed_current_A=actual.tolist(),
            observed_soc_v_t=decision_observation.tolist(),next_observed_soc_v_t=measured.tolist(),policy_info=info,reward=reward,reward_terms=terms,
            intervention=bool(details.get('intervention') or not np.allclose(limited,requested)),
            protective_stop=bool(result and result.get('protective_stop')),numerical_failure=bool(result and result.get('numerical_failure')),
            post_switch_peak_unknown=bool(result and result.get('post_switch_peak_unknown')),failure_reason=failure,safety_details=details))
        obs,mask=nextobs,nextmask
        if terminal:break
    if hold and goal is not None and not failure:
        rows.extend(_initial_rows(pack,'hold'))
        for k in range(int(round(cfg['hold_s']/dt))):
            tick=time.perf_counter();result=pack.step(np.zeros(n));plant_s+=time.perf_counter()-tick
            rows.extend(_dense(result,'hold',len(decisions)))
            decisions.append(dict(decision_index=len(decisions),phase='hold',time_s=pack.time,
                duration_s=result['interval_duration_s'],requested_current_A=[0.]*n,executed_current_A=[0.]*n,
                intervention=False,protective_stop=result.get('protective_stop',False),numerical_failure=result.get('numerical_failure',False),
                post_switch_peak_unknown=result.get('post_switch_peak_unknown',False),failure_reason=result['failure_reason']))
            if result['failure']:failure='hold: '+result['failure_reason'];break
    scores=metrics(rows,decisions,goal,failure,cfg['soc_max'])
    if hold and scores['terminal_hold_observed_s']<cfg['hold_s']-1e-5:scores['task_success']=False
    scores.update(timing_policy_s=policy_s,timing_safety_s=safety_s,timing_plant_s=plant_s,
                  timing_setup_s=setup_s,timing_total_s=time.perf_counter()-begun)
    return ep,rows,decisions,scores


def _validate_policy(policy,vectors,cfg,cases):
    records=[]
    for case in cases:
        _,_,_,score=rollout(policy,vectors,case,cfg,training=False,hold=True)
        records.append({'case_id':case['case_id'],**score})
    rank=(float(np.mean([r['task_success'] for r in records])),
          float(np.mean([r['constraints_satisfied'] for r in records])),
          -float(np.mean([r['charging_time_s'] if r['task_success'] else cfg['horizon_s']+cfg['hold_s'] for r in records])))
    return records,rank


@profile_training
def train_policy(spec,vectors,cfg,out,validation_cases,seed):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    effective=resolved_cfg({**cfg,**{k:v for k,v in spec.items() if k not in ('id','method')}})
    policy=make_policy(spec,len(vectors),seed,effective)
    print(f'TRAIN {spec["id"]} N={len(vectors)} seed={seed} steps={effective["training_steps"]} device={getattr(policy,"device","rule/CPU")}',flush=True)
    entry={'spec':copy.deepcopy(spec),'seed':seed,'n_agents':len(vectors),'method_id':spec['id'],
        'method':spec['method'],'action_kind':policy.action_kind,'learning_kind':policy.learning_kind,
        'config':effective,'checkpoint':None,'checkpoint_sha256':None,'parameter_count':policy.parameter_count(),
        'learning_device':str(getattr(policy,'device','cpu'))}
    if policy.learning_kind=='rule':
        _,rank=_validate_policy(policy,vectors,effective,validation_cases)
        entry['validation_rank']=list(rank);dump(out/'entry.json',entry);return entry
    replay=deque(maxlen=effective['replay_episodes']);on_policy=[];pending=0;steps=episode=0
    next_checkpoint=effective['checkpoint_every_steps'];best_rank=None;history=[];vhistory=[]
    # Distinct RNGs keep scenario distribution independent of replay sampling.
    scenario_rng=np.random.default_rng(seed+701000);replay_rng=np.random.default_rng(seed+702000)
    start=time.perf_counter()
    while steps<effective['training_steps']:
        random_size=3 if effective.get('replicate_training_triplet',False) else len(vectors)
        initial=np.resize([.1,.2,.3],len(vectors)) if episode%4==0 else np.resize([.3,.5,.7],len(vectors)) if episode%4==1 else np.resize(scenario_rng.uniform(.1,.75,random_size),len(vectors))
        plant=np.asarray(vectors).copy();ambient=effective['nominal_ambient_K']
        case={'case_id':f'train_{episode:06d}','initial_soc':initial.tolist(),'plant_vectors':plant.tolist(),
              'ambient_K':ambient,'test_family':'training','observation_seed':int(scenario_rng.integers(0,2**32-1))}
        if effective.get('domain_randomization',False):
            plant[:,[0,1,6,7]]*=scenario_rng.uniform(.95,1.05,(len(vectors),4))
            case.update(plant_vectors=plant.tolist(),ambient_K=float(scenario_rng.uniform(288.15,303.15)),observation_noise=[.01,.005,.5])
        frac=min(1.,steps/(effective['training_steps']*effective['epsilon_decay_fraction']))
        epsilon=effective['epsilon_start']+(effective['epsilon_end']-effective['epsilon_start'])*frac
        ep,_,_,score=rollout(policy,vectors,case,effective,training=True,epsilon=epsilon,
                            max_actions=effective['training_steps']-steps,hold=False)
        collected=len(ep['actions'])
        if collected<1:raise RuntimeError('Empty training episode')
        steps+=collected;episode+=1;pending+=collected;loss={}
        if policy.learning_kind=='on_policy':
            on_policy.append(ep)
            if pending>=effective['rollout_batch_steps'] or steps==effective['training_steps']:
                loss=policy.learn(on_policy);on_policy=[];pending=0
        else:
            replay.append(ep)
            while pending>=effective['update_every_collected_steps'] and len(replay)>=5:
                ids=replay_rng.choice(len(replay),min(len(replay),effective['batch_size']),replace=False)
                loss=policy.learn([replay[int(i)] for i in ids]);pending-=effective['update_every_collected_steps']
        if episode%effective['target_sync_every_episodes']==0:policy.sync()
        history.append({'episode':episode,'environment_steps':steps,'agent_decisions':steps*len(vectors),
            'reward':sum(ep['rewards']),'epsilon_for_value_agents':epsilon,'loss':loss,
            'reward_penalty_activation_steps':score['reward_penalty_activation_steps'],
            'target_reached':score['charging_target_reached'],'constraints_satisfied':score['constraints_satisfied'],
            'failure_reason':score['failure_reason'],'wall_s':time.perf_counter()-start})
        write_csv(out/'training.csv',history)
        if episode%10==0:print(f'{spec["id"]} N={len(vectors)} seed={seed} {steps}/{effective["training_steps"]}',flush=True)
        if steps>=next_checkpoint or steps==effective['training_steps']:
            policy.save(out/f'steps_{steps}.pt',{'steps':steps,'spec':spec,'seed':seed})
            records,rank=_validate_policy(policy,vectors,effective,validation_cases)
            vhistory.append({'steps':steps,'success_rate':rank[0],'constraint_rate':rank[1],'negative_capped_time':rank[2]})
            write_csv(out/'validation.csv',vhistory);write_csv(out/f'validation_{steps}.csv',records)
            if best_rank is None or rank>best_rank:
                best_rank=rank;policy.save(out/'selected.pt',{'steps':steps,'rank':rank,'spec':spec,'seed':seed})
            while next_checkpoint<=steps:next_checkpoint+=effective['checkpoint_every_steps']
    policy.save(out/'last.pt',{'steps':steps,'spec':spec,'seed':seed})
    entry.update(checkpoint=str((out/'selected.pt').resolve()),checkpoint_sha256=sha256(out/'selected.pt'),
        validation_rank=list(best_rank),training_wall_s=time.perf_counter()-start,environment_steps=steps,
        episodes=episode,exact_resumption_supported=False)
    dump(out/'entry.json',entry)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].plot([r['environment_steps'] for r in history],[r['reward'] for r in history])
    axes[0].set(xlabel='Pack decisions',ylabel='Episode reward')
    axes[1].plot([r['steps'] for r in vhistory],[r['success_rate'] for r in vhistory],marker='o');axes[1].set(xlabel='Pack decisions',ylabel='Validation safe success')
    fig.tight_layout();fig.savefig(out/'learning.png',dpi=150);plt.close(fig)
    return entry


def _save_trajectory(target,trace,compress):
    if not compress:
        write_csv(target/'trajectory.csv',trace);return
    with gzip.open(target/'trajectory.csv.gz','wt',encoding='utf-8-sig',newline='',compresslevel=5) as file:
        writer=csv.DictWriter(file,fieldnames=list(trace[0]) if trace else [])
        if trace:writer.writeheader();writer.writerows(trace)


def evaluate_policy(entry,vectors,cfg,cases,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Retain each method's declared reward/architecture settings in evaluation logs.
    # Disturbances belong to case definitions, not silent policy reconfiguration.
    effective=resolved_cfg(entry['config']);requested=resolved_cfg(cfg)
    for key in ['decision_s','voltage_margin_V','temperature_margin_K','soc_max','horizon_s','hold_s']:
        if requested[key]!=entry['config'][key]:
            raise ValueError(f'Frozen evaluation cannot silently alter core constraint/protocol {key}')
    policy=load_policy(entry);records=[]
    for case in cases:
        target=out/case['case_id'];target.mkdir(parents=True,exist_ok=True)
        _,trace,decisions,score=rollout(policy,vectors,case,effective,training=False,hold=True)
        row={'method_id':entry['method_id'],'method':entry['method'],'seed':entry['seed'],
             'case_id':case['case_id'],'test_family':case.get('test_family','nominal'),
             'n_agents':len(vectors),'decision_s':effective['decision_s'],'action_kind':entry['action_kind'],**score}
        for field in ('study_panel','initial_condition_family','base_case_id','feasibility_negative_control'):
            if field in case:row[field]=case[field]
        records.append(row);_save_trajectory(target,trace,effective['compress_trajectories']);write_csv(target/'decisions.csv',decisions)
        dump(target/'metrics.json',row);dump(target/'case.json',case)
        if len(records)<=2:plot_case(trace,target/'trajectory.png',f'{entry["method_id"]} N={len(vectors)} {case["case_id"]}')
        write_csv(out/'metrics.csv',records)
    return records


def run_suite(specs,vectors,cfg,out,validation_cases,test_cases):
    from parallel_jobs import train_job,execute_jobs
    out=Path(out);out.mkdir(parents=True,exist_ok=True);cfg=resolved_cfg(cfg)
    dump(out/'validation_cases.json',validation_cases);dump(out/'test_cases.json',test_cases)
    registry=[];records=[];jobs=[]
    for spec in specs:
        seeds=[-1] if spec['method'] in ('cccv','cccv_continuous','soc_rule') else cfg['seeds']
        for seed in seeds:
            folder=out/spec['id']/f'seed_{seed}'
            jobs.append(train_job(spec,vectors,cfg,folder/'training',validation_cases,seed,test_cases,folder/'test'))
    for result in execute_jobs(jobs,out/'parallel_jobs',cfg):
        registry.append(result['entry']);records.extend(result['records'])
    dump(out/'model_registry.json',registry);write_csv(out/'episode_metrics.csv',records)
    return {'registry':registry,'records':records}


def finalize(out,records,extra=None):
    out=Path(out);extra=extra or {};write_csv(out/'episode_metrics.csv',records)
    config_file=out/'effective_config.json'
    cfg=json.loads(config_file.read_text(encoding='utf-8-sig')) if config_file.exists() else DEFAULTS
    groups=[];rng=np.random.default_rng(cfg['bootstrap_seed'])
    grouping=lambda r:(r['method_id'],r['n_agents'],r['action_kind'],r['test_family'],r.get('study_panel','main'),r.get('initial_condition_family','all'))
    for key in sorted(set(grouping(r) for r in records)):
        subset=[r for r in records if grouping(r)==key]
        seeds=sorted(set(r['seed'] for r in subset));rates=np.array([np.mean([x['task_success'] for x in subset if x['seed']==s]) for s in seeds])
        ci=[None,None]
        if len(seeds)>=2 and seeds!=[-1]:ci=np.quantile(rng.choice(rates,(cfg['bootstrap_replicates'],len(rates)),replace=True).mean(1),[.025,.975]).tolist()
        times=[r['charging_time_s'] for r in subset if r['task_success']]
        groups.append({'method_id':key[0],'n_agents':key[1],'action_kind':key[2],'test_family':key[3],
            'study_panel':key[4],'initial_condition_family':key[5],
            'training_seeds':0 if seeds==[-1] else len(seeds),'episodes':len(subset),'safe_success_rate':float(rates.mean()),
            'seed_bootstrap_ci_low':ci[0],'seed_bootstrap_ci_high':ci[1],
            'constraints_rate':float(np.mean([r['constraints_satisfied'] for r in subset])),
            'sampled_physical_limits_respected_rate':float(np.mean([r['sampled_physical_limits_respected'] for r in subset])),
            'failed_or_censored':sum(not r['task_success'] for r in subset),
            'successful_only_mean_charge_s':float(np.mean(times)) if times else None,
            'mean_failure_capped_charge_s':float(np.mean([r['charging_time_s'] if r['task_success'] else cfg['horizon_s']+cfg['hold_s'] for r in subset])),
            'mean_soc_std_integral':float(np.mean([r['soc_std_integral_soc_s'] for r in subset])),
            'mean_intervention_fraction':float(np.mean([r['safety_intervention_fraction'] for r in subset])),
            'note':'Conditional charging time is not a standalone ranking. Discrete and continuous tracks have different action authority.'})
    write_csv(out/'summary.csv',groups)
    assessment={'computation_finished':True,'episodes':len(records),'all_tasks_succeeded':bool(records) and all(r['task_success'] for r in records),
        'exploratory':bool(cfg.get('exploratory',False)),'training_steps_per_run':cfg['training_steps'],
        'formal_evidence':False if cfg.get('exploratory',False) else None,
        'all_sampled_constraints_satisfied':bool(records) and all(r['constraints_satisfied'] for r in records),
        'hardware_validated':False,'lifetime_improvement_proven':False,'general_convergence_proven':False,
        'interpretation':'Failing or unfavorable outcomes are retained. Completed calculations do not imply scientific superiority.',**extra}
    dump(out/'scientific_assessment.json',assessment)
    lines=['# 实验结果','', '计算完成，以下结果包含失败与超时。是否支持方法优势需根据成功率、配对差异、终端 SOC 和限制判断。','',
        '| 方法 | 电芯数 | 动作 | 面板 / 工况 / 初态组 | 安全成功率 | 失败或超时 |','|---|---:|---|---|---:|---:|']
    for g in groups:lines.append(f'| {g["method_id"]} | {g["n_agents"]} | {g["action_kind"]} | {g["study_panel"]} / {g["test_family"]} / {g["initial_condition_family"]} | {g["safe_success_rate"]:.3f} | {g["failed_or_censored"]} |')
    lines+=['','置信区间按训练种子重采样，条件于固定测试集，不是硬件安全概率保证。',
        '输入 Wh 不代表损耗或效率。先检查终端 SOC 是否可比，再讨论能量差异。',
        '当前仍是数值证据，不能替代物理在线控制、微控制器验证、寿命验证或一般收敛证明。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'summary':groups,'assessment':assessment}
