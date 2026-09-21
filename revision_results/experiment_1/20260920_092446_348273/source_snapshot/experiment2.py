"""E2 controlled safety and sustained-balancing experiment, started only by CLI."""
from collections import deque
import copy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from common import dump, write_csv, sha256, inside
from control import Pack, SafetyController, ACTIONS
from learner import Learner
from evaluation import metrics, summarize, plot_case


def _validate(cfg):
    integers = ['training_steps','checkpoint_every_steps','target_sync_every_episodes',
                'replay_episodes','batch_size','learning_sequence_steps','update_every_collected_steps',
                'bootstrap_replicates','cpu_threads']
    for key in integers:
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Training seeds must be nonempty and unique')
    if cfg['validation_seed']==cfg['test_seed']:
        raise ValueError('Validation and test random scenario seeds must differ')
    if len(cfg['seeds']) < 5:
        print('NOTICE: fewer than five seeds; output will be labelled exploratory.')
    for dt in cfg['decision_periods_s']:
        if dt <= 0 or not float(cfg['horizon_s']/dt).is_integer() or not float(cfg['hold_s']/dt).is_integer():
            raise ValueError('Decision periods must exactly divide horizon_s and hold_s')
    if not 0 < cfg['plant_sample_dt_s'] <= 1:
        raise ValueError('Dense constraint audit interval must be at most 1 second')
    if not 0 < cfg['gamma_per_90s'] <= 1:
        raise ValueError('gamma_per_90s must be in (0,1]')
    if not 0 <= cfg['epsilon_end'] <= cfg['epsilon_start'] <= 1:
        raise ValueError('Invalid epsilon schedule')
    if not 0 < cfg['epsilon_decay_fraction'] <= 1:
        raise ValueError('Invalid epsilon decay fraction')
    if not .9 <= cfg['soc_max'] < 1 or cfg['horizon_s'] <= 0 or cfg['hold_s'] < 270:
        raise ValueError('SOC upper bound/horizon/hold do not implement protocol')
    if not 0 < cfg['voltage_margin_V'] < .2 or not 0 < cfg['temperature_margin_K'] < 10:
        raise ValueError('Prediction margins must be positive and physically interpretable')
    if cfg['numerical_failure_penalty'] <= 0:
        raise ValueError('numerical_failure_penalty config is a positive magnitude')
    if not 0 <= cfg['parameter_shift_fraction'] <= .1:
        raise ValueError('Parameter shifts outside declared 0..10% protocol')
    for key in ['validation_random_cases','test_random_cases','test_parameter_shift_cases']:
        if not isinstance(cfg[key],int) or cfg[key] < 0:
            raise ValueError(f'{key} must be nonnegative integer')


def _handoff(root, out, cfg):
    base = root/'revision_results'/'experiment_1'
    if cfg.get('experiment1_result'):
        folder = inside(Path(cfg['experiment1_result']), base)
    else:
        pointer = base/'LATEST_COMPLETED.txt'
        if not pointer.exists():
            raise FileNotFoundError('Run run_experiment_1.cmd successfully first; no completed E1 result exists.')
        started=base/'LATEST_STARTED.txt'
        if started.exists() and started.read_text(encoding='utf-8-sig').strip()!=pointer.read_text(encoding='utf-8-sig').strip():
            raise ValueError('Latest E1 run did not complete. Inspect it first; to intentionally use an older completed run supply --experiment1-result explicitly.')
        folder = inside(Path(pointer.read_text(encoding='utf-8-sig').strip()), base)
    status = json.loads((folder/'status.json').read_text(encoding='utf-8-sig'))
    if status['status'] != 'completed':
        raise ValueError('Experiment 1 was not completed')
    handoff_path = folder/'handoff.json'
    handoff = json.loads(handoff_path.read_text(encoding='utf-8-sig'))
    if handoff.get('schema_version') != 1 or handoff.get('model_family') != 'SPMe':
        raise ValueError('Unsupported model handoff schema')
    if not handoff.get('quality',{}).get('numerical_precision_passed',False):
        raise ValueError('E1 numerical precision gate did not pass')
    ready = handoff.get('readiness',{})
    if ready.get('status') != 'READY_FOR_EXPLORATORY_SIMULATION' or not ready.get('model_screening_passed',False) or not ready.get('numerical_precision_passed',False):
        raise ValueError('E1 model screening gate did not pass')
    for name, digest in handoff.get('artifact_hashes',{}).items():
        if sha256(inside(folder/name,folder)) != digest:
            raise ValueError(f'Experiment 1 artifact changed since handoff: {name}')
    for name in ('model.py','original_parameters.json'):
        recorded=handoff.get('code_hashes',{}).get(name)
        if not recorded or sha256(Path(__file__).parent/name)!=recorded:
            raise ValueError(f'Model source changed after E1 calibration: {name}; repeat E1.')
    vectors = np.asarray(handoff['parameter_vectors'],dtype=float)
    if vectors.shape != (3,8) or not np.isfinite(vectors).all():
        raise ValueError('Handoff must contain three finite 8-parameter vectors')
    dump(out/'model_handoff_used.json',{'source':str(folder),'sha256':sha256(handoff_path),'handoff':handoff})
    print('USING EXPERIMENT 1:',folder,flush=True)
    return vectors,handoff


def _case(case_id, soc, vectors, family='nominal'):
    return dict(case_id=case_id,initial_soc=[float(x) for x in soc],
                plant_vectors=np.asarray(vectors).tolist(),test_family=family)


def _cases(vectors,cfg):
    validation=[_case('validation_A',[.1,.2,.3],vectors),_case('validation_B',[.3,.5,.7],vectors)]
    rng=np.random.default_rng(cfg['validation_seed'])
    validation += [_case(f'validation_random_{i:03d}',rng.uniform(.1,.75,3),vectors)
                   for i in range(cfg['validation_random_cases'])]
    tests=[_case('A',[.1,.2,.3],vectors,'reference_seen'),_case('B',[.3,.5,.7],vectors,'reference_seen')]
    rng=np.random.default_rng(cfg['test_seed'])
    for i in range(cfg['test_random_cases']):
        tests.append(_case(f'random_{i:03d}',rng.uniform(.1,.75,3),vectors,'nominal_unseen'))
    for i in range(cfg['test_parameter_shift_cases']):
        shifted=vectors.copy()
        # Diffusivities and thermal parameters only: avoid changing lithium
        # inventory/geometry inconsistently. These are synthetic model errors.
        for index in (0,1,6,7):
            shifted[:,index]*=rng.uniform(1-cfg['parameter_shift_fraction'],1+cfg['parameter_shift_fraction'],3)
        tests.append(_case(f'model_shift_{i:03d}',rng.uniform(.1,.75,3),shifted,'unseen_parameter_shift'))
    return validation,tests


def _rule(states,cfg):
    soc=states[:,0]
    current=cfg['rule_base_A']+cfg['rule_gain_A_per_soc']*(float(soc.max())-soc)
    current=np.clip(current,0,7.5)
    current[soc>=.9]=0.
    return np.floor(current/.5+1e-10)*.5


def _initial_rows(pack,phase='charge'):
    return [dict(time_s=float(pack.time),cell=i+1,phase=phase,soc=float(s[0]),
                 voltage_V=float(s[1]),temperature_K=float(s[2]),current_A=0.,decision_index=-1)
            for i,s in enumerate(pack.physical())]


def _dense_rows(result,phase,decision_index):
    rows=[]
    for i,stat in enumerate(result['stats']):
        arrays=[np.asarray(stat[key],dtype=float) for key in ['time_s','soc','voltage_V','temperature_K']]
        if len({len(a) for a in arrays}) != 1:
            raise ValueError('Malformed dense cell trace')
        for t,s,v,temp in zip(*arrays):
            rows.append(dict(time_s=float(t),cell=i+1,phase=phase,soc=float(s),
                             voltage_V=float(v),temperature_K=float(temp),
                             current_A=float(stat['applied_current_A']),decision_index=decision_index))
    # Stable order preserves pre/post switch values at duplicate timestamps.
    return sorted(rows,key=lambda r:(r['time_s'],r['cell']))


def rollout(vectors,case,cfg,dt,learner=None,shield=False,epsilon=0.,max_actions=None,hold=False):
    options={'rtol':1e-7,'atol':1e-9,'sample_dt_s':cfg['plant_sample_dt_s'],
             'numerical_failure_penalty':-float(cfg['numerical_failure_penalty']),
             'physical_interlock':bool(shield),'physical_soc_max':cfg['soc_max']}
    pack=Pack(case['plant_vectors'],case['initial_soc'],ambient_K=cfg['nominal_ambient_K'],decision_s=dt,model_options=options)
    safety=SafetyController(vectors,case['initial_soc'],ambient_K=cfg['nominal_ambient_K'],decision_s=dt,
        voltage_margin_V=cfg['voltage_margin_V'],temperature_margin_K=cfg['temperature_margin_K'],soc_max=cfg['soc_max']) if shield else None
    if learner is not None:
        learner.reset()
    ep={'obs':[pack.obs()],'masks':[pack.mask()],'actions':[],'executed_actions':[],'rewards':[],'done':[]}
    rows=_initial_rows(pack); decisions=[];goal_time=None;failure=''
    limit=int(round(cfg['horizon_s']/dt))
    if max_actions is not None:
        limit=min(limit,int(max_actions))
    for k in range(limit):
        before=pack.physical().copy()
        if learner is None:
            requested=_rule(before,cfg)
            indices=np.rint(requested/.5).astype(np.int64)
        else:
            indices=learner.choose(pack.obs(),pack.mask(),epsilon)
            requested=ACTIONS[indices]
        detail={'intervention':False,'infeasible':False,'reason':'unfiltered diagnostic policy'}
        begin=time.perf_counter()
        applied=np.array(requested,copy=True)
        if safety is not None:
            applied,detail=safety.filter(requested,before)
        filter_s=time.perf_counter()-begin
        if detail.get('infeasible',False):
            applied=np.zeros(3)
            failure='safety_screen_infeasible: '+str(detail.get('reason',''))
            reward=-float(cfg['numerical_failure_penalty'])
            duration=0.;done=True
            result=None
        else:
            result=pack.step(applied)
            duration=float(result['interval_duration_s']);reward=float(result['reward'])
            failure=result['failure_reason'] if result['failure'] else ''
            rows.extend(_dense_rows(result,'charge',k))
            if safety is not None and duration>0:
                observed=safety.advance(applied,duration)
                if observed['failure'] and not failure:
                    failure='nominal_observer_failure: '+observed['failure_reason']
                    reward-=float(cfg['numerical_failure_penalty'])
            if result['goal'] and not failure:
                goal_time=float(pack.time)
            done=bool(failure or goal_time is not None)
        actual=np.array([stat['applied_current_A'] for stat in result['stats']]) if result else np.zeros(3)
        executed=np.rint(actual/.5).astype(np.int64)
        if not np.allclose(executed*.5,actual,atol=1e-8):
            raise ValueError('E2 discrete action safety screen returned a non-grid action')
        if learner is not None:
            learner.set_executed(executed)
        ep['actions'].append(indices);ep['executed_actions'].append(executed)
        ep['rewards'].append(reward);ep['done'].append(done)
        ep['obs'].append(pack.obs());ep['masks'].append(pack.mask())
        decisions.append(dict(decision_index=k,phase='charge',time_s=float(pack.time),duration_s=duration,
            requested_current_A=np.asarray(requested).tolist(),screened_current_A=np.asarray(applied).tolist(),executed_current_A=actual.tolist(),
            reward=reward,goal=goal_time is not None,failure_reason=failure,
            protective_stop=bool(result.get('protective_stop',False)) if result else False,
            numerical_failure=bool(result.get('numerical_failure',False)) if result else False,
            post_switch_peak_unknown=bool(result.get('post_switch_peak_unknown',False)) if result else False,
            intervention=bool(detail.get('intervention',False)),filter_wall_s=filter_s,
            safety_details=detail,reward_terms=result.get('reward_terms',{}) if result else {'failed_screen':reward}))
        if done:
            break
    if hold and goal_time is not None and not failure:
        rows.extend(_initial_rows(pack,'hold'))
        for k in range(int(round(cfg['hold_s']/dt))):
            index=len(decisions)
            result=pack.step(np.zeros(3))
            rows.extend(_dense_rows(result,'hold',index))
            decisions.append(dict(decision_index=index,phase='hold',time_s=float(pack.time),
                duration_s=float(result['interval_duration_s']),requested_current_A=[0.,0.,0.],
                screened_current_A=[0.,0.,0.],executed_current_A=[float(s['applied_current_A']) for s in result['stats']],
                reward=float(result['reward']),goal=bool(result['goal']),failure_reason=result['failure_reason'],
                intervention=False,filter_wall_s=0.,safety_details={'reason':'zero-current terminal observation with common interlock'},
                protective_stop=bool(result.get('protective_stop',False)),numerical_failure=bool(result.get('numerical_failure',False)),
                post_switch_peak_unknown=bool(result.get('post_switch_peak_unknown',False)),reward_terms=result.get('reward_terms',{})))
            if result['failure']:
                failure='terminal_hold_failure: '+result['failure_reason']
                break
    result=metrics(rows,decisions,goal_time,failure,cfg['soc_max'])
    if hold and result.get('terminal_hold_observed_s',0)<cfg['hold_s']-1e-5:
        result['task_success']=False
    if not hold:
        # Training reports charge-only results, never aliases these to full test success.
        result['charge_phase_success']=bool(goal_time is not None and result['constraints_satisfied'])
    return ep,rows,decisions,result


def _validation(vectors,cases,cfg,dt,learner,shield):
    rows=[]
    for case in cases:
        _,_,_,result=rollout(vectors,case,cfg,dt,learner,shield,hold=True)
        rows.append({'case_id':case['case_id'],**result})
    rate=float(np.mean([r['task_success'] for r in rows]))
    constraints=float(np.mean([r['constraints_satisfied'] for r in rows]))
    capped=float(np.mean([r['charging_time_s'] if r['task_success'] else cfg['horizon_s']+cfg['hold_s'] for r in rows]))
    return rows,(rate,constraints,-capped)


def _train(vectors,validation,cfg,dt,seed,shield,directory):
    directory.mkdir(parents=True,exist_ok=True)
    gamma=cfg['gamma_per_90s']**(dt/90.)
    learner=Learner(seed,learning_rate=cfg['learning_rate'],gamma=gamma)
    buffer=deque(maxlen=cfg['replay_episodes'])
    rng=np.random.default_rng(seed+800000)
    history=[];valid_history=[];steps=0;episode=0;pending_updates=0;next_checkpoint=cfg['checkpoint_every_steps']
    best_rank=None;best_path=directory/'selected.pt';best_steps=None
    start=time.perf_counter()
    while steps<cfg['training_steps']:
        if episode%4==0:
            soc=[.1,.2,.3]
        elif episode%4==1:
            soc=[.3,.5,.7]
        else:
            soc=rng.uniform(.1,.75,3)
        case=_case(f'training_{episode:06d}',soc,vectors)
        frac=min(1.,steps/(cfg['training_steps']*cfg['epsilon_decay_fraction']))
        epsilon=cfg['epsilon_start']+(cfg['epsilon_end']-cfg['epsilon_start'])*frac
        ep,_,_,score=rollout(vectors,case,cfg,dt,learner,shield,epsilon,
                              max_actions=cfg['training_steps']-steps,hold=False)
        count=len(ep['actions'])
        if not count:
            raise RuntimeError('Training collected an empty episode')
        buffer.append(ep);steps+=count;pending_updates+=count;episode+=1;loss=None
        while pending_updates>=cfg['update_every_collected_steps'] and len(buffer)>=5:
            chosen=rng.choice(len(buffer),size=min(len(buffer),cfg['batch_size']),replace=False)
            loss=learner.train([buffer[int(i)] for i in chosen],max_sequence_length=cfg['learning_sequence_steps'])
            pending_updates-=cfg['update_every_collected_steps']
        if episode%cfg['target_sync_every_episodes']==0:
            learner.sync()
        history.append(dict(episode=episode,environment_steps=steps,epsilon=epsilon,reward=sum(ep['rewards']),
            loss=loss,charging_target_reached=score['charging_target_reached'],
            charge_phase_constraints_satisfied=score['constraints_satisfied'],
            failure_reason=score['failure_reason'],initial_soc=case['initial_soc'],
            elapsed_wall_s=time.perf_counter()-start))
        write_csv(directory/'training.csv',history)
        if episode%10==0:
            print(f'dt={dt:g} seed={seed} shield={shield} steps={steps}/{cfg["training_steps"]} ep={episode} target={score["charging_target_reached"]} constraints={score["constraints_satisfied"]}',flush=True)
        if steps>=next_checkpoint or steps==cfg['training_steps']:
            learner.save(directory/f'steps_{steps}.pt',{'seed':seed,'steps':steps,'episode':episode,'shield_in_training':shield,'decision_s':dt})
            # Greedy evaluation consumes no learner RNG; it resets hidden state,
            # which is also reset at the start of the next training episode.
            records,rank=_validation(vectors,validation,cfg,dt,learner,shield)
            valid_history.append(dict(steps=steps,episode=episode,success_rate=rank[0],constraint_rate=rank[1],negative_capped_time=rank[2]))
            write_csv(directory/'validation_history.csv',valid_history)
            write_csv(directory/f'validation_steps_{steps}.csv',records)
            if best_rank is None or rank>best_rank:
                best_rank=rank;best_steps=steps
                learner.save(best_path,{'seed':seed,'steps':steps,'episode':episode,'shield_in_training':shield,
                    'decision_s':dt,'selection':'validation lexicographic safe-success, constraints, negative capped charging time'})
            while next_checkpoint<=steps:
                next_checkpoint+=cfg['checkpoint_every_steps']
    learner.save(directory/'last.pt',{'seed':seed,'steps':steps,'episode':episode,'shield_in_training':shield,'decision_s':dt})
    dump(directory/'training_completed.json',{'steps':steps,'episodes':episode,'best_validation_steps':best_steps,
         'best_validation_rank':best_rank,'gamma_per_step':gamma,'exact_resumption_supported':False,
         'training_wall_s':time.perf_counter()-start})
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([r['environment_steps'] for r in history],[r['reward'] for r in history],alpha=.6)
    axes[0].set(xlabel='Collected pack decisions',ylabel='Episode reward')
    axes[1].plot([r['steps'] for r in valid_history],[r['success_rate'] for r in valid_history],marker='o')
    axes[1].set(xlabel='Collected pack decisions',ylabel='Validation safe task success rate',ylim=(-.02,1.02))
    fig.tight_layout();fig.savefig(directory/'learning.png',dpi=150);plt.close(fig)
    return best_path


def run(root:Path,out:Path,cfg:dict):
    _validate(cfg)
    torch.set_num_threads(cfg['cpu_threads'])
    vectors,handoff=_handoff(root,out,cfg)
    validation,tests=_cases(vectors,cfg)
    dump(out/'validation_cases.json',validation);dump(out/'test_cases.json',tests)
    protocol={'methods':['qmix_raw','qmix_frozen_shield','qmix_trained_shield','rule_shield'],
        'raw_vs_frozen_shield':'identical validation-selected raw-trained checkpoint; isolates deployment safety screen',
        'trained_shield':'separately trained with screen in the environment; same training interaction budget and seeds',
        'rule':'fixed current law, not trained and not repeated to fabricate independent seeds',
        'control_authority':'three independently commanded nonnegative currents 0..7.5 A; 0.5 A grid; no shared supply-power limit in E2',
        'information':{'qmix_actor':'own SOC/V/T, own executed previous action, cell identity and recurrent history',
            'qmix_training_mixer':'all three observed SOC/V/T states during centralized training',
            'rule':'centralized access to the three SOC measurements; explicitly a stronger-information engineering comparator',
            'safety':'each cell measured SOC/V/T and its own nominal model; no true perturbed parameters'},
        'safety_model':'nominal separate model plus measured SOC/V/T discrepancies, not true perturbed plant state',
        'hard_interlock':'all shield groups share ideal simulated continuous-measurement 4.2V/309K/SOC-upper-bound event shutdown; any trip is failed episode, not successful charging or a hardware guarantee',
        'plant_soc_measurement':'ideal model SOC is available; realistic SOC estimator errors are deferred to E5',
        'sampled_safety':{'voltage_V':4.2,'temperature_K':309,'soc_max':cfg['soc_max'],'sample_dt_s':cfg['plant_sample_dt_s']},
        'reward':'original coefficients integrated over actual interval /90s; numerical/screen failure penalty explicitly added',
        'selection':'validation safe success, then constraints, then capped completion time; ties keep earlier checkpoint',
        'final_test':'frozen weights; no tuning or checkpoint selection using test outcomes',
        'precision':'E1 discharge checks plus E2 high-precision integration; dense sampling is not a continuous-time proof',
        'training_target_sync':'every 50 training episodes unless configured; no evaluation-based target updates',
        'batch_size_change':'32 rather than original128 to bound recurrent-memory cost; all E2 training groups identical',
        'config':cfg,'model_assumptions':handoff.get('assumptions',[]),
        'five_seed_protocol':len(cfg['seeds'])>=5,'authored_without_execution':True}
    dump(out/'protocol.json',protocol)
    all_results=[]
    for dt in cfg['decision_periods_s']:
        period=out/f'decision_{dt:g}s';period.mkdir()
        for seed in cfg['seeds']:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            paths={}
            for shield,label in [(False,'raw'),(True,'shield_aware')]:
                print(f'TRAIN {label} dt={dt:g}s seed={seed}',flush=True)
                paths[label]=_train(vectors,validation,cfg,dt,seed,shield,period/f'seed_{seed}'/label)
            for method,label,use_shield in [('qmix_raw','raw',False),('qmix_frozen_shield','raw',True),('qmix_trained_shield','shield_aware',True)]:
                learner=Learner(seed,learning_rate=cfg['learning_rate'],gamma=cfg['gamma_per_90s']**(dt/90.))
                learner.load(paths[label])
                for case in tests:
                    target=period/f'seed_{seed}'/'test'/method/case['case_id'];target.mkdir(parents=True)
                    _,trace,decisions,result=rollout(vectors,case,cfg,dt,learner,use_shield,hold=True)
                    record={'decision_s':dt,'method':method,'seed':seed,'case_id':case['case_id'],'test_family':case['test_family'],**result}
                    all_results.append(record)
                    write_csv(target/'trajectory.csv',trace);write_csv(target/'decisions.csv',decisions)
                    dump(target/'metrics.json',record)
                    if case['case_id'] in ('A','B'):
                        plot_case(trace,target/'trajectory.png',f'{method} | {case["case_id"]} | dt={dt:g}s | seed={seed}')
                    write_csv(out/'episode_metrics.csv',all_results)
            dump(period/f'seed_{seed}'/'completed.json',{'status':'completed','selected_checkpoints':paths})
        for case in tests:
            target=period/'rule_shield'/case['case_id'];target.mkdir(parents=True)
            _,trace,decisions,result=rollout(vectors,case,cfg,dt,learner=None,shield=True,hold=True)
            record={'decision_s':dt,'method':'rule_shield','seed':-1,'case_id':case['case_id'],'test_family':case['test_family'],**result}
            all_results.append(record)
            write_csv(target/'trajectory.csv',trace);write_csv(target/'decisions.csv',decisions);dump(target/'metrics.json',record)
            if case['case_id'] in ('A','B'):
                plot_case(trace,target/'trajectory.png',f'rule + same shield | {case["case_id"]} | dt={dt:g}s')
        write_csv(out/'episode_metrics.csv',all_results)
    groups,assessment=summarize(all_results,out,cfg)
    lines=['# 实验 2 安全与持续均衡结果','',
        '计算已完成。是否支持论文结论请查看 scientific_assessment.json 和 group_summary.csv。',
        '原策略与同权重加安全层构成配对对照；安全层参与训练的版本独立训练；规则组仅运行一次，不伪造多种子重复。','',
        '| 决策周期 s | 方法 | 测试类别 | 安全任务成功率 | 经验约束满足率 |',
        '|---:|---|---|---:|---:|']
    for group in groups:
        lines.append(f'| {group["decision_s"]:g} | {group["method"]} | {group["test_family"]} | {group["success_rate"]:.3f} | {group["empirical_constraint_satisfaction_rate"]:.3f} |')
    lines+=['','安全层为有限时域采样预测筛选，未证明不变集或全局安全；未作实物测试。',
        f'持续均衡只统计充电阶段，{cfg["hold_s"]:g}秒终端保持另列。首次均衡恰好位于终点时，充电阶段持续时间为0。',
        '失败/超时样本全部保留。成功样本充电时间是条件统计，成功率不同时不能据此直接排名。',
        '端口输入 Wh 不是能量损耗；各节 Ah 之和不是串联电池包容量；没有寿命改善结论。',
        '5个以上种子是预定正式规模，较小配置只能用于探索。安全层参数为预定工程假设，不是已由实验1确认的置信误差界。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'evaluated_episodes':len(all_results),'assessment':assessment,'results_file':'group_summary.csv'}
