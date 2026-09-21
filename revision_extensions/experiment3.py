"""E3: validation-selected baselines with separate discrete/continuous tracks."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from common import dump,write_csv
from study_core import load_chain,make_cases,train_policy,run_suite,finalize,resolved_cfg
from study_statistics import compare


LEARNED=('qmix','mappo','dqn','iql_shared','mappo_continuous','sac')


def run(root:Path,out:Path,cfg:dict):
    ctx=load_chain(root,out,cfg,required=[1,2]);effective=resolved_cfg(ctx['base_cfg'])
    vectors=ctx['vectors']
    if vectors.shape!=(3,8):raise ValueError('E3 requires three calibrated cells')
    rates=effective['tuning_learning_rates'];tuning_seeds=effective['tuning_seeds']
    if not rates or len(set(rates))!=len(rates) or any(not np.isfinite(x) or x<=0 for x in rates):
        raise ValueError('Predeclare distinct positive learning rates')
    if not tuning_seeds or len(set(tuning_seeds))!=len(tuning_seeds) or set(tuning_seeds)&set(effective['seeds']):
        raise ValueError('Tuning seeds must be distinct and separate from formal seeds')
    if not isinstance(effective['tuning_steps'],int) or effective['tuning_steps']<1:
        raise ValueError('Positive integer tuning_steps required')
    validation=make_cases(vectors,effective,effective['validation_seed'],effective['validation_count'],'E3_validation')
    tests=make_cases(vectors,effective,effective['test_seed'],effective['test_count'],'E3_test')
    for label,initial in [('A',[.1,.2,.3]),('B',[.3,.5,.7])]:
        tests.append({'case_id':'familiar_'+label,'initial_soc':initial,'plant_vectors':vectors.tolist(),
                      'test_family':'familiar_A_B','ambient_K':effective['nominal_ambient_K'],
                      'observation_seed':effective['test_seed']+700001})
    dump(out/'validation_cases.json',validation);dump(out/'test_cases.json',tests)
    protocol={
        'experiment':3,'discrete_track':['qmix','mappo','dqn','iql_shared','cccv','soc_rule'],
        'continuous_track':['mappo_continuous','sac','cccv_continuous'],
        'action_authority':{'discrete':'0..7.5 A in 0.5 A steps per cell',
                            'continuous':'0..7.5 A continuous per cell, no rounding'},
        'all_policies_train_with_same_track_safety':True,'formal_seeds':effective['seeds'],
        'same_pack_decision_budget':effective['training_steps'],
        'optimizer_allocation':{'value_based':'one sequence-batch update per 50 collected steps',
                                'SAC':f"{effective['sac_gradient_steps']} updates per {effective['update_every_collected_steps']} collected steps, minibatch {effective['sac_batch_size']}",
                                'MAPPO':f"{effective['ppo_epochs']} epochs per >= {effective['rollout_batch_steps']} on-policy steps"},
        'heldout_design':'Validation and main test use separate random initial states. Familiar training A/B states are extra diagnostics with separate summaries.',
        'tuning':{'learning_rates':rates,'seeds':tuning_seeds,'steps_per_candidate_seed':effective['tuning_steps'],
                  'selection':'mean validation safe success, constraint rate, then negative failure-capped time; fixed first tie'},
        'checkpoint_selection':'validation only; every predeclared step threshold',
        'fairness_scope':'Equal environment interactions and learning-rate search count. Update rules, architectures, optimizer steps and information scope remain method-specific and are disclosed.',
        'information':{'qmix':'shared recurrent local Q, executed-action history, agent identity; centralized mixer in training',
                       'mappo':'shared local feedforward actor and centralized critic',
                       'iql_shared':'shared recurrent local Q without mixer',
                       'dqn':'centralized joint observation and 4096 discrete joint actions',
                       'sac':'centralized joint continuous actor and twin critics',
                       'soc_rule':'global SOC spread available to deterministic rule'},
        'cccv':'Sampled CC-CV feedback under same predictive screen with observed-SOC taper toward 0.93 (75 A/SOC slope, minimum positive request 0.5 A). Select CC current from predeclared candidates using validation; not an ideal analog charger. Task still ends at all SOC>=0.90 and std<=0.02.',
        'comparisons':'Primary QMIX comparisons stay in discrete track. SAC comparisons use continuous MAPPO and continuous CC-CV. Cross-track differences are not attributed solely to algorithm.',
        'safety_scope':'Numerical sampled predictive screening plus simulated plant interlock; E2 retains raw-policy safety evidence.',
        'reviewer_links':['R1 MAPPO','R2 modern baselines/PPO/SAC','R3 DQN/IQL/discrete-action rationale'],
        'test_selection_forbidden':True,'authored_without_execution':True}
    dump(out/'protocol.json',protocol)
    tuning=deepcopy(effective);tuning['training_steps']=effective['tuning_steps']
    tuning['checkpoint_every_steps']=effective['tuning_steps']
    tuning_rows=[];chosen={};specs=[]
    for method in LEARNED:
        candidates=[]
        for j,rate in enumerate(rates):
            spec={'id':method,'method':method,'learning_rate':float(rate)}
            ranks=[]
            for seed in tuning_seeds:
                entry=train_policy(spec,vectors,tuning,out/'tuning'/method/f'lr_{j}'/f'seed_{seed}',validation,seed)
                ranks.append(entry['validation_rank'])
                tuning_rows.append({'method_id':method,'candidate':j,'learning_rate':rate,'seed':seed,
                                    'validation_rank':entry['validation_rank'],'checkpoint':entry['checkpoint']})
                write_csv(out/'tuning_results.csv',tuning_rows)
            candidates.append((tuple(np.mean(np.asarray(ranks),axis=0)),spec))
        # max preserves first declared candidate on exact ties.
        rank,selected=max(candidates,key=lambda pair:pair[0]);chosen[method]={'spec':selected,'tuning_mean_validation_rank':rank}
        specs.append(selected);dump(out/'selected_hyperparameters.json',chosen)
    for method in ('cccv','cccv_continuous'):
        candidates=[]
        for j,current in enumerate(effective['cc_current_candidates_A']):
            if not np.isfinite(current) or not 0<current<=7.5:raise ValueError('Invalid CC candidate')
            spec={'id':method,'method':method,'cc_current_A':float(current)}
            entry=train_policy(spec,vectors,effective,out/'tuning'/method/f'current_{j}',validation,-1)
            candidates.append((tuple(entry['validation_rank']),spec))
            tuning_rows.append({'method_id':method,'candidate':j,'cc_current_A':current,'seed':-1,
                                'validation_rank':entry['validation_rank']})
        if not candidates:raise ValueError('CC candidate list must not be empty')
        rank,selected=max(candidates,key=lambda pair:pair[0]);chosen[method]={'spec':selected,'validation_rank':rank};specs.append(selected)
    specs.append({'id':'soc_rule','method':'soc_rule'})
    chosen['soc_rule']={'spec':specs[-1],'selection':'fixed predeclared gains, no learned weights'}
    dump(out/'selected_hyperparameters.json',chosen);write_csv(out/'tuning_results.csv',tuning_rows)
    suite=run_suite(specs,vectors,effective,out,validation,tests)
    compare(suite['records'],out,effective,references={'discrete':'qmix','continuous':'mappo_continuous'})
    result=finalize(out,suite['records'],{'experiment':3,'formal_seed_count_met':len(effective['seeds'])>=5,
        'tuning_is_validation_only':True,'discrete_continuous_separated':True,
        'primary_comparison_file':'paired_comparison_summary.csv'})
    return {'experiment':3,'models':len(suite['registry']),'episodes':len(suite['records']),'assessment':result['assessment']}
