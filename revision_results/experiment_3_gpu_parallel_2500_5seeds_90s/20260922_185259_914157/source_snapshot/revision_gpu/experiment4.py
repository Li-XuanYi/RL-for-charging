"""E4: controlled component ablations, all trained again from initialization."""
from copy import deepcopy
from pathlib import Path
import json
from common import dump
from study_core import load_chain,run_suite,finalize,resolved_cfg
from study_statistics import compare


def run(root:Path,out:Path,cfg:dict):
    ctx=load_chain(root,out,cfg,required=[1,2,3]);effective=resolved_cfg(ctx['base_cfg']);e3=ctx['upstreams'][3]
    reference=[e for e in ctx['registry'] if e['method_id']=='qmix']
    if set(e['seed'] for e in reference)!=set(effective['seeds']):
        raise ValueError('E4 formal seeds must match E3 QMIX seeds')
    chosen=json.loads((e3/'selected_hyperparameters.json').read_text(encoding='utf-8-sig'))['qmix']['spec']
    inherited=reference[0]['config']
    # Start from the full E3 reference: optimizer/replay/checkpoint details are
    # control variables too, not just physical limits and nominal train budget.
    bookkeeping={'protocol_version','scope','bootstrap_seed','bootstrap_replicates','cpu_threads','upstream_results',
                 'result_tag','upstream_result_tags','upstream_e2_slice_manifest','exploratory'}
    for key,value in cfg.items():
        if key not in bookkeeping and key in inherited and value!=inherited[key]:
            raise ValueError(f'E4 must keep E3 {key}; only declared components vary')
    effective=deepcopy(inherited)
    effective.update({key:value for key,value in cfg.items() if key in bookkeeping})
    effective['learning_rate']=chosen['learning_rate']
    validation=json.loads((e3/'validation_cases.json').read_text(encoding='utf-8-sig'))
    tests=json.loads((e3/'test_cases.json').read_text(encoding='utf-8-sig'))
    specs=[{'id':'qmix_reference','method':'qmix'},
           {'id':'qmix_no_gru','method':'qmix_mlp'},
           {'id':'qmix_vdn','method':'vdn'},
           {'id':'qmix_static_mixer','method':'static_mixer'},
           {'id':'qmix_no_mixer_shared','method':'iql_shared'},
           {'id':'qmix_unshared','method':'qmix','shared':False},
           {'id':'iql_independent','method':'iql_independent'},
           {'id':'qmix_no_balance_reward','method':'qmix','reward_weights':{**effective['reward_weights'],'balance':0.}}]
    protocol={'experiment':4,'specs':specs,'reference':'qmix_reference','retrain_every_variant':True,
        'same_initialization_seed_list':effective['seeds'],'same_training_budget':effective['training_steps'],
        'validation_and_test_sets_reused_from':str(e3),'learning_rate_inherited_from_E3':chosen['learning_rate'],
        'interpretation':{
            'qmix_no_gru':'Remove recurrent memory, preserve hidden width; parameter count differs and is reported.',
            'qmix_vdn':'Replace nonlinear state-conditioned mixer with additive VDN.',
            'qmix_static_mixer':'Retain monotone nonlinear mixer; remove state-conditioned hypernetworks.',
            'qmix_no_mixer_shared':'Remove joint mixer; independent TD losses with shared local network.',
            'qmix_unshared':'Keep QMIX mixer, remove local-network parameter sharing.',
            'iql_independent':'Remove mixer and sharing together; multi-factor reference, not isolated one-component attribution.',
            'qmix_no_balance_reward':'Set only balance gain to zero; keep the same terminal balanced target and physical metrics.'},
        'held_fixed':'Predictive screen, plant interlock, data model, training steps, initial-state distribution, hold requirement and test scenarios.',
        'no_shield_ablation':'Handled in E2 raw/same-weights-filtered/retrained comparisons; not duplicated here.',
        'capacity_caveat':'This is a functional ablation suite, not parameter-count-matched expressivity proof.',
        'test_protocol':'The same predeclared E3 tests are reused for paired comparison; do not adapt variants after viewing outcomes.',
        'reviewer_links':['R2 GRU/hypernetwork/mixer roles','R3 QMIX rationale/IQL','balance mechanism'],
        'authored_without_execution':True}
    dump(out/'protocol.json',protocol);dump(out/'effective_config.json',effective)
    suite=run_suite(specs,ctx['vectors'],effective,out,validation,tests)
    compare(suite['records'],out,effective,references={'discrete':'qmix_reference'})
    counts=[{'method_id':e['method_id'],'seed':e['seed'],'parameter_count':e['parameter_count'],
             'checkpoint':e['checkpoint'],'validation_rank':e['validation_rank']} for e in suite['registry']]
    dump(out/'architecture_audit.json',counts)
    result=finalize(out,suite['records'],{'experiment':4,'formal_seed_count_met':len(effective['seeds'])>=5,
        'all_variants_retrained':True,'no_test_driven_component_selection':True,
        'primary_comparison_file':'paired_comparison_summary.csv'})
    return {'experiment':4,'models':len(suite['registry']),'episodes':len(suite['records']),'assessment':result['assessment']}
