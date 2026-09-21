"""E5: predeclared reward/exploration sensitivity and frozen-policy transfer.

Only run() starts work. This module was authored without executing experiments.
Test outcomes never change parameters, training budgets, or selected checkpoints.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

from common import dump, sha256, write_csv
from study_core import (load_chain, make_cases, train_policy, evaluate_policy,
                        finalize)
from study_statistics import compare


DEFAULT_WEIGHTS = {'time': .75, 'balance': 50., 'voltage': 20., 'temperature': 2.}


def _method(entry):
    return entry.get('method_id', entry.get('spec', {}).get('id'))


def _sensitivity_specs(learning_rate):
    common = {'method': 'qmix', 'shield': True,
              'reward_mode': 'raw', 'reward_weights': dict(DEFAULT_WEIGHTS),
              'epsilon_start': .5, 'learning_rate': float(learning_rate)}
    specs = [dict(deepcopy(common), id='qmix_reward_reference')]
    specs.append(dict(deepcopy(common), id='qmix_reward_all_unit_raw',
                      reward_weights={key: 1. for key in DEFAULT_WEIGHTS}))
    specs.append(dict(deepcopy(common), id='qmix_reward_all_unit_normalized',
                      reward_mode='normalized_unit',
                      reward_weights={key: 1. for key in DEFAULT_WEIGHTS},
                      reward_normalizers={'time': 1., 'balance': .1,
                                          'voltage': .1, 'temperature': 10.}))
    for component in ('balance', 'voltage', 'temperature'):
        for factor, suffix in ((.5, 'half'), (2., 'double')):
            weights = dict(DEFAULT_WEIGHTS)
            weights[component] *= factor
            specs.append(dict(deepcopy(common),
                              id=f'qmix_reward_{component}_{suffix}',
                              reward_weights=weights))
    specs.append(dict(deepcopy(common), id='qmix_epsilon_one', epsilon_start=1.))
    return specs


def _generalization_cases(vectors, cfg):
    seed = int(cfg['generalization_seed'])
    rng = np.random.default_rng(seed)
    base = make_cases(vectors, cfg, seed, int(cfg['generalization_random_count']),
                      'E5_random')
    # These additional cases are initially almost balanced. They measure loss of
    # balance during charging, rather than only time to first reach balance.
    balanced = make_cases(vectors, cfg, seed + 1,
                          int(cfg['generalization_balanced_count']), 'E5_balanced')
    for case in balanced:
        center = float(rng.uniform(.15, .75))
        offsets = rng.uniform(-.012, .012, len(vectors))
        offsets -= offsets.mean()
        case['initial_soc'] = np.clip(center + offsets, .05, .85).tolist()
        case['initial_condition_family'] = 'initially_balanced'
    for case in base:
        case['initial_condition_family'] = 'random_soc'
    base += balanced
    for index, case in enumerate(base):
        case['base_case_id'] = case['case_id']
        case['plant_vectors'] = np.asarray(vectors, dtype=float).tolist()
        case['ambient_K'] = float(cfg.get('nominal_ambient_K', 298.15))
        case['observation_seed'] = seed + 10000 + index

    scenarios = [
        ('nominal', {}),
        ('ambient_15C', {'ambient_K': 288.15}),
        ('ambient_34C', {'ambient_K': 307.15}),
        ('ambient_35C', {'ambient_K': 308.15, 'feasibility_negative_control': True}),
        ('parameters_minus_5pct', {'parameter_factor': .95}),
        ('parameters_plus_5pct', {'parameter_factor': 1.05}),
        ('parameters_minus_10pct', {'parameter_factor': .90}),
        ('parameters_plus_10pct', {'parameter_factor': 1.10}),
        ('sensor_noise_low', {'observation_noise': [.005, .005, .25]}),
        ('sensor_noise_high', {'observation_noise': [.02, .02, 1.]}),
        ('sensor_bias', {'observation_bias': [.02, -.02, -1.]}),
        ('observation_delay_one_decision', {'observation_delay_steps': 1}),
        ('current_cap_drop_30pct', {'power_drop_at_s': 900.,
                                    'current_cap_factor_after': .7}),
    ]
    result = []
    for family, changes in scenarios:
        for original in base:
            case = deepcopy(original)
            case['case_id'] = f'{family}__{original["case_id"]}'
            case['test_family'] = family
            for key, value in changes.items():
                if key != 'parameter_factor':
                    case[key] = deepcopy(value)
            if 'parameter_factor' in changes:
                shifted = np.asarray(vectors, dtype=float).copy()
                # Dp,Dn,h,cp multiplier; no electrode fraction/geometry changes.
                shifted[:, [0, 1, 6, 7]] *= changes['parameter_factor']
                case['plant_vectors'] = shifted.tolist()
                case['parameter_shift_indices'] = [0, 1, 6, 7]
            case['initial_temperature_margin_feasible'] = bool(
                case['ambient_K'] <= 309. - cfg['temperature_margin_K'])
            result.append(case)
    return result


def _frozen_entries(registry, seeds):
    selected = []
    for method in ('qmix', 'mappo'):
        for seed in seeds:
            matches = [e for e in registry if _method(e) == method
                       and int(e['seed']) == int(seed) and int(e['n_agents']) == 3]
            if len(matches) != 1:
                raise ValueError(f'E5 needs exactly one E3 {method} checkpoint for '
                                 f'seed {seed}; found {len(matches)}. Re-run E3 with '
                                 'matching seeds; do not choose using E3 test results.')
            selected.append(deepcopy(matches[0]))
    return selected


def _paired_degradation(records):
    nominal = {(r['method_id'], r['seed'], r.get('base_case_id')): r
               for r in records if r['test_family'] == 'nominal'}
    paired = []
    for row in records:
        if row['test_family'] == 'nominal':
            continue
        ref = nominal.get((row['method_id'], row['seed'], row.get('base_case_id')))
        if ref is None:
            raise ValueError('Every perturbed case must have its paired nominal case')
        paired.append({
            'method_id': row['method_id'], 'seed': row['seed'],
            'base_case_id': row['base_case_id'], 'test_family': row['test_family'],
            'initial_condition_family': row['initial_condition_family'],
            'nominal_success': ref['task_success'],
            'perturbed_success': row['task_success'],
            'success_difference': int(row['task_success']) - int(ref['task_success']),
            'nominal_constraints_satisfied': ref['constraints_satisfied'],
            'perturbed_constraints_satisfied': row['constraints_satisfied'],
            'nominal_charge_time_s': ref.get('charging_time_s'),
            'perturbed_charge_time_s': row.get('charging_time_s'),
            'time_difference_if_both_success_s':
                (row['charging_time_s'] - ref['charging_time_s'])
                if row['task_success'] and ref['task_success'] else None,
        })
    return paired


def _degradation_summary(paired, cfg):
    """Paired effects conditional on fixed cases; bootstrap independent seeds."""
    rng = np.random.default_rng(int(cfg['bootstrap_seed']))
    groups = sorted({(r['method_id'], r['test_family'], r['initial_condition_family'])
                     for r in paired})
    summary = []
    for method, family, initial in groups:
        rows = [r for r in paired if (r['method_id'], r['test_family'],
                                     r['initial_condition_family']) == (method, family, initial)]
        seeds = sorted({r['seed'] for r in rows})
        means = np.array([np.mean([r['success_difference'] for r in rows
                                  if r['seed'] == seed]) for seed in seeds])
        ci = [None, None]
        if len(seeds) >= 2:
            samples = rng.choice(means, (int(cfg['bootstrap_replicates']), len(seeds)),
                                 replace=True).mean(axis=1)
            ci = np.quantile(samples, [.025, .975]).tolist()
        summary.append({'method_id': method, 'test_family': family,
                        'initial_condition_family': initial,
                        'independent_training_seeds': len(seeds),
                        'paired_cases': len(rows),
                        'mean_paired_success_difference': float(means.mean()),
                        'seed_bootstrap_ci_low': ci[0], 'seed_bootstrap_ci_high': ci[1],
                        'sign': 'negative means degradation from paired nominal condition',
                        'inference_scope': 'Descriptive unadjusted interval over training '
                                           'seeds on a fixed synthetic test set; '
                                           'not simultaneous inference across all conditions.'})
    return summary


def run(root: Path, out: Path, cfg: dict):
    ctx = load_chain(root, out, cfg, required=[1, 2, 3, 4])
    effective = deepcopy(ctx['base_cfg'])
    effective.update(cfg)
    vectors = np.asarray(ctx['vectors'], dtype=float)
    if vectors.shape != (3, 8):
        raise ValueError('E5 requires the three calibrated E1 cell parameter vectors')
    seeds = [int(seed) for seed in effective['seeds']]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Provide at least one unique training seed')
    upstream_frozen = _frozen_entries(ctx['registry'], seeds)
    for entry in upstream_frozen:
        if sha256(Path(entry['checkpoint'])) != entry['checkpoint_sha256']:
            raise ValueError('An E3 checkpoint differs from its registered hash; '
                             'restore the complete upstream run before E5 training.')
        for key in ('decision_s', 'voltage_margin_V', 'temperature_margin_K',
                    'soc_max', 'horizon_s', 'hold_s'):
            if entry['config'][key] != effective[key]:
                raise ValueError(f'E5 and frozen E3 policy disagree on {key}; '
                                 'use the same task and safety protocol.')
    qmix_learning_rates = {float(entry['config']['learning_rate'])
                           for entry in upstream_frozen if _method(entry) == 'qmix'}
    if len(qmix_learning_rates) != 1:
        raise ValueError('E3 QMIX seeds must use one validation-selected learning '
                         'rate before reward/exploration sensitivity comparisons.')
    qmix_learning_rate = next(iter(qmix_learning_rates))
    reserved = [int(effective[k]) for k in ('validation_seed', 'test_seed',
                                           'generalization_seed')]
    if len(set(reserved)) != len(reserved):
        raise ValueError('Validation, sensitivity test, and transfer seeds must differ')
    for key in ('validation_count', 'test_count', 'generalization_random_count',
                'generalization_balanced_count'):
        if int(effective[key]) <= 0:
            raise ValueError(f'{key} must be positive')

    validation = make_cases(vectors, effective, effective['validation_seed'],
                            effective['validation_count'], 'E5_validation')
    sensitivity_tests = make_cases(vectors, effective, effective['test_seed'],
                                   effective['test_count'], 'E5_sensitivity_test')
    for case in sensitivity_tests:
        case['test_family'] = 'reward_exploration_sensitivity'
    transfer_tests = _generalization_cases(vectors, effective)
    # Freeze all test designs before the first E5 learning update.
    dump(out / 'validation_cases.json', validation)
    dump(out / 'sensitivity_test_cases.json', sensitivity_tests)
    dump(out / 'generalization_test_cases.json', transfer_tests)
    specs = _sensitivity_specs(qmix_learning_rate)
    domain_spec = dict(deepcopy(specs[0]), id='qmix_domain_random',
                       domain_randomization=True)
    protocol = {
        'experiment': 5, 'specs': specs, 'domain_random_spec': domain_spec,
        'validation_only_selection': True, 'training_seeds': seeds,
        'nominal_control_retrained_for_sensitivity': True,
        'learning_rate': qmix_learning_rate,
        'learning_rate_source': 'E3 validation-selected QMIX setting, held fixed '
                                'for every E5 reward/exploration/domain-randomization group.',
        'one_factor_at_a_time': 'Raw unit reward and normalized unit reward are '
                              'explicit additional multi-coefficient controls.',
        'unit_reward': 'All four physical penalty gains equal one; numerical '
                       'failure penalty is unchanged and reported separately.',
        'normalization': 'Predeclared scales: time 1, SOC spread excess 0.1, '
                         'voltage excess 0.1 V, temperature excess 10 K. '
                         'No scales estimated from final test outcomes.',
        'frozen_transfer': 'All E3 qmix/mappo seeds plus E5 nominal and domain '
                           'randomized QMIX; no test-time learning or best-seed selection.',
        'domain_randomization_training': {'ambient_K': [288.15, 303.15],
            'parameter_indices': [0, 1, 6, 7], 'relative_shift': [-.05, .05],
            'observation_noise_std': [.01, .005, .5]},
        'paired_cases': 'Each perturbation has the same initial SOC and sensor '
                        'random stream as its nominal reference for every method.',
        'temperature_scope': 'Stationary ambient and initial temperatures; '
                             'not a dynamic ambient-step experiment.',
        'temperature_feasibility_control': 'At default 1 K screening margin, '
            '35 C starts above the 308 K screen threshold even at zero current. '
            'This is an explicit negative control of screen feasibility, not '
            'evidence of learned-policy weakness or physical task infeasibility. '
            '34 C provides the additional warm condition inside the initial margin.',
        'dynamic_test': 'At 900 s every cell current cap becomes 70% of 7.5 A; '
                        'the 5.25 A ceiling is floored to 5.0 A on the shared '
                        '0.5 A discrete action grid. '
                        'this is a current-authority disturbance, not an electrical '
                        'constant-power or shared-supply model.',
        'parameter_scope': 'Synthetic perturbations of diffusivities, convection, '
                           'and heat capacities; not independent measured-cell validation.',
        'hold_and_failures': 'Retain all timeouts, protective stops, and numerical '
                             'failures; success requires terminal hold and constraints.',
        'shield_reward_interaction': 'The common safety screen may prevent voltage '
                                    'and temperature penalties from activating. '
                                    'Insensitivity in this protected system does '
                                    'not establish robustness of the original '
                                    'unprotected reward design.',
        'authored_without_execution': True,
    }
    dump(out / 'protocol.json', protocol)

    sensitivity_records, trained = [], []
    for spec in specs + [domain_spec]:
        for seed in seeds:
            directory = out / 'training' / spec['id'] / f'seed_{seed}'
            entry = train_policy(spec, vectors, effective, directory, validation, seed)
            trained.append(entry)
            dump(out / 'model_registry.json', trained)
            records = evaluate_policy(entry, vectors, effective, sensitivity_tests,
                                      out / 'sensitivity' / spec['id'] / f'seed_{seed}')
            for row in records:
                row['study_panel'] = 'reward_and_exploration'
            sensitivity_records.extend(records)
            write_csv(out / 'sensitivity_episode_metrics.csv', sensitivity_records)

    compare(sensitivity_records, out / 'sensitivity_statistics', effective,
            references={'discrete': 'qmix_reward_reference'})

    frozen = upstream_frozen
    frozen += [entry for entry in trained
               if _method(entry) in ('qmix_reward_reference', 'qmix_domain_random')]
    dump(out / 'transfer_model_registry.json', frozen)
    transfer_records, checkpoint_audit = [], []
    case_lookup = {case['case_id']: case for case in transfer_tests}
    for entry in frozen:
        checkpoint = Path(entry['checkpoint'])
        digest = sha256(checkpoint)
        if digest != entry['checkpoint_sha256']:
            raise ValueError(f'Checkpoint changed before frozen evaluation: {checkpoint}')
        records = evaluate_policy(entry, vectors, effective, transfer_tests,
                                  out / 'generalization' / _method(entry) /
                                  f'seed_{entry["seed"]}')
        after = sha256(checkpoint)
        if after != digest:
            raise RuntimeError('Frozen-policy evaluation modified its checkpoint')
        checkpoint_audit.append({'method_id': _method(entry), 'seed': entry['seed'],
                                 'checkpoint': str(checkpoint), 'sha256_before': digest,
                                 'sha256_after': after, 'unchanged': True})
        for row in records:
            case = case_lookup[row['case_id']]
            row['base_case_id'] = case['base_case_id']
            row['initial_condition_family'] = case['initial_condition_family']
            row['initial_temperature_margin_feasible'] = case['initial_temperature_margin_feasible']
            row['feasibility_negative_control'] = case.get('feasibility_negative_control', False)
            row['study_panel'] = 'frozen_policy_generalization'
        transfer_records.extend(records)
        write_csv(out / 'generalization_episode_metrics.csv', transfer_records)
        dump(out / 'frozen_checkpoint_audit.json', checkpoint_audit)
    paired = _paired_degradation(transfer_records)
    write_csv(out / 'paired_generalization_degradation.csv', paired)
    write_csv(out / 'paired_generalization_summary.csv',
              _degradation_summary(paired, effective))
    summary = finalize(out, sensitivity_records + transfer_records, {
        'experiment': 5, 'nominal_training_seed_count': len(seeds),
        'formal_seed_count_met': len(seeds) >= 5,
        'claims_not_established': ['real-cell transfer', 'hardware safety',
                                  'global convergence', 'lifetime improvement'],
        'reviewer_links': ['R1.1 dynamic conditions', 'R2.5 reward sensitivity',
                          'R2.6 limitations', 'R3 unit gains', 'R3 epsilon=1'],
        'frozen_checkpoint_audit': checkpoint_audit,
        'paired_sensitivity_statistics_directory': 'sensitivity_statistics',
    })
    (out / 'E5_INTERPRETATION.md').write_text(
        '# 实验 5：敏感性与冻结策略泛化\n\n'
        '先看安全任务成功率及失败原因，再比较成功样本的充电时间；'
        '不能只保留成功样本宣称算法更快。所有奖励组均重新训练，'
        '使用相同种子、交互预算、验证集与最终测试集。\n\n'
        'sensitivity_statistics 按预先指定的标称 QMIX 对照，保存配对案例与种子差异、'
        '置信区间、种子级符号翻转检验及 Holm 多重比较校正。'
        '5 个种子的双侧精确检验最小 p 值为 0.0625，不能把大量场景伪装成独立训练重复。\n\n'
        '单位原始权重和单位归一化权重为两组不同实验，不能混用。'
        '归一化尺度在训练前固定，数值失败惩罚始终单列。\n\n'
        '所有正式敏感性组使用相同安全机制；该机制可能使电压/温度惩罚很少触发。'
        '若这些增益变化效果不明显，只能说明本安全机制下的结果，'
        '不能据此证明原始无保护奖励设计对增益不敏感。\n\n'
        '泛化组冻结 E3 的 QMIX/MAPPO，以及本实验同协议训练的标称/'
        '域随机化 QMIX。不得按最终测试结果重新选种子、参数或模型。'
        'paired_generalization_degradation.csv 按相同初态配对；'
        'paired_generalization_summary.csv 分开汇总随机初态与初始均衡工况，'
        '对种子层面的配对成功率差异给出描述性置信区间，不能当作多条件同时显著性证明。'
        '温度是恒定环境条件，动态变化仅为第 900 秒的电流上限下降；'
        '设定上限为 5.25 A，按离散动作步长向下取整后实际最大为 5.0 A。\n\n'
        '默认温度裕量为 1 K，35°C 初态已超过安全筛选的 308 K 阈值，'
        '所以 35°C 单列为筛选可行性负对照，不能将其失败解释为算法学习能力差，'
        '也不能由此认定真实物理任务不可行。另设 34°C 的初始裕量内高温测试。\n\n'
        '传感噪声、参数扰动和电芯扩展均为仿真设计，不能证明真实跨电芯、'
        '跨温度泛化。硬件安全、收敛证明与寿命改善仍未解决。\n', encoding='utf-8')
    return {'experiment': 5, 'trained_checkpoints': len(trained),
            'frozen_checkpoints': len(frozen),
            'sensitivity_episodes': len(sensitivity_records),
            'generalization_episodes': len(transfer_records), 'assessment': summary}
