"""E6: separately trained pack sizes, CPU profiling, and evidence-chain closure.

No experiments execute at import time. Untrained-network timing is kept separate
from closed-loop control evidence. Hardware or theoretical claims are not inferred.
"""
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
import json
import os
import platform
import time

import numpy as np
import torch

from common import dump, sha256, write_csv
from study_core import (load_chain, make_cases, train_policy, evaluate_policy,
                        load_policy, make_policy, finalize)
from study_statistics import compare


def _parameter_inventory(policy):
    """Count unique resident tensor parameters, without guessing deployed size."""
    seen_objects, seen_parameters = set(), set()
    total = 0
    trainable = 0
    devices = set()

    def visit(value, depth=0):
        nonlocal total, trainable
        if id(value) in seen_objects or depth > 6:
            return
        seen_objects.add(id(value))
        if isinstance(value, torch.nn.Module):
            for parameter in value.parameters():
                if id(parameter) in seen_parameters:
                    continue
                seen_parameters.add(id(parameter))
                total += parameter.numel()
                trainable += parameter.numel() if parameter.requires_grad else 0
                devices.add(str(parameter.device))
        elif isinstance(value, dict):
            for child in value.values():
                visit(child, depth + 1)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child, depth + 1)
        elif hasattr(value, '__dict__') and not isinstance(
                value, (torch.optim.Optimizer, np.ndarray)):
            for child in vars(value).values():
                visit(child, depth + 1)
    visit(policy)
    deployed = getattr(policy, 'actor_parameter_count', None)
    if callable(deployed):
        deployed = deployed()
    deployed_basis = 'policy.actor_parameter_count' if deployed is not None else None
    if deployed is None:
        # E6 only profiles QMIX (agent) and MAPPO (actor). Their centralized
        # mixer/critic is not needed in decentralized execution.
        for attribute in ('actor', 'agent'):
            module = getattr(policy, attribute, None)
            if isinstance(module, torch.nn.Module):
                deployed = sum(parameter.numel() for parameter in module.parameters())
                deployed_basis = f'policy.{attribute}.parameters excluding mixer/critic/targets'
                break
    return {'resident_parameter_count': int(total),
            'resident_trainable_parameter_count': int(trainable),
            'deployed_actor_parameter_count': int(deployed) if deployed is not None else None,
            'deployed_parameter_count_basis': deployed_basis,
            'parameter_devices': sorted(devices),
            'count_note': 'Resident count includes critic/mixer/target networks if '
                          'present; not a deployable firmware memory estimate.'}


def _benchmark(policy, n_agents, cfg, label, trained, checkpoint=None):
    """Measure actual user-machine .act() latency only when E6 is invoked."""
    repetitions = int(cfg['inference_repetitions'])
    warmups = int(cfg['inference_warmup'])
    if repetitions < 10 or warmups < 1:
        raise ValueError('Timing requires >=10 repetitions and >=1 warmup')
    inventory = _parameter_inventory(policy)
    if any(device != 'cpu' for device in inventory['parameter_devices']):
        raise ValueError('This timing protocol is CPU-only; do not mix GPU timings')
    index = np.arange(n_agents, dtype=float)
    physical = np.column_stack((.15 + .6 * (index + 1) / (n_agents + 1),
                                3.5 + .4 * (index + 1) / (n_agents + 1),
                                298.15 + np.sin(index)))
    # Match study_core.rollout exactly; policy.act expects normalized SOC/V/T.
    obs = ((physical - [.5, 3.5, 308.]) / [.5, 1., 11.]).astype(np.float32)
    mask = np.ones((n_agents, 16), dtype=bool)
    policy.reset()
    samples = []
    with torch.inference_mode():
        for iteration in range(warmups + repetitions):
            # Reset between 64-step synthetic histories; not included in .act time.
            if iteration % 64 == 0:
                policy.reset()
            started = time.perf_counter_ns()
            currents, _ = policy.act(obs, mask, training=False, epsilon=0.)
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            policy.set_executed(np.asarray(currents, dtype=float))
            if iteration >= warmups:
                samples.append(elapsed_ms)
    values = np.asarray(samples)
    return {
        'method_id': label, 'n_agents': int(n_agents), 'trained': bool(trained),
        'checkpoint': str(checkpoint) if checkpoint else None,
        'warmup_calls': warmups, 'measured_calls': repetitions,
        'latency_mean_ms': float(values.mean()),
        'latency_p50_ms': float(np.quantile(values, .50)),
        'latency_p95_ms': float(np.quantile(values, .95)),
        'latency_p99_ms': float(np.quantile(values, .99)),
        'latency_max_ms': float(values.max()),
        'p50_per_cell_ms': float(np.quantile(values, .50) / n_agents),
        'p95_fraction_of_decision_period':
            float(np.quantile(values, .95) / (1000. * cfg['decision_s'])),
        'scope': 'CPU policy.act: policy input construction plus actor inference '
                 'and action selection. Excludes plant solve, safety screening, '
                 'communication, physical-observation normalization and explicit '
                 'caller reset/set_executed. Policy-internal bookkeeping is included.',
        'synthetic_observations': True, 'closed_loop_performance_evidence': False,
        'timing_samples_ms': samples, **inventory,
    }


def _tile_vectors(vectors, n_agents):
    if n_agents < 3 or n_agents % 3:
        raise ValueError('E6 pack sizes must be positive multiples of three')
    return np.tile(np.asarray(vectors, dtype=float), (n_agents // 3, 1))


def _scaled_cases(vectors, n_agents, cfg, seed, count, prefix):
    base = make_cases(vectors, cfg, seed, count, prefix)
    result = []
    for original in base:
        case = deepcopy(original)
        case['source_case_id'] = original['case_id']
        case['case_id'] = f'N{n_agents}__{original["case_id"]}'
        case['initial_soc'] = np.tile(original['initial_soc'], n_agents // 3).tolist()
        plant = original.get('plant_vectors', vectors)
        case['plant_vectors'] = _tile_vectors(plant, n_agents).tolist()
        case['test_family'] = 'replicated_triplet_scaling'
        case['scaling_construction'] = 'Replicate original calibrated triplet and '
                                       'its initial SOC; preserves per-cell SOC '
                                       'distribution while increasing agent count.'
        result.append(case)
    return result


def _read_assessment(path):
    candidates = ('scientific_assessment.json', 'summary.json', 'handoff.json')
    for name in candidates:
        source = Path(path) / name
        if source.is_file():
            return {'source': str(source), 'sha256': sha256(source),
                    'content': json.loads(source.read_text(encoding='utf-8-sig'))}
    return {'source': None, 'content': None,
            'limitation': 'No structured scientific assessment found; see run report.'}


def _closure(out, ctx, cfg, timing, training_cost, record_count):
    upstream = {f'experiment_{number}': _read_assessment(path)
                for number, path in ctx['upstreams'].items()}
    evidence = [
        {'reviewer': 'R1.1', 'concern': 'dynamic environment',
         'evidence': 'E5 frozen policies under parameter, sensor and current-cap disturbances',
         'status': 'numerical_evidence_generated_review_results',
         'remaining': 'No online hardware adaptation or dynamic ambient-step validation'},
        {'reviewer': 'R1.2; R2.2; R3 DQL', 'concern': 'strong baselines and action spaces',
         'evidence': 'E3 discrete/continuous comparison panels; identical task and shield',
         'status': 'numerical_evidence_generated_review_results',
         'remaining': 'No presumption that proposed method outperforms all baselines'},
        {'reviewer': 'R1.3; R3 online/offline', 'concern': 'data and training provenance',
         'evidence': 'E1 fit/test disclosure; E2-E6 simulator-generated online rollouts for offline policy development',
         'status': 'documented_with_explicit_assumptions',
         'remaining': 'Independent discharge/temperature validation depends on new measurements'},
        {'reviewer': 'R2.1; R3 local Bellman and mixer', 'concern': 'individual architectural contribution',
         'evidence': 'E4 recurrent/mixer/hypernetwork/sharing/reward ablations and E3 IQL',
         'status': 'numerical_evidence_generated_review_results',
         'remaining': 'Must interpret failures and uncertainty instead of assuming every component helps'},
        {'reviewer': 'R2.4', 'concern': 'scalability and computation',
         'evidence': 'E6 N=3/6/12 separately trained closed-loop experiments and large-N synthetic CPU actor timing',
         'status': 'bounded_numerical_evidence',
         'remaining': 'Large-N timing is not large-pack control, electrical topology, thermal coupling or hardware evidence'},
        {'reviewer': 'R2.5; R3 unit gains and epsilon', 'concern': 'sensitivity of reward and exploration',
         'evidence': 'E5 predeclared retraining with unit gains, fixed normalization, coefficient changes and epsilon=1',
         'status': 'numerical_evidence_generated_review_results',
         'remaining': 'Limited sensitivity grid does not establish universal robustness'},
        {'reviewer': 'R2.6; R3 convergence', 'concern': 'training instability and convergence',
         'evidence': 'E3-E6 independent seeds, learning histories, failed trajectories, costs and held-out tests',
         'status': 'empirical_stability_only',
         'remaining': 'No theoretical convergence or global safe invariant-set proof'},
        {'reviewer': 'R2.7', 'concern': 'energy, delivered charge and health',
         'evidence': 'E2-E6 terminal input Wh, delivered Ah, temperature and constraint integrals',
         'status': 'partial_metrics_only',
         'remaining': 'Input energy is not energy loss; no calibrated degradation model or life extension result'},
        {'reviewer': 'R2.8', 'concern': 'imitation learning',
         'evidence': 'Literature discussion and choice justification remain manuscript tasks',
         'status': 'not_empirically_addressed',
         'remaining': 'This code does not implement IL/DAgger and cannot establish RL superiority over IL'},
        {'reviewer': 'R2.3; R3 embedded hardware', 'concern': 'online physical control and MCU deployment',
         'evidence': 'No physical experiments in the agreed scope',
         'status': 'not_addressed_hardware_excluded',
         'remaining': 'PC timing is not MCU timing; no actual sensing/actuation latency measurement'},
        {'reviewer': 'R3 SOC inequality, language and method justification',
         'concern': 'mathematical definitions and manuscript clarity',
         'evidence': 'Code distinguishes lower target SOC from upper SOC safety bound',
         'status': 'requires_manuscript_revision',
         'remaining': 'Edit equations, architecture rationale, non-personal language and cited literature'},
    ]
    dump(out / 'reviewer_evidence_chain.json', {'upstream_assessments': upstream,
        'evidence_map': evidence, 'all_reviewer_comments_resolved': False,
        'computation_complete_does_not_establish_superiority': True})
    write_csv(out / 'reviewer_evidence_map.csv', evidence)
    lines = [
        '# 实验 3—6 证据链与实验 6 解释边界', '',
        '本文件汇总结果来源，不自动宣布审稿意见已全部解决，也不按测试结果挑选论文结论。', '',
        'E1 辨识模型 → E2 确定任务与安全口径 → E3 公平基线 → '
        'E4 组件消融 → E5 奖励/探索敏感性和冻结泛化 → E6 规模与计算成本。', '',
        '## 规模实验', '',
        f'闭环规模：{cfg["closed_loop_pack_sizes"]}；共保存 {record_count} 个测试回合。',
        '每个规模重新训练 QMIX/MAPPO，使用同一交互预算与种子；'
        '基于 3 节实测辨识参数复制扩展，不是新增实测电芯。'
        '训练和最终测试均先生成 3 节 SOC 三元组再同步复制，保持每节 SOC 分布；'
        '训练随机初态不独立抽取 N 个 SOC。不能据此宣称复杂异质大包泛化。',
        '电芯各自有 0—7.5 A 控制权限，未建模共用供电、串联拓扑或电芯间热耦合。'
        '各规模延续同一奖励公式，其中电压/温度惩罚按电芯求和；'
        '因此总惩罚随规模变化，不应把全部训练难度变化归因于网络复杂度。', '',
        '## 计算成本', '',
        '训练时间包含物理仿真、优化和验证；测试总耗时与其分项由实际计时记录。'
        '不同规模的秒数必须在同一机器、线程设置下比较。',
        'inference_timings.csv 的纯策略计时使用合成观测：已训练 N=3/6/12 '
        '与较大规模未训练网络分开标注。它不包含物理模型、安全筛选、通信与硬件延迟。'
        '大规模未训练网络的速度不能作为控制成功率或泛化结果。',
        '参数量默认为驻留网络总量，可能包括混合器、价值网络和目标网络；'
        '只有另列的 actor 参数量可用于讨论部署网络规模。', '',
        '## 审稿意见对应', '',
        '| 审稿意见 | 本轮证据 | 仍需处理 |', '|---|---|---|',
    ]
    for item in evidence:
        lines.append(f'| {item["reviewer"]} | {item["evidence"]} | {item["remaining"]} |')
    lines += ['', '所有安全、速度和均衡结论必须读取成功率、失败样本、置信区间与配对比较后形成。'
                  '硬件在线控制、嵌入式部署、IL 比较、寿命模型及全局收敛证明不在本代码的完成范围。']
    (out / 'E3_E6_EVIDENCE_CHAIN.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(root: Path, out: Path, cfg: dict):
    ctx = load_chain(root, out, cfg, required=[1, 2, 3, 4, 5])
    effective = deepcopy(ctx['base_cfg'])
    effective.update(cfg)
    # Match the per-cell SOC distribution in both training and held-out tests.
    # The core draws one three-cell random SOC scenario, then repeats it to N.
    effective['replicate_training_triplet'] = True
    dump(out / 'effective_config.json', effective)
    vectors = np.asarray(ctx['vectors'], dtype=float)
    if vectors.shape != (3, 8):
        raise ValueError('E6 expands the three calibrated E1 vectors only')
    sizes = [int(n) for n in effective['closed_loop_pack_sizes']]
    timing_sizes = [int(n) for n in effective['inference_only_pack_sizes']]
    if not sizes or 3 not in sizes:
        raise ValueError('Closed-loop sizes must include the N=3 reference')
    if len(set(sizes)) != len(sizes) or len(set(timing_sizes)) != len(timing_sizes):
        raise ValueError('Pack size lists must not contain duplicate sizes')
    if set(sizes) & set(timing_sizes):
        raise ValueError('Inference-only sizes must be separate from closed-loop sizes')
    for n_agents in sizes + timing_sizes:
        _tile_vectors(vectors, n_agents)
    seeds = [int(seed) for seed in effective['seeds']]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide at least one unique training seed')
    if effective['validation_seed'] == effective['test_seed']:
        raise ValueError('Validation and test seeds must differ')
    torch.set_num_threads(int(effective['cpu_threads']))
    dump(out / 'machine.json', {
        'platform': platform.platform(), 'processor': platform.processor(),
        'logical_cpu_count': os.cpu_count(), 'python': platform.python_version(),
        'torch': version('torch'), 'numpy': version('numpy'),
        'cpu_threads': torch.get_num_threads(), 'device': 'cpu',
        'timing_clock': 'time.perf_counter_ns monotonic wall time',
        'host_load_caveat': 'No control over OS scheduling or other foreground processes.',
    })
    specs = []
    for method in ('qmix', 'mappo'):
        matching = [entry for entry in ctx['registry']
                    if entry.get('method_id', entry['spec']['id']) == method
                    and int(entry['n_agents']) == 3]
        rates = {float(entry['config']['learning_rate']) for entry in matching}
        if len(rates) != 1:
            raise ValueError(f'E6 needs one E3 validation-selected learning rate '
                             f'for {method}, consistent across seeds.')
        specs.append({'id': method, 'method': method, 'shield': True,
                      'learning_rate': next(iter(rates))})
    protocol = {
        'experiment': 6, 'closed_loop_pack_sizes': sizes,
        'inference_only_pack_sizes': timing_sizes, 'training_seeds': seeds,
        'specs': specs, 'same_training_steps_every_size': effective['training_steps'],
        'new_training_per_size': True, 'zero_shot_transfer_across_sizes': False,
        'hyperparameter_source': 'Each method retains its E3 validation-selected '
                                 'learning rate at every N; no tuning on E6 tests.',
        'topology': 'Synthetic independently actuated replicas of three calibrated '
                    'cells; no series-circuit coupling or shared power limit.',
        'thermal_coupling': False, 'new_measured_cell_data': False,
        'matched_difficulty': 'Both training and held-out tests draw a 3-cell SOC '
                              'scenario and repeat it to N. The same scenario RNG '
                              'seed is used at every N, keeping the per-cell SOC '
                              'distribution fixed; random training initializations '
                              'do not independently draw N SOCs.',
        'reward_scaling': 'Original per-cell voltage/temperature penalties remain '
                          'summed, time and pack spread remain global; this fact is '
                          'explicit in cross-size interpretation.',
        'safety_timing': 'Record numerical plant, shield, and policy costs '
                         'separately where core timers are available.',
        'inference_scope': 'CPU act() on fixed synthetic observations; no claim '
                           'about closed-loop safety of untrained large-N networks.',
        'authored_without_execution': True,
    }
    dump(out / 'protocol.json', protocol)
    case_sets = {}
    for n_agents in sizes:
        validation = _scaled_cases(vectors, n_agents, effective,
            effective['validation_seed'], effective['validation_count'], 'E6_validation')
        tests = _scaled_cases(vectors, n_agents, effective,
            effective['test_seed'], effective['test_count'], 'E6_test')
        case_sets[n_agents] = (validation, tests)
        dump(out / f'N{n_agents}' / 'validation_cases.json', validation)
        dump(out / f'N{n_agents}' / 'test_cases.json', tests)

    registry, records, training_cost, timing = [], [], [], []
    for n_agents in sizes:
        plant_vectors = _tile_vectors(vectors, n_agents)
        validation, tests = case_sets[n_agents]
        for spec in specs:
            for seed in seeds:
                directory = out / f'N{n_agents}' / spec['id'] / f'seed_{seed}'
                start = time.perf_counter()
                entry = train_policy(spec, plant_vectors, effective,
                                     directory / 'training', validation, seed)
                training_s = time.perf_counter() - start
                registry.append(entry)
                dump(out / 'model_registry.json', registry)
                start = time.perf_counter()
                evaluated = evaluate_policy(entry, plant_vectors, effective, tests,
                                             directory / 'test')
                evaluation_s = time.perf_counter() - start
                for row in evaluated:
                    row['study_panel'] = 'closed_loop_scaling'
                records.extend(evaluated)
                cost = {'n_agents': n_agents, 'method_id': spec['id'], 'seed': seed,
                        'training_steps_budget': effective['training_steps'],
                        'training_environment_steps': entry.get('environment_steps'),
                        'training_agent_decisions':
                            entry['environment_steps'] * n_agents
                            if entry.get('environment_steps') is not None else None,
                        'training_wall_s_including_validation': training_s,
                        'evaluation_wall_s': evaluation_s,
                        'evaluated_episodes': len(evaluated)}
                for field in ('timing_policy_s', 'timing_safety_s', 'timing_plant_s',
                              'timing_setup_s', 'timing_total_s', 'decision_count'):
                    values = [r.get(field) for r in evaluated]
                    cost[field] = sum(values) if all(v is not None for v in values) else None
                training_cost.append(cost)
                write_csv(out / 'closed_loop_episode_metrics.csv', records)
                write_csv(out / 'training_and_evaluation_cost.csv', training_cost)
                policy = load_policy(entry)
                measurement = _benchmark(policy, n_agents, effective, spec['id'],
                                         True, entry['checkpoint'])
                measurement['seed'] = seed
                timing.append(measurement)
                dump(directory / 'policy_timing.json', measurement)
                write_csv(out / 'inference_timings.csv',
                          [{k: v for k, v in item.items() if k != 'timing_samples_ms'}
                           for item in timing])
                del policy

    # No simulation or reward learning is performed for this separate timing panel.
    for n_agents in timing_sizes:
        for spec in specs:
            seed = int(effective['inference_seed'])
            policy = make_policy(spec, n_agents, seed, effective)
            measurement = _benchmark(policy, n_agents, effective, spec['id'], False)
            measurement['seed'] = seed
            timing.append(measurement)
            dump(out / 'inference_only' / f'N{n_agents}' / f'{spec["id"]}.json', measurement)
            del policy
    write_csv(out / 'inference_timings.csv',
              [{k: v for k, v in item.items() if k != 'timing_samples_ms'} for item in timing])
    compare(records, out / 'paired_method_statistics', effective,
            references={'discrete': 'qmix'})
    summary = finalize(out, records, {
        'experiment': 6, 'closed_loop_pack_sizes': sizes,
        'formal_seed_count_met': len(seeds) >= 5,
        'inference_only_pack_sizes': timing_sizes,
        'inference_only_is_not_control_validation': True,
        'measured_cell_models': 3, 'larger_packs_are_synthetic_replicas': True,
        'no_new_hardware_or_degradation_evidence': True,
        'training_cost_file': 'training_and_evaluation_cost.csv',
        'timing_file': 'inference_timings.csv',
        'paired_method_statistics_directory': 'paired_method_statistics',
    })
    _closure(out, ctx, effective, timing, training_cost, len(records))
    return {'experiment': 6, 'trained_checkpoints': len(registry),
            'evaluated_episodes': len(records), 'timing_groups': len(timing),
            'assessment': summary, 'all_reviewer_comments_resolved': False}
