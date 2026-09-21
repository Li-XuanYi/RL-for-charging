"""Predeclared descriptive metrics; no claim of continuous-time safety from samples."""
import math
import numpy as np
from common import write_csv, dump


def metrics(rows, decisions, goal_time, failure_reason, soc_max=.95):
    if not rows:
        return {'task_success': False, 'failure_reason': failure_reason or 'no_trajectory',
                'charging_target_reached': False, 'constraints_satisfied': False}
    # Dense rows have three cells for each sample. Preserve both sides of jumps.
    by_cell = [[r for r in rows if r['cell'] == i] for i in (1, 2, 3)]
    charge = [[r for r in part if r['phase'] == 'charge'] for part in by_cell]
    time = np.asarray([r['time_s'] for r in charge[0]], dtype=float)
    if any(len(part) != len(charge[0]) for part in charge):
        raise ValueError('Cell traces must have identical sample times')
    if any(not np.allclose(time, [r['time_s'] for r in part]) for part in charge):
        raise ValueError('Cell traces are not synchronous')
    soc = np.column_stack([[r['soc'] for r in part] for part in charge])
    std = soc.std(axis=1)
    balanced = std <= .02 + 1e-10
    first = int(np.flatnonzero(balanced)[0]) if balanced.any() else None
    bad = np.flatnonzero(~balanced)
    sustained = (int(bad[-1]) + 1 if len(bad) else 0) if balanced[-1] else None
    v = np.array([r['voltage_V'] for r in rows], dtype=float)
    temp = np.array([r['temperature_K'] for r in rows], dtype=float)
    allsoc = np.array([r['soc'] for r in rows], dtype=float)
    finite = bool(np.isfinite(v).all() and np.isfinite(temp).all() and np.isfinite(allsoc).all())
    v_excess = np.maximum(v - 4.2, 0)
    t_excess = np.maximum(temp - 309, 0)
    soc_excess = np.maximum(allsoc - soc_max, 0)
    constraints = bool(finite and np.max(v_excess) <= 1e-5 and np.max(t_excess) <= 1e-5
                       and np.max(soc_excess) <= 1e-7 and np.min(allsoc) >= -1e-7
                       and not failure_reason)
    hold = [[r for r in part if r['phase'] == 'hold'] for part in by_cell]
    hold_ok = False
    hold_duration = 0.
    if all(hold):
        held_soc = np.column_stack([[r['soc'] for r in part] for part in hold])
        hold_duration = float(hold[0][-1]['time_s'] - hold[0][0]['time_s'])
        hold_ok = bool(np.all(held_soc.min(axis=1) >= .9 - 1e-7)
                       and np.all(held_soc.std(axis=1) <= .02 + 1e-7))
    ah = []; wh = []; v_duration = []; t_duration = []; v_area = []; t_area = []
    for part in by_cell:
        x = np.asarray([r['time_s'] for r in part])
        current = np.asarray([r['current_A'] for r in part])
        voltage = np.asarray([r['voltage_V'] for r in part])
        temperature = np.asarray([r['temperature_K'] for r in part])
        ah.append(float(np.trapz(current, x) / 3600))
        wh.append(float(np.trapz(current * voltage, x) / 3600))
        v_duration.append(float(np.trapz((voltage > 4.2 + 1e-5).astype(float), x)))
        t_duration.append(float(np.trapz((temperature > 309 + 1e-5).astype(float), x)))
        v_area.append(float(np.trapz(np.maximum(voltage - 4.2, 0), x)))
        t_area.append(float(np.trapz(np.maximum(temperature - 309, 0), x)))
    charge_decisions=[d for d in decisions if d.get('phase','charge')=='charge']
    return {
        'charging_target_reached': goal_time is not None,
        'charging_time_s': goal_time,
        'task_success': bool(goal_time is not None and constraints and hold_ok),
        'constraints_satisfied': constraints,
        'terminal_hold_soc_satisfied': hold_ok, 'terminal_hold_observed_s': hold_duration,
        'failure_reason': failure_reason,
        'first_balance_sample_s': float(time[first]) if first is not None else None,
        'sustained_balance_start_s': float(time[sustained]) if sustained is not None else None,
        'sustained_balance_charging_duration_s': float(time[-1] - time[sustained]) if sustained is not None else 0.,
        'balance_maintained_for_270s_during_charge': bool(sustained is not None and time[-1]-time[sustained] >= 270),
        'imbalance_after_first_area_soc_s': float(np.trapz(np.maximum(std[first:] - .02, 0), time[first:])) if first is not None else None,
        'rebalance_loss_count': int(np.sum(balanced[:-1] & ~balanced[1:])),
        'soc_std_integral_soc_s': float(np.trapz(std, time)),
        'final_charge_soc_1': float(soc[-1, 0]), 'final_charge_soc_2': float(soc[-1, 1]),
        'final_charge_soc_3': float(soc[-1, 2]), 'final_charge_soc_std': float(std[-1]),
        'maximum_soc_deviation_from_pack_mean': float(np.max(np.abs(soc - soc.mean(axis=1, keepdims=True)))),
        'max_voltage_V': float(v.max()), 'max_temperature_K': float(temp.max()),
        'max_voltage_excess_V': float(v_excess.max()), 'max_temperature_excess_K': float(t_excess.max()),
        'max_soc_excess': float(soc_excess.max()),
        'voltage_violation_cell_seconds_sample_estimate': sum(v_duration),
        'temperature_violation_cell_seconds_sample_estimate': sum(t_duration),
        'voltage_excess_integral_Vs': sum(v_area), 'temperature_excess_integral_Ks': sum(t_area),
        'cell1_charge_Ah': ah[0], 'cell2_charge_Ah': ah[1], 'cell3_charge_Ah': ah[2],
        'sum_cell_charge_Ah': sum(ah), 'sum_cell_terminal_input_Wh': sum(wh),
        'safety_intervention_steps': sum(bool(d.get('intervention', False)) for d in decisions),
        'protective_stop_count':sum(bool(d.get('protective_stop',False)) for d in decisions),
        'numerical_failure_count':sum(bool(d.get('numerical_failure',False)) for d in decisions),
        'post_switch_peak_unknown':any(bool(d.get('post_switch_peak_unknown',False)) for d in decisions),
        'recorded_maxima_complete':not any(bool(d.get('post_switch_peak_unknown',False)) for d in decisions),
        'decision_count': len(charge_decisions),'terminal_hold_step_count':len(decisions)-len(charge_decisions),
        'safety_intervention_fraction': sum(bool(d.get('intervention', False)) for d in charge_decisions)/max(1, len(charge_decisions)),
        'sample_finite': finite,
        'final_observed_time_s': float(max(r['time_s'] for r in rows)),
        'violation_measurement': 'dense sampled approximation incl switch endpoints; not certified continuous maxima',
    }


def summarize(records, out, cfg):
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    results = []
    groups = sorted(set((r['decision_s'], r['method'], r['test_family']) for r in records))
    for dt, method, family in groups:
        subset = [r for r in records if (r['decision_s'],r['method'],r['test_family']) == (dt,method,family)]
        # The deterministic rule is evaluated once, not falsely repeated as five seeds.
        seeds = sorted(set(r['seed'] for r in subset))
        seed_rates = np.array([np.mean([r['task_success'] for r in subset if r['seed']==s]) for s in seeds])
        ci = [None, None]
        if len(seeds) >= 2:
            draws = rng.choice(seed_rates, size=(cfg['bootstrap_replicates'],len(seeds)),replace=True).mean(axis=1)
            ci = np.quantile(draws,[.025,.975]).tolist()
        success_times = [r['charging_time_s'] for r in subset if r['task_success']]
        results.append(dict(decision_s=dt,method=method,test_family=family,
            independent_training_seeds=len(seeds) if method != 'rule_shield' else 0,
            evaluated_episodes=len(subset),success_rate=float(seed_rates.mean()),
            success_rate_seed_bootstrap_ci_low=ci[0],success_rate_seed_bootstrap_ci_high=ci[1],
            empirical_constraint_satisfaction_rate=float(np.mean([r.get('constraints_satisfied',False) for r in subset])),
            successful_only_mean_charging_s=float(np.mean(success_times)) if success_times else None,
            failed_or_censored_count=sum(not r['task_success'] for r in subset),
            caveat='Success time is conditional on success; not a speed ranking if success rates differ. CI covers training seeds on fixed test cases.'))
    write_csv(out/'group_summary.csv',results)
    paired = []
    lookup={(r['decision_s'],r['method'],r['seed'],r['case_id']):r for r in records}
    for row in records:
        if row['method']!='qmix_raw':
            continue
        candidate=lookup.get((row['decision_s'],'qmix_frozen_shield',row['seed'],row['case_id']))
        if candidate:
            paired.append({'decision_s':row['decision_s'],'seed':row['seed'],'case_id':row['case_id'],
                'raw_success':row['task_success'],'same_weights_shield_success':candidate['task_success'],
                'same_weights_success_difference':int(candidate['task_success'])-int(row['task_success']),
                'raw_max_voltage_V':row.get('max_voltage_V'),
                'same_weights_shield_max_voltage_V':candidate.get('max_voltage_V')})
    write_csv(out/'paired_same_policy_safety_effect.csv',paired)
    safe = [r for r in records if r['method'] in ('qmix_frozen_shield','qmix_trained_shield','rule_shield')]
    assessment={'computed':True,'test_cases_fixed_before_training':True,
        'all_safety_groups_sampled_constraints_satisfied':all(r.get('constraints_satisfied',False) for r in safe),
        'all_safety_groups_tasks_succeeded':all(r['task_success'] for r in safe),
        'any_safe_task_observed':any(r['task_success'] for r in safe),
        'continuous_time_or_hardware_guarantee':False,
        'efficiency_or_lifetime_improvement_proven':False,
        'safe_controller_is_finite_horizon_screening_not_a_certified_invariant_filter':True,
        'interpretation':'Read raw failures, safety interventions, success rates and terminal SOC; computational completion does not prove paper claims.'}
    dump(out/'scientific_assessment.json',assessment)
    return results,assessment


def plot_case(rows, path, title):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(11,7))
    for i in (1,2,3):
        part=[r for r in rows if r['cell']==i]
        for ax,key,label in zip(axes.flat,['soc','current_A','voltage_V','temperature_K'],['SOC','Current / A','Voltage / V','Temperature / K']):
            ax.plot([r['time_s'] for r in part],[r[key] for r in part],label=f'Cell {i}',lw=1)
            ax.set(xlabel='Time / s',ylabel=label);ax.grid(alpha=.2)
    axes[0,0].legend();axes[0,0].axhline(.9,color='gray',ls='--')
    axes[1,0].axhline(4.2,color='gray',ls='--');axes[1,1].axhline(309,color='gray',ls='--')
    fig.suptitle(title);fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
