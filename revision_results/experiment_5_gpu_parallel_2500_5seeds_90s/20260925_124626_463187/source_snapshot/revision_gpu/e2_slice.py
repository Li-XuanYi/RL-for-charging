"""Audit the completed 90-second E2 subset; never alter the parent run status."""
import json
from pathlib import Path
from common import dump,sha256,inside
from portable import project_path


def audit(root,out,cfg):
    if not cfg.get('exploratory'):raise ValueError('Partial E2 evidence is exploratory only')
    manifest_path=project_path(cfg['upstream_e2_slice_manifest'])
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    source=inside(project_path(manifest['source_run']),Path(root)/'revision_results'/'experiment_2_exploratory_2500_2seeds')
    if float(manifest['decision_s'])!=float(cfg['decision_s']):raise ValueError('E2 subset decision period differs')
    expected=manifest['artifact_sha256']
    for name,digest in expected.items():
        if sha256(inside(source/name,source))!=digest:raise ValueError('E2 evidence changed: '+name)
    def load(name):
        if name not in expected:raise ValueError('Missing frozen evidence hash: '+name)
        return json.loads((source/name).read_text(encoding='utf-8-sig'))
    upstream_cfg=load('config.json');case_ids=[c['case_id'] for c in load('test_cases.json')]
    if len(set(case_ids))!=len(case_ids):raise ValueError('Duplicate E2 case IDs')
    if manifest['seeds']!=upstream_cfg['seeds']:raise ValueError('E2 subset must include all declared E2 seeds')
    period=f"decision_{float(manifest['decision_s']):g}s"
    records=[]
    for seed in manifest['seeds']:
        completed=load(f'{period}/seed_{seed}/completed.json')
        if completed['status']!='completed':raise ValueError('E2 seed not fully evaluated')
        for kind in ('raw','shield_aware'):
            relative=f'{period}/seed_{seed}/{kind}/selected.pt'
            if relative not in expected or project_path(completed['selected_checkpoints'][kind])!=(source/relative).resolve():
                raise ValueError('Selected E2 model does not match frozen evidence')
        for method in ('qmix_raw','qmix_frozen_shield','qmix_trained_shield'):
            for case_id in case_ids:
                record=load(f'{period}/seed_{seed}/test/{method}/{case_id}/metrics.json')
                if (record['method'],record['seed'],record['case_id'],record['decision_s'])!=(method,seed,case_id,manifest['decision_s']):
                    raise ValueError('E2 metric identity mismatch')
                records.append(record)
    for case_id in case_ids:
        record=load(f'{period}/rule_shield/{case_id}/metrics.json')
        if (record['method'],record['seed'],record['case_id'],record['decision_s'])!=('rule_shield',-1,case_id,manifest['decision_s']):
            raise ValueError('E2 rule identity mismatch')
        records.append(record)
    safe=[]
    for record in records:
        if type(record['task_success']) is not bool:raise ValueError('Invalid E2 success flag')
        if record['task_success'] and (not record['constraints_satisfied'] or not record['terminal_hold_soc_satisfied']
                or record['terminal_hold_observed_s']<upstream_cfg['hold_s']-1e-5):
            raise ValueError('E2 success lacks terminal hold/constraint evidence')
        if record['method']!='qmix_raw':safe.append(record)
    summary={method:{'cases':sum(r['method']==method for r in records),
                     'successes':sum(r['method']==method and r['task_success'] for r in records)}
             for method in sorted({r['method'] for r in records})}
    assessment={'any_safe_task_observed':any(r['task_success'] for r in safe),
        'parent_experiment_completed':False,'uses_completed_subset_only':True,'decision_s':manifest['decision_s'],
        'upstream_training_seeds':manifest['seeds'],'downstream_training_seeds':cfg['seeds'],
        'scope':'E2 90s diagnostic evidence only; does not certify other decision periods or training convergence',
        'summary':summary,'manifest_sha256':sha256(manifest_path),'source':str(source)}
    dump(Path(out)/'e2_completed_90s_evidence.json',assessment)
    dump(Path(out)/'e2_slice_manifest_used.json',manifest)
    return source,assessment
