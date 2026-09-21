"""Re-evaluate recorded optimizer fits on a finer mesh; never forge a pass.

This helper does not optimize, relax thresholds or update old result folders.
All E1 predictions, engineering screens and numerical checks still run normally.
"""
import json
from pathlib import Path
import re
import shutil
import numpy as np
from common import dump,sha256,inside


def prepare(root,out,cfg,audit):
    if not cfg.get('reuse_optimizer_result'):
        return None
    source=inside(Path(cfg['reuse_optimizer_result']),Path(root)/'revision_results'/'experiment_1')
    if source==Path(out).resolve():raise ValueError('Recovery needs a separate new result directory')
    expected=cfg.get('reuse_source_hashes',{})
    mandatory={'config.json','status.json','summary.json','protocol.json','data_audit.json',
               'metrics.json','source_snapshot/model.py','source_snapshot/original_parameters.json'}
    if not mandatory.issubset(expected):raise ValueError('Recovery requires the recorded source hashes')
    for name,digest in expected.items():
        if sha256(inside(source/name,source))!=digest:
            raise ValueError('Recovery source changed: '+name)
    load=lambda name:json.loads((source/name).read_text(encoding='utf-8-sig'))
    old_cfg=load('config.json');old_protocol=load('protocol.json');old_summary=load('summary.json')
    # The effective protocol includes original default values missing in config.json.
    old_effective=old_protocol['config']
    ignored={'protocol_version','scope','reuse_optimizer_result','reuse_source_hashes'}
    for key in (set(old_effective)|set(cfg))-ignored:
        if old_effective.get(key)!=cfg.get(key):
            raise ValueError('Recovery must preserve original fitting settings and thresholds: '+key)
    if not old_summary['quality'].get('computation_completed'):
        raise ValueError('Original E1 fitting did not finish; frozen-parameter recovery is unavailable')
    if load('data_audit.json')!=audit:
        raise ValueError('Raw measurements or metadata changed; repeat identification instead of reusing fits')
    here=Path(__file__).resolve().parent
    if sha256(here/'original_parameters.json')!=expected['source_snapshot/original_parameters.json']:
        raise ValueError('Physical parameter dictionary changed; cannot reuse fits')
    def remove_mesh_assignments(text):
        return re.sub(r'^(?:MESH|REFINED_MESH)\s*=.*$', '', text, flags=re.MULTILINE).strip()
    before=(source/'source_snapshot'/'model.py').read_text(encoding='utf-8-sig')
    after=(here/'model.py').read_text(encoding='utf-8-sig')
    if remove_mesh_assignments(before)!=remove_mesh_assignments(after):
        raise ValueError('Recovery permits mesh-only model changes; repeat fitting after physical model changes')
    expected_fits={f'current_{float(current):g}A/cell_{cell}/{kind}/seed_{seed}/fit.json'
                   for current in cfg['currents'] for cell in (1,2,3)
                   for kind in ('full_fit','prefix_fit') for seed in cfg['seeds']}
    if not expected_fits.issubset(expected):raise ValueError('Missing recorded optimizer fit hashes')
    record={'source':str(source),'source_hashes':expected,'original_optimization_mesh':old_protocol['solver']['mesh'],
            'parameters_refitted_on_new_mesh':False,'new_optimizer_runs':0,
            'interpretation':'Frozen coarse-mesh optimizer parameters and seed selection are reused. All predictions, fit errors and precision gates are recomputed on the new mesh. Not fine-mesh optimal parameter identification.'}
    dump(Path(out)/'fit_recovery.json',record)
    print('RECOVERY: reusing saved optimizer parameters; recalculating predictions and unchanged gates.',flush=True)
    return record


def read_fit(record,current,cell,kind,seed,n_train,target):
    source=Path(record['source'])
    relative=f'current_{float(current):g}A/cell_{cell}/{kind}/seed_{seed}/fit.json'
    path=inside(source/relative,source)
    if sha256(path)!=record['source_hashes'][relative]:raise ValueError('Optimizer fit changed: '+relative)
    fit=json.loads(path.read_text(encoding='utf-8-sig'))
    if fit['seed']!=seed or fit['training_samples']!=n_train:
        raise ValueError('Saved fit seed or chronological split differs')
    vector=np.asarray(fit['vector'],dtype=float)
    if vector.shape!=(8,) or not np.isfinite(vector).all():raise ValueError('Invalid saved parameter vector')
    if not isinstance(fit['success'],bool) or not np.isfinite(fit['objective']):raise ValueError('Invalid saved fit objective')
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    # Preserve the original PSO result exactly; its objective is explicitly old-mesh.
    shutil.copy2(path,target/'fit.json')
    dump(target/'recovery_origin.json',{'source_fit':str(path),'sha256':record['source_hashes'][relative],
         'original_optimization_mesh':record['original_optimization_mesh'],'new_optimizer_run':False})
    print(f'REUSE cell={cell} current={current:g}A {kind} seed={seed}',flush=True)
    return fit
