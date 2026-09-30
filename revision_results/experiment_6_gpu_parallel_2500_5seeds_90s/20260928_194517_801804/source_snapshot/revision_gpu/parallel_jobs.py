"""Independent jobs, separate processes and outputs, one visible GPU per worker."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import paths
from common import dump


def train_job(spec,vectors,cfg,training_out,validation,seed,tests,test_out):
    return dict(kind='train_evaluate',spec=spec,vectors=vectors.tolist(),cfg=cfg,
                training_out=str(training_out),validation=validation,seed=int(seed),
                tests=tests,test_out=str(test_out))


def evaluation_job(entry,vectors,cfg,tests,test_out):
    return dict(kind='frozen_evaluate',entry=entry,vectors=vectors.tolist(),cfg=cfg,
                tests=tests,test_out=str(test_out))


def execute_jobs(jobs,out,cfg):
    """Return in declared order, regardless of nondeterministic completion order."""
    from study_core import code_fingerprint
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    slots=json.loads(os.environ.get('REVISION_GPU_SLOTS','[]'))
    if not slots:
        raise RuntimeError('Use the GPU .sh/.cmd launcher; no GPU worker allocation found')
    dump(out/'scheduler.json',{'gpu_slots':slots,'job_count':len(jobs),
        'parallel_unit':'independent algorithm/seed/pack-size job; no distributed-gradient training',
        'per_worker_cpu_threads':cfg['cpu_threads'],'order':'results merged in declared order'})
    fingerprint=code_fingerprint();queue=[];results={};active={}
    for index,job in enumerate(jobs):
        task={**job,'code_fingerprint':fingerprint}
        signature=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()
        folder=out/f'job_{index:04d}';folder.mkdir(exist_ok=True)
        result_file=folder/'result.json'
        if result_file.exists():
            existing=json.loads(result_file.read_text(encoding='utf-8'))
            if existing.get('signature')!=signature:
                raise ValueError('Existing worker result has another configuration/source: '+str(folder))
            # Only reuse trusted complete outputs with unchanged checkpoint bytes.
            from common import sha256
            from portable import project_path
            entry=existing['entry']
            if entry['checkpoint'] and sha256(project_path(entry['checkpoint']))!=entry['checkpoint_sha256']:
                raise ValueError('Completed worker checkpoint was modified')
            results[index]=existing
            print('REUSE COMPLETED JOB:',index,folder,flush=True)
            continue
        task.update(signature=signature,result_file=str(result_file))
        dump(folder/'task.json',task);queue.append((index,folder))
    last_progress=0.
    try:
        while queue or active:
            for slot,gpu in enumerate(slots):
                if slot in active or not queue:continue
                index,folder=queue.pop(0)
                worker_env=os.environ.copy()
                worker_env.update(CUDA_VISIBLE_DEVICES=gpu,REVISION_WORKER_GPU=gpu,
                    OMP_NUM_THREADS=str(cfg['cpu_threads']),MKL_NUM_THREADS=str(cfg['cpu_threads']),
                    MPLCONFIGDIR=str(paths.ROOT/'revision_cache'/'matplotlib_workers'/str(slot)))
                log=(folder/'worker.log').open('w',encoding='utf-8')
                try:
                    process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                        str(folder/'task.json')],cwd=paths.ROOT,env=worker_env,stdout=log,stderr=subprocess.STDOUT)
                except BaseException:
                    log.close();raise
                active[slot]=(process,log,index,folder)
                print(f'START JOB {index+1}/{len(jobs)} GPU={gpu} LOG={folder/"worker.log"}',flush=True)
            for slot,(process,log,index,folder) in list(active.items()):
                code=process.poll()
                if code is None:continue
                log.close();del active[slot]
                if code!=0:
                    raise RuntimeError(f'Worker {index} failed (exit {code}); inspect {folder/"worker.log"}')
                result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
                results[index]=result
                print(f'FINISH JOB {index+1}/{len(jobs)} ({len(results)} complete)',flush=True)
            now=time.monotonic()
            if now-last_progress>=30:
                dump(out/'progress.json',{'completed':len(results),'total':len(jobs),
                    'active':[{'index':x[2],'log':str(x[3]/'worker.log'),'pid':x[0].pid} for x in active.values()]})
                print(f'PARALLEL PROGRESS: {len(results)}/{len(jobs)} complete; {len(active)} active',flush=True)
                last_progress=now
            if active:time.sleep(.2)
    finally:
        # Only terminate child processes created by this scheduler; never other users' jobs.
        for process,log,index,folder in active.values():
            if process.poll() is None:process.terminate()
        for process,log,index,folder in active.values():
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            log.close()
    dump(out/'progress.json',{'completed':len(results),'total':len(jobs),'active':[]})
    return [results[index] for index in range(len(jobs))]


def worker(task_file):
    import signal
    if os.name=='nt':signal.signal(signal.SIGBREAK,signal.default_int_handler)
    import numpy as np
    import torch
    from common import sha256
    from portable import project_path
    from study_core import train_policy,evaluate_policy,code_fingerprint
    task=json.loads(Path(task_file).read_text(encoding='utf-8-sig'))
    if task['code_fingerprint']!=code_fingerprint():raise ValueError('Source changed after dispatch')
    cfg=task['cfg'];cfg['device']='cuda:0';torch.set_num_threads(int(cfg['cpu_threads']))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    vectors=np.asarray(task['vectors'],dtype=float)
    training_s=0.;started=time.perf_counter()
    if task['kind']=='train_evaluate':
        entry=train_policy(task['spec'],vectors,cfg,Path(task['training_out']),task['validation'],task['seed'])
        training_s=time.perf_counter()-started
    elif task['kind']=='frozen_evaluate':entry=task['entry']
    else:raise ValueError('Unknown worker task kind')
    device_record={'gpu_uuid':os.environ.get('REVISION_WORKER_GPU'),
                   'gpu_name':torch.cuda.get_device_name(0),'torch':torch.__version__,
                   'cuda_runtime':torch.version.cuda,'cpu_threads':torch.get_num_threads()}
    dump(Path(task_file).parent/'worker_device.json',device_record)
    digest=sha256(project_path(entry['checkpoint'])) if entry['checkpoint'] else None
    if digest!=entry.get('checkpoint_sha256'):raise ValueError('Checkpoint hash mismatch before evaluation')
    started=time.perf_counter()
    records=evaluate_policy(entry,vectors,cfg,task['tests'],Path(task['test_out']))
    evaluation_s=time.perf_counter()-started
    after=sha256(project_path(entry['checkpoint'])) if entry['checkpoint'] else None
    if after!=digest:raise ValueError('Evaluation changed checkpoint')
    result={'entry':entry,'records':records,'training_s':training_s,'evaluation_s':evaluation_s,
        'signature':task['signature'],'worker_gpu':os.environ.get('REVISION_WORKER_GPU'),'worker_device':device_record,
        'audit':{'method_id':entry['method_id'],'seed':entry['seed'],'checkpoint':entry['checkpoint'],
                 'sha256_before':digest,'sha256_after':after,'unchanged':True}}
    target=Path(task['result_file']);temporary=target.with_suffix('.tmp')
    dump(temporary,result);os.replace(temporary,target)


if __name__=='__main__':worker(sys.argv[1])
