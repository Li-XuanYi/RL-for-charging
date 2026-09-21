"""Manual entry point. Never launches work by import."""
import argparse
import contextlib
from datetime import datetime
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time
import traceback
os.environ.setdefault('PYBAMM_DISABLE_TELEMETRY','true')
os.environ.setdefault('MPLBACKEND','Agg')
import paths
from common import dump,sha256


class Tee:
    def __init__(self,stream,file):self.stream,self.file=stream,file
    def write(self,text):self.stream.write(text);self.file.write(text);self.file.flush()
    def flush(self):self.stream.flush();self.file.flush()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',choices=['3','4','5','6'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--resume',type=Path,help='Reuse complete independent jobs in a failed/interrupted run; partial jobs restart')
    args=parser.parse_args()
    config_path=args.config or paths.HERE/'configs'/f'experiment_{args.experiment}.json'
    cfg=json.loads(config_path.read_text(encoding='utf-8-sig'))
    tag=cfg.get('result_tag','')
    if not isinstance(tag,str) or (tag and (not tag.isascii() or not all(c.isalnum() or c=='_' for c in tag))):
        raise ValueError('Invalid result_tag')
    if cfg.get('exploratory') and not tag:raise ValueError('Exploratory results need a separate output tag')
    base=paths.ROOT/'revision_results'/(f'experiment_{args.experiment}'+(f'_{tag}' if tag else ''))
    if args.resume:
        from portable import project_path
        out=project_path(args.resume)
        if not out.is_relative_to(base.resolve()):raise ValueError('Resume directory must belong to this experiment/profile')
        previous=json.loads((out/'requested_config.json').read_text(encoding='utf-8-sig'))
        if previous!=cfg:raise ValueError('Resume requires exactly the same scientific configuration')
        provenance=json.loads((out/'provenance.json').read_text(encoding='utf-8-sig'))
        for record in provenance['source_files']:
            current=paths.ROOT/record['path'].replace('\\','/')
            if sha256(current)!=record['sha256']:raise ValueError('Resume source differs: '+str(current))
        previous_status=json.loads((out/'status.json').read_text(encoding='utf-8-sig'))
        if previous_status.get('status')=='completed':
            print('ALREADY COMPLETED:',out);return
    else:
        out=base/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    os.environ['MPLCONFIGDIR']=str(paths.ROOT/'revision_cache'/'matplotlib')
    os.environ['XDG_CACHE_HOME']=str(paths.ROOT/'revision_cache'/'xdg')
    dump(out/'requested_config.json',cfg)
    dump(out/'status.json',{'status':'running','experiment':args.experiment})
    (base/'LATEST_STARTED.txt').write_text(str(out),encoding='utf-8')
    files=[]
    for source_root,label in [(paths.HERE,'revision_gpu'),(paths.BASE,'revision_experiments')]:
        for source in source_root.rglob('*'):
            if source.is_file() and source.suffix in ('.py','.json','.md','.txt') and '__pycache__' not in source.parts:
                relative=Path(label)/source.relative_to(source_root)
                target=out/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target);files.append({'path':str(relative),'sha256':sha256(source)})
    dump(out/'provenance.json',{'python':sys.version,'platform':platform.platform(),
        'packages':{k:version(k) for k in ['numpy','torch','pybamm','casadi','scipy','matplotlib']},
        'source_files':files,'config_sha256':sha256(config_path),
        'delivery_validation':'Static review only. Assistant did not execute any Python/CMD/test or experiment.'})
    start=time.perf_counter()
    with (out/'run.log').open('a' if args.resume else 'w',encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
            print('OUTPUT:',out,flush=True)
            try:
                module=importlib.import_module(f'experiment{args.experiment}')
                result=module.run(paths.ROOT,out,cfg)
                if cfg.get('exploratory'):
                    dump(out/'EXPLORATORY_RUN.json',{'training_steps':cfg['training_steps'],'seeds':cfg['seeds'],
                        'decision_s':cfg['decision_s'],'formal_evidence':False,'device':cfg.get('device')})
                    report=out/'REPORT.md'
                    if report.exists():report.write_text('# 短预算 GPU 探索结果：不能作为收敛或算法优越性的最终证明\n\n'+report.read_text(encoding='utf-8-sig'),encoding='utf-8')
                dump(out/'status.json',{'status':'completed','experiment':args.experiment,'elapsed_s':time.perf_counter()-start,
                    'exploratory':bool(cfg.get('exploratory',False)),'result_tag':tag,
                    'result':result,'meaning':'Computations finished; inspect scientific_assessment.json for evidence and limitations.'})
                (base/'LATEST_COMPLETED.txt').write_text(str(out),encoding='utf-8')
                print('COMPLETED:',out,flush=True)
            except BaseException as exc:
                dump(out/'status.json',{'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                    'error':str(exc),'type':type(exc).__name__,'elapsed_s':time.perf_counter()-start})
                traceback.print_exc();raise


if __name__=='__main__':
    if os.name=='nt':signal.signal(signal.SIGBREAK,signal.default_int_handler)
    main()
