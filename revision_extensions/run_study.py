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
    args=parser.parse_args()
    config_path=args.config or paths.HERE/'configs'/f'experiment_{args.experiment}.json'
    cfg=json.loads(config_path.read_text(encoding='utf-8-sig'))
    base=paths.ROOT/'revision_results'/f'experiment_{args.experiment}'
    out=base/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    os.environ['MPLCONFIGDIR']=str(paths.ROOT/'revision_cache'/'matplotlib')
    os.environ['XDG_CACHE_HOME']=str(paths.ROOT/'revision_cache'/'xdg')
    dump(out/'requested_config.json',cfg)
    dump(out/'status.json',{'status':'running','experiment':args.experiment})
    (base/'LATEST_STARTED.txt').write_text(str(out),encoding='utf-8')
    files=[]
    for source_root,label in [(paths.HERE,'revision_extensions'),(paths.BASE,'revision_experiments')]:
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
    with (out/'run.log').open('w',encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
            print('OUTPUT:',out,flush=True)
            try:
                module=importlib.import_module(f'experiment{args.experiment}')
                result=module.run(paths.ROOT,out,cfg)
                dump(out/'status.json',{'status':'completed','experiment':args.experiment,'elapsed_s':time.perf_counter()-start,
                    'result':result,'meaning':'Computations finished; inspect scientific_assessment.json for evidence and limitations.'})
                (base/'LATEST_COMPLETED.txt').write_text(str(out),encoding='utf-8')
                print('COMPLETED:',out,flush=True)
            except BaseException as exc:
                dump(out/'status.json',{'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                    'error':str(exc),'type':type(exc).__name__,'elapsed_s':time.perf_counter()-start})
                traceback.print_exc();raise


if __name__=='__main__':main()
