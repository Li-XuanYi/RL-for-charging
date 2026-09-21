"""Manual Linux/Windows entry: isolated CUDA environment, then one study."""
from pathlib import Path
import json
import os
import shutil
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENV = ROOT / '.venv-revision-gpu'
CACHE = ROOT / 'revision_cache' / 'gpu_environment'
env = os.environ.copy()
env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYBAMM_DISABLE_TELEMETRY='true',
           MPLBACKEND='Agg', MPLCONFIGDIR=str(ROOT/'revision_cache'/'matplotlib'),
           PIP_CACHE_DIR=str(ROOT/'revision_cache'/'pip'),
           XDG_CACHE_HOME=str(ROOT/'revision_cache'/'xdg'),
           PYTHONPYCACHEPREFIX=str(ROOT/'revision_cache'/'pycache'),
           CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')


def run(command):
    print('RUN:', subprocess.list2cmdline([str(x) for x in command]), flush=True)
    options={'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {}
    process=subprocess.Popen([str(x) for x in command],cwd=ROOT,env=env,**options)
    try:
        code=process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            process.send_signal(signal.CTRL_BREAK_EVENT if os.name=='nt' else signal.SIGINT)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        raise
    if code:raise subprocess.CalledProcessError(code,command)


def select_gpu():
    command = ['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
               '--format=csv,noheader,nounits']
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    devices = []
    for line in result.stdout.splitlines():
        index, uuid, free, utilization = [part.strip() for part in line.split(',')]
        devices.append(dict(index=index, uuid=uuid, free_MiB=int(free), utilization_pct=int(utilization)))
    # Respect scheduler visibility, then expose only one GPU to the coordinator.
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    def matches(value, device):
        return value == device['index'] or (value.startswith('GPU-') and device['uuid'].startswith(value))
    if visible is not None:
        allowed = [value.strip() for value in visible.split(',') if value.strip()]
        devices = [d for d in devices if any(matches(value, d) for value in allowed)]
        if not devices:
            raise RuntimeError('CUDA_VISIBLE_DEVICES does not map to a full GPU visible in nvidia-smi. '
                               'Use the scheduler allocation; MIG/ambiguous mappings need administrator setup.')
    explicit = os.environ.get('GPUS',os.environ.get('GPU_INDEX'))
    if explicit is not None:
        requested=[v.strip() for v in explicit.split(',') if v.strip()]
        if not requested or len(set(requested))!=len(requested):raise ValueError('GPUS must list distinct physical indices/UUIDs')
        selected=[]
        for value in requested:
            found=[d for d in devices if matches(value,d)]
            if len(found)!=1:raise RuntimeError('Requested GPU not inside scheduler visibility: '+value)
            selected+=found
        if len({d['uuid'] for d in selected})!=len(selected):raise ValueError('Duplicate GPU aliases')
        devices=selected
    busy_allowed = env.get('ALLOW_BUSY_GPU') == '1'
    per_gpu=int(env.get('WORKERS_PER_GPU','1'))
    maximum=int(env.get('MAX_WORKERS','6'))
    if per_gpu<1 or maximum<1:raise ValueError('WORKERS_PER_GPU and MAX_WORKERS must be positive')
    eligible = [d for d in devices if d['free_MiB'] >= 4096*per_gpu and
                (d['utilization_pct'] <= 30 or (busy_allowed and explicit is not None))]
    if not eligible:
        raise RuntimeError('No available GPU: require >=4096 MiB free and <=30% utilization. '
                           'Wait for an allocated/idle card. To intentionally share your allocated card, '
                           'set GPUS and ALLOW_BUSY_GPU=1. No existing jobs are stopped.')
    if explicit is not None and len(eligible)!=len(devices):
        raise RuntimeError('One of the explicitly requested cards is busy or lacks memory. Select available cards in GPUS.')
    selected=sorted(eligible,key=lambda d:(d['utilization_pct'],-d['free_MiB']))
    cpu_count=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    if env.get('SLURM_CPUS_PER_TASK'):cpu_count=min(cpu_count,int(env['SLURM_CPUS_PER_TASK']))
    slots=[d['uuid'] for _ in range(per_gpu) for d in selected]
    limit=min(maximum,max(1,cpu_count//4),len(slots))
    if os.name!='nt':
        available=[int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')]
        if available:limit=min(limit,max(1,available[0]//(4*1024*1024)))
    slots=slots[:limit]
    env['CUDA_VISIBLE_DEVICES']=slots[0]
    env['REVISION_GPU_SLOTS']=json.dumps(slots)
    env['REVISION_GPU_SELECTION']=json.dumps(selected)
    print('GPU JOB SLOTS:',slots,'CPU allocation:',cpu_count,flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE/'last_device_selection.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')


def python_environment():
    python = ENV / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        candidates = [[sys.executable]]
        if env.get('REPRO_PYTHON'):
            candidates.insert(0, [env['REPRO_PYTHON']])
        if os.name == 'nt' and shutil.which('py'):
            candidates.extend([['py', '-3.12'], ['py', '-3.11']])
        for name in ('python3.12', 'python3.11'):
            if shutil.which(name):
                candidates.append([name])
        base = None
        for command in candidates:
            try:
                probe = subprocess.run(command + ['-c',
                    'import sys; print(sys.executable); sys.exit(0 if sys.version_info[:2] in [(3,11),(3,12)] else 1)'],
                    env=env, capture_output=True, text=True)
                if probe.returncode == 0:
                    base = probe.stdout.strip()
                    break
            except OSError:
                continue
        if base is None:
            raise RuntimeError('Install Python 3.11/3.12 with venv, or set REPRO_PYTHON to its executable.')
        run([base, '-m', 'venv', ENV])
    requirement = CACHE/'requirements-without-torch.txt'
    lines = (HERE/'requirements-lock.txt').read_text(encoding='utf-8-sig').splitlines()
    requirement.write_text('\n'.join(s for s in lines if not s.startswith('torch=='))+'\n', encoding='utf-8')
    check = (
        "import importlib.metadata as m,sys; from pathlib import Path; "
        "lines=Path(sys.argv[1]).read_text().splitlines(); "
        "assert all(m.version(a)==b for a,b in (s.split('==') for s in lines if s and not s.startswith('#'))); "
        "assert m.version('torch')=='2.7.1+cu128'"
    )
    inspected = subprocess.run([str(python), '-c', check, str(requirement)], env=env,
                               capture_output=True, text=True)
    if inspected.returncode != 0:
        run([python, '-m', 'pip', 'install', '--disable-pip-version-check', '--index-url',
             env.get('REPRO_PIP_INDEX_URL', 'https://pypi.org/simple'), '-r', requirement])
        run([python, '-m', 'pip', 'install', '--disable-pip-version-check', '--index-url',
             'https://download.pytorch.org/whl/cu128', 'torch==2.7.1+cu128'])
        run([python, '-c', check, requirement])
    run([python, '-m', 'pip', 'check'])
    return python


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('3', '4', '5', '6'):
        raise ValueError('Expected experiment number 3, 4, 5 or 6')
    CACHE.mkdir(parents=True, exist_ok=True)
    # Prevent these four launchers overlapping on this project, including setup.
    # OS locks release on process exit; no stale lock needs to be deleted manually.
    with (CACHE/'launch.lock').open('a+b') as lock:
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another GPU study launcher is active in this project. Run E3 -> E4 -> E5 -> E6 serially.') from exc
        select_gpu()
        python = python_environment()
        # Installation may take time: recheck allocation before touching CUDA.
        select_gpu()
        run([python, '-c',
            "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
            "assert torch.version.cuda=='12.8'; "
            "x=torch.ones((16,16),device='cuda:0'); y=x@x; torch.cuda.synchronize(); "
            "assert bool(torch.isfinite(y).all()); "
            "print('CUDA READY:',torch.__version__,torch.cuda.get_device_name(0))"])
        run([python, '-u', HERE/'run_study.py', *sys.argv[1:]])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted. Partial results are retained; not marked completed.')
        sys.exit(130)
    except Exception as exc:
        print('FAILED:', exc, flush=True)
        sys.exit(1)
