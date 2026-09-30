"""Environment setup happens only when the user launches one of the two CMD files."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENV = ROOT / '.venv-revision'
EXISTING = ROOT / '.venv-reproduce' / 'Scripts' / 'python.exe'
env = os.environ.copy()
env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYBAMM_DISABLE_TELEMETRY='true',
           MPLBACKEND='Agg', MPLCONFIGDIR=str(ROOT/'revision_cache'/'matplotlib'),
           PIP_CACHE_DIR=str(ROOT/'revision_cache'/'pip'),
           XDG_CACHE_HOME=str(ROOT/'revision_cache'/'xdg'),
           PYTHONPYCACHEPREFIX=str(ROOT/'revision_cache'/'pycache'))


def run(command):
    print(subprocess.list2cmdline([str(x) for x in command]), flush=True)
    subprocess.run([str(x) for x in command], cwd=ROOT, env=env, check=True)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('1', '2'):
        raise ValueError('Expected experiment number 1 or 2')
    (ROOT/'revision_cache').mkdir(exist_ok=True)
    requirement = HERE/'requirements-lock.txt'
    check = "import importlib.metadata as m,sys; from pathlib import Path; lines=Path(sys.argv[1]).read_text(encoding='utf-8-sig').splitlines(); assert all(m.version(a)==b for a,b in (s.split('==') for s in lines if s and not s.startswith('#')))"
    python = ENV/'Scripts'/'python.exe'
    candidates = [python, EXISTING]
    selected = None
    for candidate in candidates:
        if candidate.exists():
            inspected = subprocess.run([str(candidate), '-c', check, str(requirement)], env=env,
                                       capture_output=True, text=True)
            if inspected.returncode == 0:
                selected = candidate
                break
    if selected is None:
        if not python.exists():
            versions = [[sys.executable]]
            if os.environ.get('REPRO_PYTHON'):
                versions.insert(0, [os.environ['REPRO_PYTHON']])
            if shutil.which('py'):
                versions.extend([['py', '-3.12'], ['py', '-3.11']])
            base = None
            for command in versions:
                try:
                    probe = subprocess.run(command + ['-c', 'import sys; print(sys.executable); sys.exit(0 if sys.version_info[:2] in [(3,11),(3,12)] else 1)'],
                                           env=env, capture_output=True, text=True)
                    if probe.returncode == 0:
                        base = probe.stdout.strip()
                        break
                except OSError:
                    continue
            if base is None:
                raise RuntimeError('Python 3.11/3.12 required. Set REPRO_PYTHON to its full executable path.')
            run([base, '-m', 'venv', ENV])
        run([python, '-m', 'pip', 'install', '--disable-pip-version-check', '--index-url',
             env.get('REPRO_PIP_INDEX_URL', 'https://pypi.tuna.tsinghua.edu.cn/simple'), '-r', requirement])
        selected = python
    run([selected, '-m', 'pip', 'check'])
    run([selected, '-u', HERE/'run_experiment.py', *sys.argv[1:]])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted; partial files remain under revision_results.')
        sys.exit(130)
    except Exception as exc:
        print('FAILED:', exc)
        sys.exit(1)
