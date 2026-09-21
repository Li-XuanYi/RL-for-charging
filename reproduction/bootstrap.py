"""Create an isolated environment beside the paper and run the experiment."""
from pathlib import Path
import os
import subprocess
import sys
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENV = ROOT / ".venv-reproduce"
PYTHON = ENV / "Scripts" / "python.exe"
env = os.environ.copy()
env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYBAMM_DISABLE_TELEMETRY="true",
           MPLBACKEND="Agg", MPLCONFIGDIR=str(ROOT/"reproduction_cache"/"matplotlib"),
           PIP_CACHE_DIR=str(ROOT/"reproduction_cache"/"pip"))


def run(command):
    print("Running:", subprocess.list2cmdline([str(x) for x in command]), flush=True)
    with subprocess.Popen([str(x) for x in command], env=env, cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace") as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def base_python():
    candidates = []
    if os.environ.get("REPRO_PYTHON"):
        candidates.append([os.environ["REPRO_PYTHON"]])
    candidates.extend([[sys.executable],
        [str(Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe")]])
    if shutil.which("py"):
        candidates += [["py", "-3.12"], ["py", "-3.11"]]
    for candidate in candidates:
        try:
            test = subprocess.run(candidate + ["-c", "import sys; print(sys.executable); sys.exit(0 if sys.version_info[:2] in [(3,11),(3,12)] else 1)"],
                                  capture_output=True, text=True, env=env)
            if test.returncode==0:
                return test.stdout.strip()
        except OSError:
            pass
    raise RuntimeError("Python 3.11/3.12 is required. Install it or set REPRO_PYTHON to its python.exe.")


def main():
    args = sys.argv[1:]
    mode = args.pop(0) if args and args[0] in ("smoke","quick","full") else "quick"
    (ROOT/"reproduction_cache").mkdir(exist_ok=True)
    if not PYTHON.exists():
        run([base_python(), "-m", "venv", ENV])
    req = HERE/"requirements-lock.txt"
    # Check exact direct dependency versions, not just import availability.
    check = "import importlib.metadata as m,sys; from pathlib import Path; lines=Path(sys.argv[1]).read_text(encoding='utf-8-sig').splitlines(); assert all(m.version(a)==b for a,b in (s.split('==') for s in lines if s and not s.startswith('#')))"
    result = subprocess.run([str(PYTHON),"-c",check,str(req)],capture_output=True,env=env)
    if result.returncode:
        index = os.environ.get("REPRO_PIP_INDEX_URL","https://pypi.tuna.tsinghua.edu.cn/simple")
        run([PYTHON,"-m","pip","install","--disable-pip-version-check","--index-url",index,"-r",req])
    run([PYTHON,"-m","pip","check"])
    run([PYTHON,"-u",HERE/"run.py","--root",ROOT,"--mode",mode,*args])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Partial data remain in reproduction_results.")
        sys.exit(130)
    except Exception as exc:
        print("FAILED:",exc,flush=True)
        sys.exit(1)
