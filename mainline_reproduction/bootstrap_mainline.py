"""Reuse the existing isolated environment; initialize it if it was removed."""
from pathlib import Path
import importlib.util
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('environment_bootstrap',ROOT/'reproduction/bootstrap.py')
setup=importlib.util.module_from_spec(spec);spec.loader.exec_module(setup)


def main():
    if not setup.PYTHON.exists():
        setup.run([setup.base_python(),'-m','venv',setup.ENV])
        setup.run([setup.PYTHON,'-m','pip','install','--index-url',setup.env.get('REPRO_PIP_INDEX_URL','https://pypi.tuna.tsinghua.edu.cn/simple'),'-r',ROOT/'reproduction/requirements-lock.txt'])
    setup.run([setup.PYTHON,'-m','pip','check'])
    args=sys.argv[1:]
    if args and args[0]=='full':args=['--episodes','1200','--particles','30','--fit-iterations','100',*args[1:]]
    setup.run([setup.PYTHON,'-u',HERE/'run_mainline.py','--root',ROOT,*args])


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('Interrupted. Partial results are preserved.');sys.exit(130)
    except Exception as error:print('FAILED:',error);sys.exit(1)
