"""Locate the unchanged experiment 1/2 package; no experiment runs on import."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=ROOT/'revision_experiments'
if not BASE.is_dir():
    raise FileNotFoundError('Keep revision_extensions beside the existing revision_experiments directory.')
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))
