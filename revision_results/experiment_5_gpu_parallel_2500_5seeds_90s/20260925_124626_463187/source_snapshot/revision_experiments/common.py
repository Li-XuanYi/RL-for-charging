"""File-only utilities; importing this module never starts an experiment."""
import csv
import hashlib
import json
import math
from pathlib import Path


def clean(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False,
                               allow_nan=False), encoding='utf-8')
    temp.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(clean(v), ensure_ascii=False) if
                             isinstance(v, (dict, list, tuple)) else clean(v)
                             for k, v in row.items()})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inside(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if not path.is_relative_to(parent):
        raise ValueError(f'Path must stay within {parent}: {path}')
    return path
