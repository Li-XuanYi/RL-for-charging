"""Only the CLI main starts work. One separate output directory per invocation."""
import argparse
import contextlib
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault('PYBAMM_DISABLE_TELEMETRY', 'true')
os.environ.setdefault('MPLBACKEND', 'Agg')
from common import dump, inside, sha256


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log
    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        self.log.flush()
    def flush(self):
        self.stream.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=['1', '2'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--experiment1-result', type=Path,
                        help='Completed experiment_1 run folder; otherwise latest completed')
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config or HERE / 'configs' / f'experiment_{args.experiment}.json'
    cfg = json.loads(config_path.read_text(encoding='utf-8-sig'))
    tag = cfg.get('result_tag', '')
    if not isinstance(tag, str) or (tag and (not tag.isascii() or not all(c.isalnum() or c == '_' for c in tag))):
        raise ValueError('result_tag must contain only ASCII letters, digits and underscores')
    if cfg.get('exploratory', False) and not tag:
        raise ValueError('Exploratory runs require a separate result_tag to protect formal results')
    result_name = f'experiment_{args.experiment}' + (f'_{tag}' if tag else '')
    results = inside(root / 'revision_results' / result_name, root)
    results.mkdir(parents=True, exist_ok=True)
    out = results / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir()
    for key in ['MPLCONFIGDIR', 'XDG_CACHE_HOME']:
        os.environ[key] = str(root / 'revision_cache' / key.lower())
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    if args.experiment1_result:
        cfg['experiment1_result'] = str(args.experiment1_result.resolve())
    dump(out / 'config.json', cfg)
    source_files=[p for p in HERE.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.txt','.md']
                  and '__pycache__' not in p.parts]
    for source in source_files:
        destination=out/'source_snapshot'/source.relative_to(HERE)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    dump(out / 'status.json', {'status': 'running', 'experiment': args.experiment})
    (results / 'LATEST_STARTED.txt').write_text(str(out), encoding='utf-8')
    dump(out / 'provenance.json', {
        'utc_started': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'platform': platform.platform(),
        'packages': {p: version(p) for p in ['numpy', 'scipy', 'pybamm', 'casadi', 'torch', 'matplotlib']},
        'config_path': str(config_path.resolve()), 'config_sha256': sha256(config_path),
        'source': [{'path': str(p.relative_to(HERE)), 'sha256': sha256(p)} for p in source_files],
        'validation': 'Authoring was static only; no experiment or Python program was run by the assistant.'})
    started = time.perf_counter()
    with (out / 'run.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print('OUTPUT:', out, flush=True)
            try:
                if args.experiment == '1':
                    from experiment1 import run
                else:
                    from experiment2 import run
                result = run(root, out, cfg)
                if cfg.get('exploratory', False):
                    label = {'exploratory': True, 'formal_evidence': False,
                             'training_steps_per_run': cfg['training_steps'],
                             'training_seeds': cfg['seeds'], 'result_tag': tag,
                             'interpretation': 'Reduced-budget exploratory run; not a substitute for formal convergence or robustness evidence.'}
                    dump(out / 'EXPLORATORY_RUN.json', label)
                    assessment_path = out / 'scientific_assessment.json'
                    if assessment_path.exists():
                        assessment = json.loads(assessment_path.read_text(encoding='utf-8-sig'))
                        assessment.update(label)
                        dump(assessment_path, assessment)
                    report_path = out / 'REPORT.md'
                    if report_path.exists():
                        report_path.write_text('# 探索版结果：不作为正式论文比较结论\n\n'
                            + f'每组 {cfg["training_steps"]} 步；种子 {cfg["seeds"]}；独立目录 {result_name}。\n\n'
                            + report_path.read_text(encoding='utf-8-sig'), encoding='utf-8')
                dump(out / 'status.json', {'status': 'completed', 'experiment': args.experiment,
                    'exploratory': bool(cfg.get('exploratory', False)), 'result_tag': tag,
                    'elapsed_s': time.perf_counter() - started, 'result': result,
                    'meaning': 'Computations completed; inspect scientific assessment, not this status alone.'})
                (results / 'LATEST_COMPLETED.txt').write_text(str(out), encoding='utf-8')
                print('COMPLETED:', out, flush=True)
            except BaseException as exc:
                dump(out / 'status.json', {'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                    'error': str(exc), 'error_type': type(exc).__name__,
                    'elapsed_s': time.perf_counter() - started})
                traceback.print_exc()
                raise


if __name__ == '__main__':
    main()
