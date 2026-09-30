#!/usr/bin/env bash
# Read-only status snapshot for the chained E5(resume) -> E6 run.
# Safe to run repeatedly; writes nothing except stdout.
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
E5_BASE="$ROOT/revision_results/experiment_5_gpu_parallel_2500_5seeds_90s"
E5_RUN="$E5_BASE/20260925_124626_463187"
E6_BASE="$ROOT/revision_results/experiment_6_gpu_parallel_2500_5seeds_90s"
CHAIN_LOG="$ROOT/revision_cache/e5e6_chain.log"

echo "=== e5e6 status $(date -Is) ==="
unit="$(systemctl --user is-active rlcharge-e5e6 2>/dev/null || true)"
printf 'unit      : %s\n' "${unit:-unknown}"
systemctl --user show rlcharge-e5e6 -p ActiveEnterTimestamp -p ExecMainStatus --no-pager 2>/dev/null |
  while read -r line; do printf 'unit-meta : %s\n' "$line"; done

python3 - "$E5_BASE" "$E5_RUN" "$E6_BASE" <<'PY'
import json, sys
from pathlib import Path
base, run, e6base = (Path(a) for a in sys.argv[1:4])

def st(p):
    try:
        return json.loads((p / 'status.json').read_text())['status']
    except Exception:
        return 'n/a'

def prog(p):
    try:
        d = json.loads(Path(p).read_text())
        return f"{d['completed']}/{d['total']} ({len(d.get('active', []))} active)"
    except Exception:
        return 'n/a'

print(f"e5        : {st(run)}")
print(f"e5 phase1 : {prog(run / 'parallel_sensitivity_jobs' / 'progress.json')}")
print(f"e5 phase2 : {prog(run / 'parallel_generalization_jobs' / 'progress.json')}")
lc = base / 'LATEST_COMPLETED.txt'
ls = base / 'LATEST_STARTED.txt'
print(f"e5 gate   : completed-file={'yes' if lc.exists() else 'no'} "
      f"points-here={'yes' if lc.exists() and lc.read_text().strip() == str(run) else 'no'} "
      f"started==completed={'yes' if lc.exists() and ls.exists() and lc.read_text().strip() == ls.read_text().strip() else 'no'}")
if e6base.exists():
    dirs = sorted((d for d in e6base.iterdir() if d.is_dir()), key=lambda d: d.name)
    latest = dirs[-1] if dirs else None
    print(f"e6        : {st(latest) if latest else 'n/a'}  ({latest.name if latest else '-'})")
    if latest:
        print(f"e6 gate   : completed-file={'yes' if (e6base / 'LATEST_COMPLETED.txt').exists() else 'no'}")
else:
    print("e6        : not-started (no result directory)")
PY

echo "gpu       : $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"
for log in "$CHAIN_LOG" "$E5_RUN/run.log"; do
  [ -f "$log" ] || continue
  last="$(grep -v '^[[:space:]]*$' "$log" | tail -1)"
  printf 'last-log  : %s | %s\n' "$(basename "$log")" "$last"
done
