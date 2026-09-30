#!/usr/bin/env bash
# Supervisor: drive E5 (resume) then E6 to completion under the cluster GPU
# idle-reaper, which SIGTERMs GPU processes judged idle for 2 h.
#
# Observed 2026-09-26..28 (/var/log/gpu_kill_history.log): our CPU-plant-bound
# workers look GPU-idle and get killed, and parallel_jobs.py is fail-fast, so a
# single kill aborts the whole run. This script simply re-invokes the official
# launcher with --resume until the run is marked completed; already finished
# jobs are reused, killed jobs restart. It does NOT modify any revision_gpu code
# and does not fake GPU activity.
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

E5_BASE="$ROOT/revision_results/experiment_5_gpu_parallel_2500_5seeds_90s"
E5_RUN="$E5_BASE/20260925_124626_463187"
E6_BASE="$ROOT/revision_results/experiment_6_gpu_parallel_2500_5seeds_90s"
MAX_E5_ATTEMPTS="${MAX_E5_ATTEMPTS:-20}"
MAX_E6_ATTEMPTS="${MAX_E6_ATTEMPTS:-40}"
RETRY_SLEEP_S="${RETRY_SLEEP_S:-120}"
say() { echo "[e5e6-supervisor $(date -Is)] $*"; }

completed_dir() {  # $1 = base dir; prints LATEST_COMPLETED content or empty
  [ -f "$1/LATEST_COMPLETED.txt" ] && tr -d '\r\n' < "$1/LATEST_COMPLETED.txt" || true
}

status_of() {  # $1 = run dir
  python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['status'])" "$1/status.json" 2>/dev/null || echo unknown
}

# ---------------- E5: resume the same interrupted run ----------------
ok=0
for attempt in $(seq 1 "$MAX_E5_ATTEMPTS"); do
  say "E5 attempt ${attempt}/${MAX_E5_ATTEMPTS}: resume $E5_RUN"
  bash run_experiment_5_gpu_parallel_2500_5seeds_90s.sh --resume "$E5_RUN"
  if [ "$(completed_dir "$E5_BASE")" = "$E5_RUN" ]; then ok=1; break; fi
  say "E5 attempt ${attempt} ended with status=$(status_of "$E5_RUN"); sleeping ${RETRY_SLEEP_S}s"
  sleep "$RETRY_SLEEP_S"
done
if [ "$ok" != 1 ]; then
  say "E5 did not complete after ${MAX_E5_ATTEMPTS} attempts; not starting E6."
  exit 1
fi
say "E5 completed. $(status_of "$E5_RUN")"

# ---------------- E6: fresh run, then resume whatever dir it created -------
e6_run=""
for attempt in $(seq 1 "$MAX_E6_ATTEMPTS"); do
  if [ -z "$e6_run" ]; then
    say "E6 attempt ${attempt}/${MAX_E6_ATTEMPTS}: fresh start"
    bash run_experiment_6_gpu_parallel_2500_5seeds_90s.sh
  else
    say "E6 attempt ${attempt}/${MAX_E6_ATTEMPTS}: resume $e6_run"
    bash run_experiment_6_gpu_parallel_2500_5seeds_90s.sh --resume "$e6_run"
  fi
  newest="$(ls -1dt "$E6_BASE"/*/ 2>/dev/null | head -1)"
  newest="${newest%/}"
  if [ -n "$newest" ] && [ "$(completed_dir "$E6_BASE")" = "$newest" ]; then
    say "E6 completed: $newest"
    exit 0
  fi
  [ -n "$newest" ] && e6_run="$newest"
  say "E6 attempt ${attempt} ended (run=${e6_run:-none} status=$( [ -n "$e6_run" ] && status_of "$e6_run" || echo none )); sleeping ${RETRY_SLEEP_S}s"
  sleep "$RETRY_SLEEP_S"
done
say "E6 did not complete after ${MAX_E6_ATTEMPTS} attempts."
exit 1
