#!/usr/bin/env bash
# E6 emergency decoupling driver (2026-09-30).
#
# WHY
#   revision_gpu/parallel_jobs.py is fail-fast: when the cluster GPU idle-reaper
#   (gpu_quota, /var/log/gpu_kill_history.log) SIGTERMs one worker, the coordinator's
#   finally-block terminates the other five and run_study marks the run failed. On
#   resume only jobs that already have result.json are reused, so every in-flight
#   N=12 job (~7.8 h) restarts from scratch. Kills were observed at 04:30, 11:21,
#   20:21 and 23:46 (intervals 6.9 h, 9.0 h, 3.4 h), and one 3.4 h attempt finished
#   zero jobs, so the official path cannot converge.
#
# WHAT THIS DOES (and does not do)
#   Drives each pending job as an INDEPENDENT process via the official worker entry
#   point `parallel_jobs.py <task.json>` with that job's coordinator-signed task.json.
#   A kill now costs at most that single job; a finished job immediately persists
#   result.json, which the official coordinator REUSEs on the next --resume (it
#   re-validates the task signature and the checkpoint sha256 itself).
#   No file under the code-fingerprint gate (revision_gpu/*.py,
#   revision_experiments/{control,model}.py, original_parameters.json) is touched,
#   so the E6 lineage gate and the resume source-hash check both still pass.
#
# WHEN ALL PENDING JOBS ARE DONE
#   Hands back to the official launcher with --resume, retrying until status.json
#   reads "completed", so the serial CPU/CUDA timing panel and finalisation run in
#   the normal official code path.
#
# Usage: E6_RUN_DIR=<dir> bash run_e6_decoupled.sh        (defaults to the 2026-09-28 run)
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUN="${E6_RUN_DIR:-$ROOT/revision_results/experiment_6_gpu_parallel_2500_5seeds_90s/20260928_194517_801804}"
JOBS="$RUN/parallel_scaling_jobs"
PY="$ROOT/.venv-revision-gpu/bin/python"
LAUNCHER="$ROOT/run_experiment_6_gpu_parallel_2500_5seeds_90s.sh"
MAX_TRIES="${MAX_TRIES:-40}"
RETRY_SLEEP_S="${RETRY_SLEEP_S:-120}"
say(){ echo "[e6-decoupled $(date -Is)] $*"; }

[ -d "$JOBS" ] || { say "no job dir: $JOBS"; exit 1; }
[ -x "$PY" ]   || { say "no interpreter: $PY"; exit 1; }

# Prefer cards with real free memory. Our footprint is tiny (~0.8 GiB/worker), the
# work itself is CPU-bound, so utilisation is not used as a filter here.
pick_gpus(){
  "$PY" - <<'PY'
import subprocess
rows=[]
out=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
for line in out.splitlines():
    idx,uuid,free,util=[p.strip() for p in line.split(',')]
    rows.append((int(free),uuid))
for floor in (16384, 8192, 2048):
    ok=[u for f,u in sorted(rows,key=lambda r:-r[0]) if f>=floor]
    if ok:
        print(' '.join(ok)); break
PY
}

mapfile -t PENDING < <(for d in "$JOBS"/job_*; do
  [ -f "$d/result.json" ] && continue
  [ -f "$d/task.json" ]   || continue
  echo "$d"
done | sort)

if [ "${#PENDING[@]}" -eq 0 ]; then
  say "nothing pending; every job already has result.json"
else
  mapfile -t GPUS < <(pick_gpus | tr ' ' '\n' | grep -v '^$')
  [ "${#GPUS[@]}" -gt 0 ] || { say "no usable GPU found; aborting"; exit 1; }
  say "pending=${#PENDING[@]} gpus=${#GPUS[@]}  tries/job=$MAX_TRIES"
  for d in "${PENDING[@]}"; do say "  pending $(basename "$d")"; done

  run_one(){  # $1=job dir  $2=gpu uuid  $3=slot
    local d="$1" gpu="$2" slot="$3" n=0 rc=0
    while :; do
      if [ -f "$d/result.json" ]; then say "DONE $(basename "$d")"; return 0; fi
      n=$((n+1))
      if [ "$n" -gt "$MAX_TRIES" ]; then say "GIVE-UP $(basename "$d") after $MAX_TRIES attempts"; return 1; fi
      say "START $(basename "$d") attempt $n gpu=$gpu"
      MPLCONFIGDIR="$ROOT/revision_cache/matplotlib_workers/decoupled_$slot" \
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      CUDA_VISIBLE_DEVICES="$gpu" REVISION_WORKER_GPU="$gpu" \
      PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYBAMM_DISABLE_TELEMETRY=true MPLBACKEND=Agg \
      PIP_CACHE_DIR="$ROOT/revision_cache/pip" XDG_CACHE_HOME="$ROOT/revision_cache/xdg" \
      PYTHONPYCACHEPREFIX="$ROOT/revision_cache/pycache" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        "$PY" -u "$ROOT/revision_gpu/parallel_jobs.py" "$d/task.json" >> "$d/worker.log" 2>&1
      rc=$?
      say "EXIT $(basename "$d") attempt $n rc=$rc"
      [ -f "$d/result.json" ] || sleep "$RETRY_SLEEP_S"
    done
  }

  i=0
  for d in "${PENDING[@]}"; do
    slot=$i
    gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
    run_one "$d" "$gpu" "$slot" &
    i=$((i+1))
  done
  wait
fi

missing=0
for d in "$JOBS"/job_*; do [ -f "$d/result.json" ] || { say "STILL MISSING $(basename "$d")"; missing=$((missing+1)); }; done
if [ "$missing" -ne 0 ]; then say "$missing job(s) without result.json; not handing off"; exit 1; fi
say "all jobs present; handing off to the official launcher with --resume"

for attempt in $(seq 1 "$MAX_TRIES"); do
  say "handoff attempt $attempt: bash $(basename "$LAUNCHER") --resume $RUN"
  bash "$LAUNCHER" --resume "$RUN"
  status="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['status'])" "$RUN/status.json" 2>/dev/null || echo unknown)"
  if [ "$status" = completed ]; then say "E6 COMPLETED: $RUN"; exit 0; fi
  say "handoff attempt $attempt ended with status=$status; sleeping ${RETRY_SLEEP_S}s"
  sleep "$RETRY_SLEEP_S"
done
say "handoff did not reach 'completed'"; exit 1
