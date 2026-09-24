#!/usr/bin/env bash
# package_e5_e6_handoff.sh
#
# Bundle the exact trees needed to hand off experiments E5 and E6 to another
# machine, without the venv / caches / unrelated experiment tag dirs.
#
# Usage:
#   bash package_e5_e6_handoff.sh
#
# Optional env vars:
#   E4_DIR               override the E4 tag directory
#   ALLOW_INCOMPLETE_E4  =1 -> warn instead of exiting when no completed E4 run
#   DRY_RUN              =1 -> print the file list and key hashes, do not tar
#
# Output:
#   E5_E6_handoff_<YYYYmmdd_HHMMSS>.tar.gz   (relative paths preserved under ROOT)
#   handoff_manifest_sha256.txt              (sha256 of key pinned files)
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
TARBALL="$ROOT/E5_E6_handoff_${STAMP}.tar.gz"
LIST_FILE="$ROOT/.handoff_filelist_${STAMP}.txt"
MANIFEST="$ROOT/handoff_manifest_sha256.txt"

E4TAG="${E4_DIR:-$ROOT/revision_results/experiment_4_gpu_parallel_2500_5seeds_90s}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
info() { printf '%s\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. Pick the newest E4 timestamp dir whose status.json == "completed".
# ---------------------------------------------------------------------------
pick_completed_e4() {
  local tag="$1"
  [[ -d "$tag" ]] || return 1
  python3 - "$tag" <<'PY'
import json, sys
from pathlib import Path
tag = Path(sys.argv[1])
best = None
for child in sorted(tag.iterdir()):
    if not child.is_dir():
        continue
    status = child / 'status.json'
    if not status.is_file():
        continue
    try:
        data = json.loads(status.read_text(encoding='utf-8-sig'))
    except Exception:
        continue
    if data.get('status') == 'completed':
        best = child  # sorted() -> last completed wins (newest timestamp name)
if best is None:
    sys.exit(1)
print(best)
PY
}

E4_RUN=""
if E4_RUN="$(pick_completed_e4 "$E4TAG")"; then
  info "E4 completed run: $E4_RUN"
else
  if [[ "${ALLOW_INCOMPLETE_E4:-0}" == "1" ]]; then
    warn "No completed E4 run under: $E4TAG"
    warn "ALLOW_INCOMPLETE_E4=1 set -> packaging WITHOUT a completed E4 tree."
  else
    die "No E4 run with status.json == \"completed\" under: $E4TAG
     E5 must not start until E4 finishes (LATEST_COMPLETED.txt gate + launch.lock).
     Set ALLOW_INCOMPLETE_E4=1 only for a non-final dry layout."
  fi
fi

# ---------------------------------------------------------------------------
# 2. Resolve pinned upstream paths from the configs / manifest (no guessing).
# ---------------------------------------------------------------------------
E1_RUN_REL="$(python3 - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
cfg = json.loads((root / 'revision_gpu/configs/experiment_5.json').read_text(encoding='utf-8-sig'))
run = cfg.get('upstream_results', {}).get('1')
if not run:
    sys.exit('experiment_5.json has no upstream_results["1"]')
print(run)
PY
)" || die "Cannot read E1 pinned run from experiment_5.json"

E2_MANIFEST_REL="revision_gpu/configs/e2_completed_90s_manifest.json"
E2_RUN_REL="$(python3 - "$ROOT/$E2_MANIFEST_REL" <<'PY'
import json, sys
from pathlib import Path
manifest = Path(sys.argv[1])
data = json.loads(manifest.read_text(encoding='utf-8-sig'))
run = data.get('source_run')
if not run:
    sys.exit('manifest has no source_run')
print(run)
PY
)" || die "Cannot read E2 source_run from $E2_MANIFEST_REL"

E3_TAG_REL="revision_results/experiment_3_gpu_parallel_2500_5seeds_90s"
E3_RUN_REL="$E3_TAG_REL/20260922_185259_914157"
E5_TAG_REL="revision_results/experiment_5_gpu_parallel_2500_5seeds_90s"

info "E1 pinned run : $E1_RUN_REL"
info "E2 run dir    : $E2_RUN_REL"
info "E3 run dir    : $E3_RUN_REL"

# ---------------------------------------------------------------------------
# 3. Build the staging file list (paths relative to ROOT, original layout).
# ---------------------------------------------------------------------------
: > "$LIST_FILE"
add() {
  local rel="$1"
  [[ -e "$ROOT/$rel" ]] || { warn "missing (skipped): $rel"; return 0; }
  printf '%s\n' "$rel" >> "$LIST_FILE"
}

# (a) source trees
add "revision_gpu"
add "revision_experiments"

# (b) E1 pinned run dir (whole dir)
add "$E1_RUN_REL"

# (c) E2 run dir + frozen manifest
add "$E2_RUN_REL"
add "$E2_MANIFEST_REL"

# (d) E3 tag dir + LATEST pointers
add "$E3_RUN_REL"
add "$E3_TAG_REL/LATEST_STARTED.txt"
add "$E3_TAG_REL/LATEST_COMPLETED.txt"

# (e) E4 tag dir + LATEST pointers (only present once E4 completed)
if [[ -n "$E4_RUN" ]]; then
  add "${E4_RUN#"$ROOT"/}"
fi
add "$E4TAG/LATEST_STARTED.txt"
add "$E4TAG/LATEST_COMPLETED.txt"

# (g) E5 tag dir (only needed by E6; optional here)
add "$E5_TAG_REL"
add "$E5_TAG_REL/LATEST_STARTED.txt"
add "$E5_TAG_REL/LATEST_COMPLETED.txt"

# this document
add "HANDOFF_E5_E6.md"

# de-duplicate while preserving order
sort -u "$LIST_FILE" -o "$LIST_FILE"

info ""
info "Files/dirs to package:"
sed 's/^/  /' "$LIST_FILE"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  info ""
  info "DRY_RUN=1 -> not creating tarball."
else
  # -------------------------------------------------------------------------
  # 4. Create the tarball, excluding caches/venv/pyc, preserving layout.
  # -------------------------------------------------------------------------
  tar \
    --create \
    --gzip \
    --file "$TARBALL" \
    --directory "$ROOT" \
    --files-from "$LIST_FILE" \
    --exclude='.venv-revision-gpu' \
    --exclude='revision_cache' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo'
  info ""
  info "Tarball    : $TARBALL"
  info "Size       : $(du -h "$TARBALL" | cut -f1) ($(stat -c '%s' "$TARBALL") bytes)"
  info "sha256     : $(sha256sum "$TARBALL" | cut -d' ' -f1)"
fi

rm -f "$LIST_FILE"

# ---------------------------------------------------------------------------
# 5. Emit a manifest of key pinned file hashes for the collaborator.
# ---------------------------------------------------------------------------
: > "$MANIFEST"
emit_hash() {
  local label="$1" rel="$2"
  if [[ -f "$ROOT/$rel" ]]; then
    printf '%s  %s  %s\n' "$(sha256sum "$ROOT/$rel" | cut -d' ' -f1)" "$label" "$rel" >> "$MANIFEST"
  else
    printf '%-12s  %s  %s\n' "MISSING" "$label" "$rel" >> "$MANIFEST"
  fi
}
emit_hash "E1_handoff"        "$E1_RUN_REL/handoff.json"
emit_hash "E2_manifest"       "$E2_MANIFEST_REL"
emit_hash "E3_model_registry" "$E3_RUN_REL/model_registry.json"
emit_hash "E3_LATEST_COMPLETED" "$E3_TAG_REL/LATEST_COMPLETED.txt"

info ""
info "Wrote $MANIFEST:"
sed 's/^/  /' "$MANIFEST"
info ""
info "Done."
