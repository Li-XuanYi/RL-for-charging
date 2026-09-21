#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ -n "${REPRO_PYTHON:-}" ]]; then
  BOOTSTRAP_PYTHON="$REPRO_PYTHON"
else
  BOOTSTRAP_PYTHON="python3"
fi
exec "$BOOTSTRAP_PYTHON" -u "$ROOT/revision_gpu/bootstrap_gpu.py" 4 "$@"
