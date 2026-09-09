"""Execute unblocked manifest rows sequentially with durable logs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["run_order"]))
    return rows


def select_rows(rows, run_ids, max_runs):
    selected = [
        row
        for row in rows
        if row["implementation_status"] == "ready" and row["command"]
    ]
    if run_ids:
        requested = set(run_ids)
        selected = [row for row in selected if row["run_id"] in requested]
        missing = requested - {row["run_id"] for row in selected}
        if missing:
            raise ValueError(
                "Requested run IDs are absent or blocked: "
                + ", ".join(sorted(missing))
            )
    if max_runs is not None:
        selected = selected[:max_runs]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("experiment_manifest.csv"),
    )
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--max-runs", type=int, default=1)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute; without this flag the runner is a dry run.",
    )
    args = parser.parse_args()
    rows = select_rows(read_rows(args.manifest), args.run_id, args.max_runs)
    if not rows:
        raise RuntimeError("No unblocked manifest rows selected")

    log_root = PROJECT_ROOT / "results" / "run_logs"
    if args.execute:
        log_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        print(f"{row['run_id']}: {row['command']}", flush=True)
        if not args.execute:
            continue

        log_path = log_root / f"{row['run_id']}.log"
        registry_path = log_root / f"{row['run_id']}.json"
        record = {
            "run_id": row["run_id"],
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": row["command"],
            "log_path": str(log_path.resolve()),
        }
        registry_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        with log_path.open("w", encoding="utf-8") as log_handle:
            result = subprocess.run(
                row["command"],
                cwd=PROJECT_ROOT,
                shell=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        record["exit_code"] = result.returncode
        record["status"] = "complete" if result.returncode == 0 else "failed"
        registry_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(
                f"{row['run_id']} failed; inspect {log_path}"
            )


if __name__ == "__main__":
    main()
