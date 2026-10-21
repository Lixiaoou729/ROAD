"""Select lowest FID from protocol-matched 500-step checkpoint evaluations."""

import argparse
import json
from pathlib import Path

from ..artifacts import digest, write_json
from .protocol import select_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    rows = [json.loads(Path(f).read_text()) for f in a.results]
    if any(r.get("schema") != 2 or not r.get("fid_protocol_hash") for r in rows):
        raise ValueError("Selection requires version-2 FID results with complete provenance")
    if any(type(r.get("completed_steps")) is not int or r["completed_steps"] <= 0 or r["completed_steps"] % 500 for r in rows):
        raise ValueError("Paper selection uses 500-step checkpoints")
    if len({r["completed_steps"] for r in rows}) != len(rows):
        raise ValueError("Duplicate checkpoint steps")
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {
        "schema": 2,
        "selected": select_checkpoint(rows),
        "selection_table": sorted(rows, key=lambda row: row["completed_steps"]),
        "input_result_hashes": [digest(path) for path in a.results],
        "coverage": "provided_candidates_only",
    })


if __name__ == "__main__":
    main()
