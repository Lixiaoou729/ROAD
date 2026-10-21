"""Validate ROAD GenEval output, then call a user-installed official scorer."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ..artifacts import digest, write_json
from .protocol import TAGS, load_generation, require_selection


def summarize_results(results, expected_files):
    if len(results) != len(expected_files) or {r["filename"] for r in results} != set(
        expected_files
    ):
        raise ValueError("GenEval results do not cover exactly the generated images")
    if {r["tag"] for r in results} != set(TAGS) or any(
        type(r["correct"]) is not bool for r in results
    ):
        raise ValueError("Invalid GenEval tasks or accuracy values")
    means = {
        tag: sum(r["correct"] for r in results if r["tag"] == tag)
        / sum(r["tag"] == tag for r in results)
        for tag in TAGS
    }
    return {
        "task_accuracy": means,
        "overall": sum(means.values()) / len(TAGS),
        "count": len(results),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["images", "geneval-repo", "detector", "output"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--selection", help="Require the FID-selected checkpoint")
    p.add_argument("--python", default=sys.executable)
    a = p.parse_args()
    root = Path(a.images).resolve()
    out = Path(a.output).resolve()
    repo = Path(a.geneval_repo).resolve()
    protocol, rows, generation_sha = load_generation(root, "geneval")
    if a.selection:
        require_selection(a.selection, protocol["checkpoint_sha256"])
    if out.exists():
        raise FileExistsError(out)
    scorer = repo / "evaluation/evaluate_images.py"
    summary = repo / "evaluation/summary_scores.py"
    if not scorer.is_file() or not summary.is_file():
        raise ValueError("Official GenEval checkout required")
    metadata = protocol.get("metadata_files", {})
    if set(metadata) != {f"{i:05d}/metadata.jsonl" for i in range(553)} or any(
        digest(root / name) != sha for name, sha in metadata.items()
    ):
        raise ValueError("GenEval prompt metadata missing or changed")
    detector = Path(a.detector).resolve()
    if not detector.is_dir():
        raise ValueError("Detector asset directory missing")
    detector_hashes = {
        str(f.relative_to(detector)): digest(f)
        for f in sorted(detector.rglob("*"))
        if f.is_file()
    }
    if not detector_hashes:
        raise ValueError("Detector assets are empty")
    out.mkdir(parents=True)
    subprocess.run(
        [
            a.python,
            str(scorer),
            str(root),
            "--outfile",
            str(out / "results.jsonl"),
            "--model-path",
            str(Path(a.detector).resolve()),
        ],
        cwd=repo,
        check=True,
    )
    scored = [
        json.loads(line)
        for line in (out / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    aggregate = summarize_results(scored, [str(root / r["file"]) for r in rows])
    write_json(out / "metrics.json", aggregate)
    result = subprocess.run(
        [a.python, str(summary), str(out / "results.jsonl")],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (out / "summary.txt").write_text(result.stdout)
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "generation": protocol,
                "generation_sha256": generation_sha,
                "scorer_sha256": digest(scorer),
                "summary_sha256": digest(summary),
                "detector_hashes": detector_hashes,
                "scorer_sources": {
                    str(f.relative_to(repo)): digest(f)
                    for f in sorted((repo / "evaluation").rglob("*.py"))
                },
            },
            indent=2,
        )
    )
    print(result.stdout)


if __name__ == "__main__":
    main()
