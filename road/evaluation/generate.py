"""Generate an immutable COCO10K/GenEval image set with explicit provenance."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch

from ..artifacts import digest, object_digest, write_json
from ..infer import Pipeline
from .protocol import numerical_environment, require_selection, validate_prompts

PROFILES = ("release_v2", "reference_code")


def noise_batch(pipe, jobs, profile, protocol, canonical_batch_size, total):
    shape = pipe.backend.latent_shape
    if profile == "reference_code" and protocol == "coco10k":
        # The inspected historical helper ignores base_seed and seeds the CPU
        # generator with the canonical batch INDEX. Runtime batching may differ.
        groups = {}
        rows = []
        for job in jobs:
            index = job["image_index"]
            batch, offset = divmod(index, canonical_batch_size)
            if batch not in groups:
                count = min(canonical_batch_size, total - batch * canonical_batch_size)
                groups[batch] = torch.randn((count, *shape), generator=torch.Generator().manual_seed(batch), dtype=torch.float32)
            rows.append(groups[batch][offset])
        return torch.stack(rows)
    return torch.stack([
        torch.randn(shape, generator=torch.Generator().manual_seed(job["seed"]), dtype=torch.float32)
        for job in jobs
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "base-model", "prompts", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--protocol", choices=["coco10k", "geneval"], required=True)
    parser.add_argument("--profile", choices=PROFILES, default="release_v2")
    parser.add_argument("--reference-batch-size", type=int)
    parser.add_argument("--selection", help="FID selection JSON; required checkpoint must match")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    validate_prompts(rows, args.protocol)
    if args.batch_size < 1 or (args.seed is not None and not 0 <= args.seed < 2**63 - 10000):
        raise ValueError("Batch size must be positive and seed in [0, 2**63 - 10000)")
    legacy_coco = args.profile == "reference_code" and args.protocol == "coco10k"
    if legacy_coco:
        if args.reference_batch_size is None or args.reference_batch_size < 1:
            raise ValueError("reference_code COCO requires the historical --reference-batch-size")
        if args.seed is not None:
            raise ValueError("Historical COCO batch-index seeds ignore base_seed; omit --seed")
    elif args.reference_batch_size is not None:
        raise ValueError("--reference-batch-size only applies to reference_code COCO")
    base_seed = args.seed if args.seed is not None else (42 if args.profile == "reference_code" and args.protocol == "geneval" else 10)
    checkpoint_sha = digest(Path(args.checkpoint) / "road.json")
    if args.selection:
        require_selection(args.selection, checkpoint_sha)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a fresh generation directory; partial runs cannot be scored")
    pipe = Pipeline(args.checkpoint, args.base_model, args.device, args.precision)
    if args.protocol == "geneval" and pipe.meta["family"] != "sd35m":
        raise ValueError("The ROAD GenEval protocol is defined for SD3.5-M")
    output.mkdir(parents=True)
    # The M helper rounds; the inspected XL evaluator and release-v1 truncate.
    quantization = (
        "round" if args.profile == "reference_code" and pipe.meta["family"] == "sd35m"
        else "truncate"
    )
    jobs = []
    for index, row in enumerate(rows):
        for slot in range(4 if args.protocol == "geneval" else 1):
            image_index = len(jobs)
            seed_index = slot if args.profile == "reference_code" and args.protocol == "geneval" else image_index
            jobs.append({
                "file": f"{index:05d}/samples/{slot:04d}.png" if args.protocol == "geneval" else f"{index:05d}.png",
                "caption": row["prompt"] if args.protocol == "geneval" else row["caption"],
                "prompt_index": index,
                "image_index": image_index,
                "seed": image_index // args.reference_batch_size if legacy_coco else base_seed + seed_index,
            })
        if args.protocol == "geneval":
            folder = output / f"{index:05d}"
            folder.mkdir()
            (folder / "metadata.jsonl").write_text(json.dumps(row) + "\n")
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start:start + args.batch_size]
        noise = noise_batch(pipe, batch, args.profile, args.protocol, args.reference_batch_size, len(jobs))
        images = pipe.generate(
            [row["caption"] for row in batch], noise, quantization=quantization
        )
        if len(images) != len(batch):
            raise RuntimeError("Generator returned an incomplete batch")
        for row, image in zip(batch, images):
            destination = output / row["file"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination)
            row["sha256"] = digest(destination)
    write_json(output / "images.json", jobs)
    source_root = Path(__file__).resolve().parents[1]
    source_paths = [source_root / "infer.py", Path(__file__), *sorted((source_root / "models").glob("*.py"))]
    protocol = {
        "schema": 2,
        "profile": args.profile,
        "protocol": args.protocol,
        "paper_equivalence": "not_established",
        "prompt_sha256": digest(args.prompts),
        "checkpoint_sha256": checkpoint_sha,
        "completed_steps": pipe.meta["completed_steps"],
        "family": pipe.meta["family"],
        "base_model_sha256": pipe.meta["identity"]["base"],
        "precision": pipe.precision,
        "vae_dtype": pipe.meta["identity"]["config"]["training"]["vae_dtype"],
        "base_seed": None if legacy_coco else base_seed,
        "noise_policy": "cpu_fp32_batch_index" if legacy_coco else ("cpu_fp32_seed_plus_slot_per_prompt" if args.profile == "reference_code" and args.protocol == "geneval" else "cpu_fp32_seed_plus_global_image_index"),
        "canonical_batch_size": args.reference_batch_size,
        "runtime_batch_size": args.batch_size,
        "numerical_environment": numerical_environment(pipe.device),
        "count": len(jobs),
        "resolution": 1024,
        "pixel_quantization": quantization,
        "images_sha256": digest(output / "images.json"),
        "generator_sources": {str(path.relative_to(source_root)): digest(path) for path in source_paths},
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "numpy", "Pillow")},
    }
    if args.protocol == "geneval":
        protocol["metadata_files"] = {f"{i:05d}/metadata.jsonl": digest(output / f"{i:05d}/metadata.jsonl") for i in range(553)}
    write_json(output / "protocol.json", protocol)
    write_json(output / "_SUCCESS", {"protocol_sha256": digest(output / "protocol.json"), "images_sha256": protocol["images_sha256"], "generation_sha256": object_digest(protocol)})


if __name__ == "__main__":
    main()
