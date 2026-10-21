"""Versioned release protocol, distinct from an unverified historical run."""

from pathlib import Path

from PIL import Image

TAGS = ("single_object", "two_object", "counting", "colors", "position", "color_attr")


def crop_resize(image, size, resample=Image.Resampling.LANCZOS):
    image = image.convert("RGB")
    w, h = image.size
    edge = min(w, h)
    # Match the research PIL helper, including odd-edge round-to-even cases.
    x = max(int(round((w - edge) / 2.0)), 0)
    y = max(int(round((h - edge) / 2.0)), 0)
    return image.crop((x, y, x + edge, y + edge)).resize((size, size), resample)


def validate_prompts(rows, protocol):
    if protocol not in ("coco10k", "geneval"):
        raise ValueError("Unknown evaluation protocol")
    expected = 10000 if protocol == "coco10k" else 553
    if len(rows) != expected:
        raise ValueError(f"{protocol} requires {expected} prompts, got {len(rows)}")
    if protocol == "geneval":
        if set(r["tag"] for r in rows) != set(TAGS):
            raise ValueError("GenEval needs all six official tasks")
        if any("include" not in r or "prompt" not in r for r in rows):
            raise ValueError("Use complete official GenEval metadata")
    else:
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate COCO prompt IDs")
        if any("caption" not in r or "reference" not in r for r in rows):
            raise ValueError("COCO rows require caption and paired reference")


def validate_images(root, manifest):
    root = Path(root)
    if len({r["file"] for r in manifest}) != len(manifest):
        raise ValueError("Duplicate image paths")
    for row in manifest:
        path = (root / row["file"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Image escapes generation directory")
        with Image.open(path) as im:
            if im.size != (1024, 1024) or im.mode != "RGB":
                raise ValueError(f"Expected original 1024 RGB output: {path.name}")


def select_checkpoint(results):
    if not results:
        raise ValueError("No checkpoint results")
    identities = {r.get("fid_protocol_hash", r.get("protocol_hash")) for r in results}
    if len(identities) != 1 or not all(identities):
        raise ValueError("Cannot select across different evaluation protocols")
    import math

    if any(not math.isfinite(r["fid"]) for r in results):
        raise ValueError("Nonfinite FID")
    return min(results, key=lambda r: (r["fid"], r["completed_steps"]))


def require_selection(path, checkpoint_sha256, generation_sha256=None):
    """Prevent metrics for a selected row from silently using another run."""
    import json
    selected = json.loads(Path(path).read_text())["selected"]
    if selected["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Evaluation checkpoint differs from FID selection")
    if generation_sha256 is not None and selected.get("generation_sha256") != generation_sha256:
        raise ValueError("COCO metrics must use the identical FID-selected generated images")
    return selected


def load_generation(root, expected_protocol):
    import json
    from ..artifacts import digest, object_digest

    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    complete = json.loads((root / "_SUCCESS").read_text())
    if protocol.get("schema") != 2 or protocol.get("profile") not in ("release_v2", "reference_code"):
        raise ValueError("Use version-2 generation metadata; older outputs are not silently relabeled")
    if protocol.get("protocol") != expected_protocol:
        raise ValueError("Wrong generation protocol")
    if complete.get("protocol_sha256") != digest(root / "protocol.json") or complete.get("images_sha256") != digest(root / "images.json"):
        raise ValueError("Generation metadata changed after completion")
    if protocol.get("images_sha256") != complete.get("images_sha256"):
        raise ValueError("Generation image manifest identity mismatch")
    generation_sha = object_digest(protocol)
    if complete.get("generation_sha256") != generation_sha:
        raise ValueError("Generation identity mismatch")
    rows = json.loads((root / "images.json").read_text())
    expected = 10000 if expected_protocol == "coco10k" else 2212
    if len(rows) != expected or protocol.get("count") != expected:
        raise ValueError("Incomplete generation count")
    validate_images(root, rows)
    if any(digest(root / row["file"]) != row["sha256"] for row in rows):
        raise ValueError("Generated image bytes changed")
    return protocol, rows, generation_sha


def numerical_environment(device):
    """Record arithmetic settings without changing the caller's backend policy."""
    import torch

    return {
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }
