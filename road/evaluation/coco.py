"""Score a completed COCO10K generation with explicit local metric assets."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from ..artifacts import asset_fingerprint, digest, object_digest, write_json
from .protocol import (
    crop_resize, load_generation, numerical_environment, require_selection, validate_prompts,
)


class ImageStore:
    def __init__(self, root, rows):
        self.root = Path(root)
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return np.stack([self[i] for i in range(*index.indices(len(self)))])
        with Image.open(self.root / self.rows[index]["file"]) as image:
            return np.asarray(image.convert("RGB")).copy()


@torch.inference_mode()
def clip_score(images, captions, path, device, batch):
    from transformers import AutoTokenizer, CLIPModel

    model = CLIPModel.from_pretrained(path, local_files_only=True).to(device).eval()
    if model.config.vision_config.image_size != 256:
        raise ValueError("Paper CLIP scorer requires the 256px ViT-g/14 asset")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(
        1, 3, 1, 1
    )
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(
        1, 3, 1, 1
    )
    scores = []
    for start in range(0, len(images), batch):
        pixels = (
            torch.stack(
                [
                    torch.from_numpy(
                        np.asarray(
                            crop_resize(
                                Image.fromarray(x), 256, Image.Resampling.BICUBIC
                            )
                        ).copy()
                    ).permute(2, 0, 1)
                    for x in images[start : start + batch]
                ]
            )
            .to(device)
            .float()
            / 255
        )
        text = tokenizer(
            ["A photo depicts " + s for s in captions[start : start + batch]],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        a = model.get_image_features(pixel_values=(pixels - mean) / std)
        b = model.get_text_features(**text)
        scores.extend(
            (
                torch.nn.functional.normalize(a.float(), dim=-1)
                * torch.nn.functional.normalize(b.float(), dim=-1)
            )
            .sum(-1)
            .cpu()
            .tolist()
        )
    return float(np.mean(scores))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["images", "prompts", "assets", "output"]:
        p.add_argument("--" + name, required=True)
    p.add_argument(
        "--metrics", default="fid,clip,hpsv20,pickscore,aesthetic,image_reward"
    )
    p.add_argument("--fid-cache", help="Optional content-addressed reference statistics directory")
    p.add_argument("--selection", help="Require this FID-selected checkpoint AND identical COCO image set")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    root = Path(a.images)
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    protocol, rows, generation_sha = load_generation(root, "coco10k")
    if a.selection:
        require_selection(a.selection, protocol["checkpoint_sha256"], generation_sha)
    prompts = [
        json.loads(l) for l in Path(a.prompts).read_text().splitlines() if l.strip()
    ]
    validate_prompts(prompts, "coco10k")
    if (
        protocol["protocol"] != "coco10k"
        or protocol["prompt_sha256"] != digest(a.prompts)
        or len(rows) != 10000
    ):
        raise ValueError("COCO protocol mismatch")
    if any(
        r["prompt_index"] != i
        or r["caption"] != prompts[i]["caption"]
        for i, r in enumerate(rows)
    ):
        raise ValueError("Image order/content mismatch")
    assets = json.loads(Path(a.assets).read_text())
    requested = [name.strip() for name in a.metrics.split(",")]
    allowed = {"fid", "clip", "hpsv20", "pickscore", "aesthetic", "image_reward"}
    if (
        not requested
        or set(requested) - allowed
        or len(set(requested)) != len(requested)
    ):
        raise ValueError("Invalid metric selection")
    if a.batch_size < 1:
        raise ValueError("batch-size must be positive")
    images = ImageStore(root, rows)
    captions = [r["caption"] for r in rows]
    device = torch.device(a.device)
    results = {}
    identities = {}
    # Only hash paths needed by selected metrics; a FID-only selection run does
    # not need every preference-model asset.
    required = {
        "fid": ["inception"],
        "clip": ["clip"],
        "hpsv20": ["hpsv20"],
        "pickscore": ["pickscore", "pickscore_processor"],
        "aesthetic": ["aesthetic_clip", "aesthetic_head"],
        "image_reward": ["image_reward", "image_reward_med_config"],
    }
    for key in {k for m in requested for k in required[m]}:
        path = Path(assets[key])
        identities[key] = asset_fingerprint(path) if path.is_dir() else digest(path)
    fid_definition = None
    if "fid" in requested:
        from .fid import score
        refbase = Path(a.prompts).resolve().parent
        results["fid"], fid_definition = score(
            [root / row["file"] for row in rows],
            [refbase / row["reference"] for row in prompts],
            assets["inception"], device, a.batch_size, a.fid_cache,
        )
    if "clip" in requested:
        results["clip"] = clip_score(
            images, captions, assets["clip"], device, a.batch_size
        )
    from .preference import compute_aesthetic, compute_hps_hf, compute_pickscore

    common = {
        "preference_metric_dtype": "fp32",
        "reference_logit_scale": protocol["profile"] == "reference_code",
    }
    if "hpsv20" in requested:
        args = SimpleNamespace(
            **common, hps_tokenizer_path=assets["hpsv20"], hps_batch_size=a.batch_size
        )
        results.update(
            compute_hps_hf(args, images, captions, device, "hpsv20", assets["hpsv20"])
        )
    if "pickscore" in requested:
        args = SimpleNamespace(
            **common,
            pickscore_model_path=assets["pickscore"],
            pickscore_processor_path=assets["pickscore_processor"],
            pickscore_batch_size=a.batch_size,
        )
        results.update(compute_pickscore(args, images, captions, device))
    if "aesthetic" in requested:
        args = SimpleNamespace(
            **common,
            aesthetic_clip_path=assets["aesthetic_clip"],
            aesthetic_head_path=assets["aesthetic_head"],
            aesthetic_batch_size=a.batch_size,
        )
        results.update(compute_aesthetic(args, images, captions, device))
    if "image_reward" in requested:
        from .image_reward import import_image_reward

        RM = import_image_reward()

        model = RM.load(
            assets["image_reward"],
            device=device,
            med_config=assets["image_reward_med_config"],
        )
        with torch.inference_mode():
            scores = [
                float(model.score(r["caption"], str(root / r["file"]))) for r in rows
            ]
        results["image_reward"] = float(np.mean(scores))
    if not all(np.isfinite(v) for v in results.values()):
        raise ValueError("Nonfinite metric")
    # Selection identity excludes checkpoint/image content but includes
    # data, seeds, scorer assets and implementation versions.
    import importlib.metadata

    definition = {
        k: v
        for k, v in protocol.items()
        if k not in ("checkpoint_sha256", "completed_steps", "images_sha256")
    }
    generation_definition = dict(definition)
    definition.update(
        profile=protocol["profile"],
        preference_scale="learned_exp_logit_scale" if protocol["profile"] == "reference_code" else "cosine_times_100",
        asset_provenance=assets.get("provenance", {}),
        metric_batch_size=a.batch_size,
        metric_environment=numerical_environment(device),
        fid_definition=fid_definition,
        protocol_source=digest(Path(__file__).with_name("protocol.py")),
        asset_hashes=identities,
        metrics=requested,
        scorer_source=digest(__file__),
        preference_source=digest(Path(__file__).with_name("preference.py")),
        image_reward_adapter=digest(Path(__file__).with_name("image_reward.py")),
        versions={
            k: importlib.metadata.version(k)
            for k in ("torch", "transformers", "diffusers", "Pillow", "numpy")
            + (("clean-fid", "scipy") if "fid" in requested else ())
            + (("image-reward",) if "image_reward" in requested else ())
        },
    )
    result = {
        "schema": 2,
        "generation_sha256": generation_sha,
        "fid_protocol_hash": object_digest({"generation": generation_definition, "fid": fid_definition}) if fid_definition is not None else None,
        **results,
        "completed_steps": protocol["completed_steps"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "protocol_hash": hashlib.sha256(
            json.dumps(definition, sort_keys=True).encode()
        ).hexdigest(),
        "definition": definition,
        "historical_result_reproduced": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)


if __name__ == "__main__":
    main()
