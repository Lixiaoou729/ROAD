"""Clean-FID on explicit file lists; optional content-addressed reference stats."""

import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from ..artifacts import digest, object_digest, write_json
from .protocol import crop_resize, numerical_environment


def preprocess_512(array):
    return np.asarray(crop_resize(Image.fromarray(array), 512)).copy()


def score(generated, reference, inception, device, batch_size, cache=None):
    from cleanfid import fid, resize
    from cleanfid.inception_torchscript import InceptionV3W

    inception = Path(inception)
    if inception.name != "inception-2015-12-05.pt":
        raise ValueError("Expected the explicit clean-fid Inception TorchScript asset")
    definition = {
        "reference_images": [digest(path) for path in reference],
        "count": len(reference),
        "inception_sha256": digest(inception),
        "preprocessing": "RGB_round_center_crop_512_Lanczos_then_clean_299_float_bicubic",
        "source_sha256": digest(__file__),
        "crop_source": digest(Path(__file__).with_name("protocol.py")),
        "clean_fid_source": digest(fid.__file__),
        "resize_source": digest(resize.__file__),
        "versions": {name: importlib.metadata.version(name) for name in ("clean-fid", "Pillow", "numpy", "torch", "scipy")},
        "numerical_environment": numerical_environment(device),
        "batch_size": batch_size,
    }
    identity = object_digest(definition)
    model = InceptionV3W(str(inception.parent), download=False, resize_inside=False).to(device).eval()

    def statistics(files):
        features = fid.get_files_features(
            [str(path) for path in files], model=model, device=device,
            batch_size=batch_size, num_workers=0, mode="clean",
            custom_image_tranform=preprocess_512,
        )
        if features.shape != (len(files), 2048) or not np.isfinite(features).all():
            raise ValueError("Invalid Inception features")
        return np.mean(features, axis=0), np.cov(features, rowvar=False)

    cache_path = Path(cache) / (identity + ".npz") if cache else None
    if cache_path is not None and cache_path.exists():
        meta = json.loads(cache_path.with_suffix(".json").read_text())
        if meta.get("identity") != identity or meta.get("definition") != definition or meta.get("sha256") != digest(cache_path):
            raise ValueError("Reference statistics cache is corrupt or incomplete")
        with np.load(cache_path, allow_pickle=False) as arrays:
            real_mean, real_cov = arrays["mean"], arrays["covariance"]
        if real_mean.shape != (2048,) or real_cov.shape != (2048, 2048) or not np.isfinite(real_mean).all() or not np.isfinite(real_cov).all():
            raise ValueError("Invalid reference statistics shape/values")
    else:
        real_mean, real_cov = statistics(reference)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".partial")
            with temporary.open("xb") as stream:
                np.savez(stream, mean=real_mean, covariance=real_cov)
            os.link(temporary, cache_path)
            temporary.unlink()
            write_json(cache_path.with_suffix(".json"), {"identity": identity, "sha256": digest(cache_path), "definition": definition})
    fake_mean, fake_cov = statistics(generated)
    value = float(fid.frechet_distance(fake_mean, fake_cov, real_mean, real_cov))
    if not np.isfinite(value):
        raise ValueError("Nonfinite FID")
    return value, definition
