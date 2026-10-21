"""Selected research adapters with explicit release/reference score conventions."""

import gc
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)

CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


class ScalarMetricAccumulator:
    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.sum_sq = 0.0

    def update(self, scores):
        if isinstance(scores, torch.Tensor):
            values = scores.detach().float().cpu().numpy()
        else:
            values = np.asarray(scores, dtype=np.float64)
        values = values.reshape(-1).astype(np.float64)
        if values.size == 0:
            return
        if not np.isfinite(values).all():
            bad = values[~np.isfinite(values)]
            raise RuntimeError(
                f"Preference metric produced {len(bad)} non-finite score(s); "
                f"first values={bad[:8].tolist()}."
            )
        self.n += int(values.size)
        self.sum += float(values.sum())
        self.sum_sq += float(np.square(values).sum())

    def compute(self):
        if self.n == 0:
            raise RuntimeError("Cannot compute an empty scalar metric.")
        mean = self.sum / self.n
        variance = max(0.0, self.sum_sq / self.n - mean * mean)
        if not np.isfinite(mean) or not np.isfinite(variance):
            raise RuntimeError(
                "Preference metric aggregation produced a non-finite result."
            )
        return {
            "mean": float(mean),
            "std": float(np.sqrt(variance)),
            "count": int(self.n),
            "sum": float(self.sum),
        }


def iter_image_prompt_batches(images, captions: Sequence[str], batch_size: int):
    if len(images) != len(captions):
        raise ValueError(
            f"Image/caption count mismatch: {len(images)} images vs {len(captions)} captions."
        )
    if batch_size < 1:
        raise ValueError("Metric batch size must be >= 1.")
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        yield np.asarray(images[start:end], dtype=np.uint8), list(captions[start:end])


def iter_metric_batches(
    images, captions: Sequence[str], batch_size: int, description: str
):
    """Yield the canonical SDXL batches while reporting processed images."""

    interactive = bool(getattr(sys.stderr, "isatty", lambda: False)())
    progress = tqdm(
        total=len(images),
        desc=description,
        unit="img",
        dynamic_ncols=True,
        file=sys.stderr,
        mininterval=1.0 if interactive else 30.0,
        disable=False,
    )
    try:
        for batch_images, batch_prompts in iter_image_prompt_batches(
            images, captions, batch_size
        ):
            yield batch_images, batch_prompts
            progress.update(len(batch_images))
    finally:
        progress.close()


def images_to_pil(images_np) -> List[Image.Image]:
    return [
        Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
        for image in images_np
    ]


def _pil_bicubic():
    return getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)


def _resize_shortest_edge_pil(image: Image.Image, target_size: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image.size}")
    scale = float(target_size) / float(min(width, height))
    resized_width = max(target_size, int(round(width * scale)))
    resized_height = max(target_size, int(round(height * scale)))
    return image.resize((resized_width, resized_height), _pil_bicubic())


def _center_crop_pil(image: Image.Image, crop_size: int) -> Image.Image:
    width, height = image.size
    left = max(0, int(round((width - crop_size) / 2.0)))
    top = max(0, int(round((height - crop_size) / 2.0)))
    return image.crop((left, top, left + crop_size, top + crop_size))


def preprocess_clip_batch_pil(pil_images, image_size=224):
    """Pure PIL/NumPy/Torch approximation of standard CLIP preprocessing.

    This intentionally avoids torchvision.transforms. HPS/ImageReward results
    should be sanity-checked on a tiny local sample against the official
    preprocessing when the official path is available in a separate environment.
    """
    tensors = []
    mean = torch.tensor(CLIP_IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(CLIP_IMAGE_STD, dtype=torch.float32).view(3, 1, 1)
    for image in pil_images:
        image = image.convert("RGB")
        image = _resize_shortest_edge_pil(image, int(image_size))
        image = _center_crop_pil(image, int(image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        tensor = (tensor - mean) / std
        tensors.append(tensor)
    if not tensors:
        raise ValueError("Cannot preprocess an empty image batch.")
    return torch.stack(tensors, dim=0)


def _image_size_from_value(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return int(value[0])
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_vision_image_size(model, default=224):
    candidates = [
        getattr(model, "image_size", None),
        getattr(model, "input_resolution", None),
    ]
    visual = getattr(model, "visual", None)
    if visual is not None:
        candidates.extend(
            [
                getattr(visual, "image_size", None),
                getattr(visual, "input_resolution", None),
            ]
        )
    blip = getattr(model, "blip", None)
    visual_encoder = (
        getattr(blip, "visual_encoder", None)
        if blip is not None
        else getattr(model, "visual_encoder", None)
    )
    if visual_encoder is not None:
        patch_embed = getattr(visual_encoder, "patch_embed", None)
        candidates.extend(
            [
                getattr(visual_encoder, "image_size", None),
                getattr(visual_encoder, "img_size", None),
                getattr(patch_embed, "img_size", None)
                if patch_embed is not None
                else None,
            ]
        )
    config = getattr(model, "config", None)
    vision_config = (
        getattr(config, "vision_config", None) if config is not None else None
    )
    candidates.append(
        getattr(vision_config, "image_size", None)
        if vision_config is not None
        else None
    )
    for value in candidates:
        image_size = _image_size_from_value(value)
        if image_size is not None and image_size > 0:
            return image_size
    return int(default)


def metric_torch_dtype(dtype_name: str, device: torch.device):
    if device.type != "cuda":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown preference metric dtype: {dtype_name}")


def module_floating_dtype(model):
    if model is None:
        return None
    for tensor in model.parameters(recurse=True):
        if tensor.is_floating_point():
            return tensor.dtype
    for tensor in model.buffers(recurse=True):
        if tensor.is_floating_point():
            return tensor.dtype
    return None


def move_batch_to_device(batch, device, floating_dtype=None):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if floating_dtype is not None and value.is_floating_point():
                out[key] = value.to(device=device, dtype=floating_dtype)
            else:
                out[key] = value.to(device=device)
        else:
            out[key] = value
    return out


def release_metric_memory():
    """Release allocator caches after callers clear their true local references."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_transformers_auto_model(
    model_path: str, processor_path: str, dtype: torch.dtype, device: torch.device
):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
    kwargs = {"local_files_only": True}
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        kwargs["torch_dtype"] = dtype
    model = AutoModel.from_pretrained(model_path, **kwargs).eval().to(device)
    return model, processor


def _processor_inputs(
    processor, images, prompts, device: torch.device, floating_dtype=None
):
    image_inputs = processor(images=images, return_tensors="pt")
    text_inputs = processor(
        text=prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    image_inputs = move_batch_to_device(
        image_inputs, device, floating_dtype=floating_dtype
    )
    text_inputs = move_batch_to_device(text_inputs, device, floating_dtype=None)
    return image_inputs, text_inputs


@torch.inference_mode()
def compute_pickscore(args, images, captions, device: torch.device) -> Dict[str, float]:
    dtype = metric_torch_dtype(args.preference_metric_dtype, device)
    start_time = time.time()
    print(
        "[pref-metric] start pickscore: "
        f"model_path={args.pickscore_model_path}, processor_path={args.pickscore_processor_path}, "
        f"batch_size={args.pickscore_batch_size}"
    )
    model = processor = batches = None
    batch_images = batch_prompts = pil_images = None
    image_inputs = text_inputs = image_embs = text_embs = scores = None
    try:
        model, processor = _load_transformers_auto_model(
            args.pickscore_model_path,
            args.pickscore_processor_path,
            dtype,
            device,
        )
        model_dtype = (
            module_floating_dtype(model) if device.type == "cuda" else torch.float32
        )
        acc = ScalarMetricAccumulator()
        batches = iter_metric_batches(
            images, captions, args.pickscore_batch_size, "PickScore"
        )
        for batch_images, batch_prompts in batches:
            pil_images = images_to_pil(batch_images)
            image_inputs, text_inputs = _processor_inputs(
                processor,
                pil_images,
                batch_prompts,
                device,
                floating_dtype=model_dtype,
            )
            image_embs = model.get_image_features(**image_inputs)
            text_embs = model.get_text_features(**text_inputs)
            image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp() if args.reference_logit_scale else 100.0
            scores = logit_scale * (text_embs * image_embs).sum(dim=-1)
            acc.update(scores)
            del pil_images, image_inputs, text_inputs, image_embs, text_embs, scores
        result = acc.compute()
        elapsed = time.time() - start_time
        print(
            f"[pref-metric] done pickscore: mean={result['mean']:.6f}, count={result['count']}, time={elapsed:.1f}s"
        )
        return {
            "pickscore": result["mean"],
            "pickscore_std": result["std"],
            "pickscore_count": result["count"],
            "pickscore_sum": result["sum"],
            "pickscore_time": float(elapsed),
        }
    finally:
        if batches is not None:
            batches.close()
        batch_images = batch_prompts = pil_images = None
        image_inputs = text_inputs = image_embs = text_embs = scores = None
        batches = processor = model = None
        release_metric_memory()


class AestheticLinearHead(torch.nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(input_dim, 1))

    def forward(self, x):
        return self.layers(x)


def _load_aesthetic_head(path: str, device: torch.device, dtype: torch.dtype):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, torch.nn.Module):
        head = state
    else:
        weight = None
        for key, value in state.items():
            if key.endswith("weight") and getattr(value, "ndim", None) == 2:
                weight = value
                break
        input_dim = int(weight.shape[1]) if weight is not None else 768
        if input_dim != 768:
            raise RuntimeError(
                f"Aesthetic head expected input dimension 768, got {input_dim} from {path}"
            )
        head = AestheticLinearHead(input_dim=input_dim)
        try:
            head.load_state_dict(state, strict=True)
        except RuntimeError:
            remapped = {}
            for key, value in state.items():
                if key in ("weight", "bias"):
                    remapped[f"layers.0.{key}"] = value
                else:
                    remapped[key] = value
            head.load_state_dict(remapped, strict=True)
    return head.eval().to(
        device=device, dtype=dtype if device.type == "cuda" else torch.float32
    )


@torch.inference_mode()
def compute_aesthetic(args, images, captions, device: torch.device) -> Dict[str, float]:
    dtype = metric_torch_dtype(args.preference_metric_dtype, device)
    start_time = time.time()
    print(
        "[pref-metric] start aesthetic: "
        f"clip_path={args.aesthetic_clip_path}, head_path={args.aesthetic_head_path}, batch_size={args.aesthetic_batch_size}"
    )
    model = processor = head = batches = None
    batch_images = pil_images = image_inputs = image_features = scores = None
    try:
        model, processor = _load_transformers_auto_model(
            args.aesthetic_clip_path, args.aesthetic_clip_path, dtype, device
        )
        head = _load_aesthetic_head(args.aesthetic_head_path, device, dtype)
        model_dtype = (
            module_floating_dtype(model) if device.type == "cuda" else torch.float32
        )
        acc = ScalarMetricAccumulator()
        dummy_prompts = [""] * len(images)
        batches = iter_metric_batches(
            images, dummy_prompts, args.aesthetic_batch_size, "Aesthetic Score"
        )
        for batch_images, _ in batches:
            pil_images = images_to_pil(batch_images)
            image_inputs = processor(images=pil_images, return_tensors="pt")
            image_inputs = move_batch_to_device(
                image_inputs, device, floating_dtype=model_dtype
            )
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            scores = head(
                image_features.to(dtype=next(head.parameters()).dtype)
            ).reshape(-1)
            acc.update(scores)
            del pil_images, image_inputs, image_features, scores
        result = acc.compute()
        elapsed = time.time() - start_time
        print(
            f"[pref-metric] done aesthetic: mean={result['mean']:.6f}, count={result['count']}, time={elapsed:.1f}s"
        )
        return {
            "aesthetic_score": result["mean"],
            "aesthetic_score_std": result["std"],
            "aesthetic_score_count": result["count"],
            "aesthetic_score_sum": result["sum"],
            "aesthetic_score_time": float(elapsed),
        }
    finally:
        if batches is not None:
            batches.close()
        batch_images = pil_images = image_inputs = image_features = scores = None
        batches = head = processor = model = None
        release_metric_memory()


def _load_hps_hf_model(
    model_path: str, tokenizer_path: str, device: torch.device, dtype: torch.dtype
):
    from transformers import AutoTokenizer, CLIPModel

    kwargs = {"local_files_only": True}
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        kwargs["torch_dtype"] = dtype
    model = CLIPModel.from_pretrained(model_path, **kwargs).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or model_path, local_files_only=True
    )
    image_size = infer_vision_image_size(model, default=224)
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) if config is not None else None
    context_length = getattr(text_config, "max_position_embeddings", None) or 77
    print(
        "[pref-metric] HPS HF backend: "
        f"model_path={model_path}, tokenizer_path={tokenizer_path or model_path}, "
        f"image_size={image_size}, context_length={context_length}"
    )
    return model, tokenizer, int(image_size), int(context_length)


@torch.inference_mode()
def compute_hps_hf(
    args, images, captions, device: torch.device, metric_name: str, model_path: str
) -> Dict[str, float]:
    dtype = metric_torch_dtype(args.preference_metric_dtype, device)
    start_time = time.time()
    tokenizer_path = getattr(args, "hps_tokenizer_path", None) or model_path
    print(
        f"[pref-metric] start {metric_name}: backend=hf, model={model_path}, batch_size={args.hps_batch_size}"
    )
    model = tokenizer = batches = None
    batch_images = batch_prompts = pil_images = pixel_values = text_inputs = None
    image_features = text_features = scores = None
    try:
        model, tokenizer, image_size, context_length = _load_hps_hf_model(
            model_path, tokenizer_path, device, dtype
        )
        acc = ScalarMetricAccumulator()
        batches = iter_metric_batches(images, captions, args.hps_batch_size, "HPSv2")
        for batch_images, batch_prompts in batches:
            pil_images = images_to_pil(batch_images)
            pixel_values = preprocess_clip_batch_pil(
                pil_images, image_size=image_size
            ).to(device)
            if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
                pixel_values = pixel_values.to(dtype=dtype)
            text_inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=context_length,
                return_tensors="pt",
            )
            text_inputs = move_batch_to_device(text_inputs, device, floating_dtype=None)

            image_features = model.get_image_features(pixel_values=pixel_values)
            text_features = model.get_text_features(**text_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            scale = model.logit_scale.exp() if args.reference_logit_scale else 100.0
            scores = scale * (image_features * text_features).sum(dim=-1)
            acc.update(scores)
            del (
                pil_images,
                pixel_values,
                text_inputs,
                image_features,
                text_features,
                scores,
            )
        result = acc.compute()
        elapsed = time.time() - start_time
        print(
            f"[pref-metric] done {metric_name}: mean={result['mean']:.6f}, count={result['count']}, time={elapsed:.1f}s"
        )
        return {
            metric_name: result["mean"],
            f"{metric_name}_std": result["std"],
            f"{metric_name}_count": result["count"],
            f"{metric_name}_sum": result["sum"],
            f"{metric_name}_time": float(elapsed),
        }
    finally:
        if batches is not None:
            batches.close()
        batch_images = batch_prompts = pil_images = pixel_values = text_inputs = None
        image_features = text_features = scores = None
        batches = tokenizer = model = None
        release_metric_memory()
