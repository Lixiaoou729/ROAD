"""Frozen DINOv3-B/16 selected taps, with gradients retained to RGB inputs."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class DinoFeatures(nn.Module):
    def __init__(self, path, checkpoint_features=True):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            path, local_files_only=True, attn_implementation="eager"
        )
        config = self.model.config
        if (
            config.hidden_size != 768
            or config.num_hidden_layers != 12
            or config.patch_size != 16
        ):
            raise ValueError(
                "The release requires DINOv3 ViT-B/16 (768 dim, 12 blocks)"
            )
        self.registers = int(config.num_register_tokens)
        if self.registers != 4:
            raise ValueError("Expected four DINOv3 registers")
        self.theta = float(getattr(config, "rope_theta", 100.0))
        self.checkpoint_features = checkpoint_features
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.model.requires_grad_(False).eval()
        self._features = {}
        self._handles = []
        # Transformers 4.57 DINOv3 exposes layer; permit encoder.layer too.
        encoder = getattr(self.model, "encoder", self.model)
        blocks = getattr(encoder, "layer", getattr(encoder, "layers", None))
        if blocks is None:
            raise RuntimeError("Could not locate pinned DINOv3 blocks")
        for key, index in enumerate((2, 5, 8, 11)):

            def capture(_module, _args, output, key=key):
                self._features[key] = output[0] if isinstance(output, tuple) else output

            self._handles.append(blocks[index].register_forward_hook(capture))

    def train(self, mode=True):
        super().train(False)
        return self

    def _extract(self, x):
        self._features.clear()
        try:
            self.model(pixel_values=x, output_hidden_states=False)
            return tuple(
                torch.cat(
                    [
                        self._features[k][:, :1],
                        self._features[k][:, 1 + self.registers :],
                    ],
                    1,
                )
                for k in range(4)
            )
        finally:
            self._features.clear()

    def forward(self, images):
        # Preserve normalization-before-bicubic-antialiased resize.
        x = ((images + 1) * 0.5 - self.mean) / self.std
        x = F.interpolate(
            x, (224, 224), mode="bicubic", align_corners=False, antialias=True
        )
        if self.checkpoint_features and x.requires_grad:
            return checkpoint(
                self._extract, x, use_reentrant=False, preserve_rng_state=True
            )
        return self._extract(x)
