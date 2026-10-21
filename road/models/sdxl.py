"""Native SDXL epsilon prediction, dual-CLIP conditioning, and VAE scaling."""

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


class Encoder:
    def __init__(self, path, device="cpu", dtype=torch.float32):
        self.device = device
        self.tokenizers = [
            CLIPTokenizer.from_pretrained(path, subfolder=s, local_files_only=True)
            for s in ("tokenizer", "tokenizer_2")
        ]
        self.encoders = [
            cls.from_pretrained(
                path, subfolder=s, local_files_only=True, torch_dtype=dtype
            )
            .to(device)
            .eval()
            .requires_grad_(False)
            for cls, s in [
                (CLIPTextModel, "text_encoder"),
                (CLIPTextModelWithProjection, "text_encoder_2"),
            ]
        ]

    @torch.no_grad()
    def encode(self, prompts):
        tokens = []
        masks = []
        for tokenizer, encoder in zip(self.tokenizers, self.encoders):
            inp = tokenizer(
                prompts,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            output = encoder(inp.input_ids.to(self.device), output_hidden_states=True)
            tokens.append(output.hidden_states[-2])
            masks.append(inp.attention_mask.to(self.device).bool())
        return {
            "tokens": torch.cat(tokens, -1),
            "pooled": output.text_embeds,
            "mask": masks[0] | masks[1],
        }


class Backend:
    latent_shape = (4, 128, 128)

    def __init__(
        self,
        path,
        vae_dtype=torch.bfloat16,
        generator_checkpointing=True,
        vae_checkpointing=True,
        generator_path=None,
        scheduler_path=None,
    ):
        self.generator = UNet2DConditionModel.from_pretrained(
            generator_path or path,
            **({} if generator_path else {"subfolder": "unet"}),
            local_files_only=True,
        )
        self.vae = (
            AutoencoderKL.from_pretrained(
                path, subfolder="vae", torch_dtype=vae_dtype, local_files_only=True
            )
            .eval()
            .requires_grad_(False)
        )
        self.scheduler = DDPMScheduler.from_pretrained(
            scheduler_path or path,
            **({} if scheduler_path else {"subfolder": "scheduler"}),
            local_files_only=True
        )
        if self.scheduler.config.prediction_type != "epsilon":
            raise ValueError("SDXL epsilon scheduler required")
        if generator_checkpointing:
            self.generator.enable_gradient_checkpointing()
        if vae_checkpointing:
            self.vae.enable_gradient_checkpointing()
        self.vae_dtype = vae_dtype

    def predict(self, noise, condition):
        times = torch.full((len(noise),), 999, device=noise.device, dtype=torch.long)
        time_ids = torch.tensor(
            [1024, 1024, 0, 0, 1024, 1024],
            device=noise.device,
            dtype=condition["tokens"].dtype,
        ).expand(len(noise), -1)
        output = self.generator(
            noise,
            times,
            encoder_hidden_states=condition["tokens"],
            added_cond_kwargs={
                "text_embeds": condition["pooled"],
                "time_ids": time_ids,
            },
        ).sample
        alpha = self.scheduler.alphas_cumprod[999].to(noise.device)
        return (noise - (1 - alpha).sqrt() * output) / alpha.sqrt()

    def decode(self, latent):
        with torch.autocast(latent.device.type, enabled=False):
            return self.vae.decode(
                latent.to(self.vae_dtype) / self.vae.config.scaling_factor,
                return_dict=False,
            )[0]
