"""Family dispatch, independent of all training/critic imports."""

import torch


def encoder(family, path, device="cpu", dtype=torch.float32):
    if family == "sdxl":
        from .sdxl import Encoder

        return Encoder(path, device, dtype)
    if family != "sd35m":
        raise ValueError(family)
    from .sd35_text_condition import SD35OnlineTextEncoder

    native = SD35OnlineTextEncoder(path, device=device, dtype=dtype)

    class Encoder:
        @torch.no_grad()
        def encode(self, prompts):
            c = native.encode(prompts, t5_max_sequence_length=77)
            return {
                "tokens": c.prompt_embeds,
                "pooled": c.pooled_prompt_embeds,
                "mask": torch.ones(
                    c.prompt_embeds.shape[:2],
                    device=c.prompt_embeds.device,
                    dtype=torch.bool,
                ),
            }

    return Encoder()


def backend(
    family,
    path,
    vae_dtype=torch.bfloat16,
    generator_checkpointing=True,
    vae_checkpointing=True,
    generator_path=None,
    scheduler_path=None,
):
    if family == "sdxl":
        from .sdxl import Backend

        return Backend(
            path, vae_dtype, generator_checkpointing, vae_checkpointing, generator_path, scheduler_path
        )
    if family != "sd35m":
        raise ValueError(family)
    from .sd35_generator_backend import SD35GeneratorBackend
    from .sd35_text_condition import TextCondition

    native = SD35GeneratorBackend.from_pretrained(
        path,
        vae_dtype=vae_dtype,
        gradient_checkpointing=generator_checkpointing,
        vae_gradient_checkpointing=vae_checkpointing,
        transformer_path=generator_path,
        scheduler_path=scheduler_path,
    )

    class Backend:
        vae = native.vae
        latent_shape = native.latent_shape
        generator = native.transformer
        scheduler = native.scheduler

        def predict(self, noise, condition):
            native.transformer = self.generator
            c = TextCondition(condition["tokens"], condition["pooled"])
            return native.one_step(noise, c).pred_clean_latent

        def decode(self, latent):
            return native.decode_model_latent(latent)

    return Backend()
