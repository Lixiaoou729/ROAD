# Third-party notices

## Retained ProjectedGAN code

`pg_d2o/` was copied from the user-provided sibling
`codex_oom_sdxl_d2o_transformer/` with caches and platform metadata excluded.
It contains the established DINOv3 feature/projector and ProjectedGAN
discriminator implementation. The migration did not rewrite or relicense that
code. Its existing provenance and notices remain controlling.

## Stability AI SD3.5 reference

The local snapshot at commit
`8565799a3b41eb0c7ba976d18375f0f753f56402` is MIT licensed, copyright
Stability AI. Its equations and tensor construction were consulted. No source
file or substantial verbatim snippet was copied.

## Hugging Face Diffusers

Diffusers v0.36.0 at commit
`9380e58821092ad05e618fa8a259030136ba43c0` is Apache-2.0 licensed. Public
model/scheduler/VAE APIs are called directly. Small behavioral adaptations are:

- the SD3 pipeline's CLIP channel concatenation, padding to T5 width, T5
  sequence concatenation, and pooled CLIP concatenation;
- actual FlowMatch scheduler shift/grid construction and Euler sigma→0
  reference stepping;
- the DreamBooth example's flow state and velocity conventions; and
- VAE scale/shift normalization plus the built-in decoder checkpointing flag.

These are re-expressed in project-specific typed helpers; no upstream file is
vendored.

## SenseFlow, SwD, and RTDMD

The consulted snapshots are Apache-2.0 licensed at the recorded revisions in
`UPSTREAM_REFERENCES.md`. They were used as strategy cross-checks only. No
objective, trainer, or substantial snippet was copied into this baseline.
