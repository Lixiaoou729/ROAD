# Third-party attribution and pending provenance review

## Research sources

This implementation was extracted and simplified from the author's local SDXL implementation and the SD3.5-M research snapshot. Sanitized source aliases and input hashes are recorded in `docs/source_manifest.json`. Changes include shared orchestration, strict manifests/checkpoint APIs, narrowed attention and metric helpers, and the behavioral corrections listed in `docs/behavior_changes.md`.

The containing research repository identifies [DMD2](https://github.com/tianweiy/DMD2) under [CC BY-NC-SA 4.0](https://github.com/tianweiy/DMD2/blob/main/LICENSE.md). Its full local root license is retained in `third_party/licenses/DMD2-CC-BY-NC-SA-4.0.md`. Exact inheritance boundaries and compatible licensing of the consolidated work require owner review.

The `pg_d2o` component lineage includes [ProjectedGAN](https://github.com/autonomousvision/projected-gan) and D2O. Their available MIT notices are preserved as `ProjectedGAN-MIT.txt` and `D2O-MIT.txt`; this does not automatically assign MIT to ROAD-specific additions or DMD2-derived code. Recovered `token_spatial.py`, selected Transformer attention classes and re-expressed CCM/CSM logic retain this provenance. No entire historical ProjectedGAN tree is distributed here.

The original SD3.5-M notice is preserved in `third_party/SD35_SOURCE_NOTICES.md`. It records consultation of Stability AI's SD3.5 MIT reference and Diffusers APIs, with native flow, text and latent helpers re-expressed locally. ROAD retains these helpers, narrows text-cache handling and removes unused infrastructure.

Selected preference adapters were extracted from the research `main/coco_eval_clip/preference_metrics.py`; reporting-scale and safe-load changes are documented. These adapters are not asserted to be license-independent of the research tree.

## Runtime dependencies and assets

PyTorch, Hugging Face Transformers/Diffusers/Accelerate, clean-fid, OpenCLIP and ImageReward remain separately installed dependencies. Their own distributions carry their licenses. GenEval is invoked from a separately provisioned upstream checkout. No pretrained weights or upstream model implementation trees are vendored.

- [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md): separate terms for backbone assets.
- [SDXL model license](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md): CreativeML Open RAIL++-M.
- [SD3.5-M model license](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/LICENSE.md): Stability AI Community License.
- Metric model and detector weights require their providers' terms and revision records.

These links identify review inputs, not a legal conclusion about redistribution. Check current applicable terms before a public release. No unsupported permissive license is assigned.
