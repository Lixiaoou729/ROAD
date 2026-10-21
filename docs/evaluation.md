# Evaluation and checkpoint selection

Generation, scoring and FID selection use version-2 manifests. Each stage records the checkpoint, image bytes and evaluation definition. Use a single profile, fixed prompts, base snapshot and precision for a comparison. Exact paper assets and table-generation settings must be established separately; neither profile is a certification of paper-equivalent scores.

## Profiles

| Convention | `release_v2` (default) | `reference_code` |
|---|---|---|
| COCO noise | CPU FP32 seed `base_seed + image_index`; base 10 by default | CPU FP32 full canonical batches; seed is batch index, ignoring base seed |
| GenEval noise | Unique global image indices across prompts | Four slot seeds, default 42–45, repeated per prompt |
| HPSv2/PickScore scale | 100 × normalized cosine | Learned `exp(logit_scale)` × normalized cosine |
| RGB-to-uint8 | Truncate after clipping/scaling | SD3.5-M rounds; SDXL truncates |
| FID / CLIP | Same explicit preprocessing and assets | Same explicit preprocessing and assets |

`reference_code` COCO requires `--reference-batch-size` from the run record and rejects `--seed`. Runtime `--batch-size` can differ: the generator reconstructs canonical batch noise and selects its rows. That preserves the noise identity, but different GPU batching can still alter floating-point outputs. GenEval permits an explicit base seed. The batch/slot noise definition comes from the inspected SD3.5-M helper and is available for both families; it does not certify the original XL CUDA RNG trajectory. Profile selection controls evaluation conventions only; it adds no training objective or architecture option.

## COCO10K inputs and generation

Preserve the unchanged order of the fixed [DMD2 COCO10K archive](https://huggingface.co/tianweiy/DMD2/blob/main/data/coco/coco10k.zip): one caption and paired COCO-2014 reference per sample. Serialize JSONL rows as:

```json
{"id":"original-image-id","caption":"Original caption","reference":"subset/original-file.jpg"}
```

Exactly 10,000 unique IDs are required. Reference paths are relative to this manifest. Do not resample captions or substitute COCO-2017. Convert trusted original pickle data offline; the release does not unpickle downloaded prompts. Counts alone cannot establish paper-draw identity: retain the manifest's provenance and hash.

With an existing step-500 generator export and prepared assets:

```bash
python -m road.evaluation.generate --protocol coco10k --profile release_v2 --checkpoint outputs/sdxl/generator-0000500 --base-model assets/sdxl --prompts data/coco10k.jsonl --output outputs/coco-step500 --batch-size 4 --seed 10
python -m road.evaluation.coco --images outputs/coco-step500 --prompts data/coco10k.jsonl --assets configs/evaluation_assets.example.json --metrics fid --fid-cache outputs/fid-reference-cache --output outputs/fid-step500.json
```

For the research helper profile, replace `--profile release_v2 --seed 10` with `--profile reference_code --reference-batch-size "$REFERENCE_BATCH_SIZE"`, after setting that variable from the actual run record. It is not inferred from current GPU memory or runtime batch size.

Outputs must use new directories. A complete set has `_SUCCESS`, ordered image hashes and protocol metadata. Scoring verifies all 1024×1024 RGB files and rejects partial or altered generations. Version-1 generation directories must be regenerated, not relabeled. Metadata records family, checkpoint step/hash, base snapshot, seeds, native precision/VAE policy, source hashes, versions and numerical backend settings.

## Metric assets and conventions

Edit `configs/evaluation_assets.example.json` to point to provisioned local files. Paths are relative to the working directory. Its optional `provenance` mapping should record each asset's provider/revision and any conversion steps. Only assets for requested metrics are loaded and hashed.

| Metric | Required assets and computation |
|---|---|
| FID | Explicit `inception-2015-12-05.pt` clean-fid TorchScript asset. RGB conversion, rounded center crop, 512px Lanczos, then clean-mode channel-wise 299px float bicubic resize, for both generated and reference images. Compare the generated 10K to the exact paired 10K references. |
| CLIP | HF-converted OpenCLIP ViT-g/14 LAION-2B `laion2b_s12b_b42k`, with the research 256px config. Rounded center crop/bicubic resize, native normalization, caption prefix `A photo depicts `, mean normalized cosine. The implementation checks image size and fingerprints files; size alone does not prove model/checkpoint identity. |
| HPSv2.0 | Matching HF-converted HPSv2.0 snapshot/tokenizer and its processor conventions. Profile-dependent scale above. Do not substitute HPSv2.1. |
| PickScore | `yuvalkirstain/PickScore_v1` and matching processor (the upstream example uses LAION bigG). Profile-dependent scaled cosine; never candidate-set softmax. |
| Aesthetic | Research 768-dimensional CLIP snapshot and matching linear 768→1 predictor, mean raw score. Confirm the exact predictor used for the paper's “V2” label; other MLP heads are not interchangeable. |
| ImageReward | Official ImageReward-v1.0 checkpoint and `med_config.json`, mean raw score. ImageReward 1.5 uses the pinned OpenAI CLIP dependency. Its local adapter handles three Transformers helper import moves; real-weight scoring must be validated separately. |

The explicit Inception path prevents an implicit FID weight download. ImageReward also relies on BERT tokenizer/config caches; these are not completely covered by the explicit weight inventory. Provision and record them in the controlled evaluator environment. Official GenEval may likewise access additional assets. Fully offline execution has not been certified.

Omit `--metrics` for the full list or request comma-separated `fid,clip,hpsv20,pickscore,aesthetic,image_reward`. Every metric reads the same original generated image set and applies its specified preprocessing. Results record explicit asset hashes, provenance, source hashes, versions and score conventions.

### Reference-statistics cache

`--fid-cache` stores mean/covariance in a content-addressed NPZ, with a checksum and definition JSON. The key includes ordered reference image hashes, Inception hash, preprocessing/source, package versions, device/backend arithmetic settings and feature batch size. Any relevant change creates a different key. Corrupt/incomplete entries fail rather than being silently reused. Population statistics from a different COCO set are not accepted.

FID reads explicit original file lists and preprocesses in memory; it does not materialize two temporary 10K PNG trees. First use still computes all reference features; subsequent candidates reuse the matching statistics. Keep metric batch size/settings fixed and serialize jobs writing the same cache key; concurrent writers fail safely rather than replacing an existing result.

## Select once, report one checkpoint

Repeat generation/FID at 500-update intervals. After producing step-500 and step-1000 results with identical settings:

```bash
python -m road.evaluation.select --results outputs/fid-step500.json outputs/fid-step1000.json --output outputs/selected.json
```

The lowest finite FID wins; ties select the earlier step. Selection rejects mixed FID definitions, invalid steps and duplicate candidates, and records the supplied candidate table and result-file hashes. It covers only supplied files: provide the complete planned series yourself. FID-only and full-metric results can share a `fid_protocol_hash`; the broader all-metric `protocol_hash` is separate.

To score all metrics for the selected COCO generation, set `SELECTED_IMAGES` to that generation directory, then run:

```bash
python -m road.evaluation.coco --images "$SELECTED_IMAGES" --prompts data/coco10k.jsonl --assets configs/evaluation_assets.example.json --selection outputs/selected.json --fid-cache outputs/fid-reference-cache --output outputs/selected-metrics.json
```

The guard checks both selected checkpoint hash and exact generation identity, preventing a fresh seed/image set from silently replacing the FID-selected images. Keep the same metric batch size and environment if reusing FID. For a selected checkpoint's GenEval run, pass the same selection file to generation and scoring; the checkpoint must match, while the prompt/image set is naturally different.

## GenEval for SD3.5-M

Provision a pinned [official GenEval checkout](https://github.com/djghosh13/geneval) at `assets/geneval`, its detector at `assets/geneval-detector`, and a separate compatible Python environment. Set `GENEVAL_PYTHON` to that environment's interpreter and `SELECTED_CHECKPOINT` to the FID-selected SD3.5-M generator export. With a selection file from that SD3.5-M run:

```bash
python -m road.evaluation.generate --protocol geneval --profile release_v2 --checkpoint "$SELECTED_CHECKPOINT" --base-model assets/sd35m --prompts assets/geneval/prompts/evaluation_metadata.jsonl --selection outputs/sd35m-selected.json --output outputs/geneval-selected
python -m road.evaluation.geneval --images outputs/geneval-selected --selection outputs/sd35m-selected.json --geneval-repo assets/geneval --detector assets/geneval-detector --python "$GENEVAL_PYTHON" --output outputs/geneval-scores
```

Use all 553 official metadata records and four images per prompt. The wrapper checks all 2,212 image hashes, preserves metadata, invokes the official evaluator/summary with default thresholds, and records scorer/detector hashes and stdout. The final score is the unweighted mean of six task accuracies. Upstream Mask2Former/OpenCLIP/compiler/CUDA requirements are separate from ROAD's training environment.

No paper scores are asserted here. Exact converted asset identities and table-generation settings require confirmation before claiming comparability; see [implementation decisions](behavior_changes.md).
