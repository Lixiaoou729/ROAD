> **Historical record (2026-10-16).** The code was subsequently refined on 2026-10-19 without running tests or runtime validation, as requested. Results below apply only to the earlier implementation. See the remaining validation requirements below.

# Validation record and remaining release gates

Validated locally on macOS arm64 / Python 3.11, with PyTorch 2.8.0, Diffusers 0.36.0 and Transformers 4.57.1. CUDA and full pretrained assets were not used. `validation_environment.txt` records resolved installed packages; `validation_checks.json` records CLI and preservation checks.

## Executed checks

`python -m pytest -q`: **32 passed, no skips**, 23.42 seconds on the final test run. One warning comes from an upstream CUDA autocast decorator being disabled on this CPU machine.

| Coverage | Evidence and limit |
|---|---|
| JR1 / pooled R1 | Analytical cancellation example; enumerate all Rademacher probes to recover exact normalized Jacobian energy; compare weight gradients and accumulated batch scaling; lazy interval and gamma zero. |
| Critic | Patch+CLS count, dummy-register exclusion, nonzero cross-attention output, nonzero initial gate gradient, SD3.5-M LayerNorm adapter, and higher-order backward through checkpointed heads. Tiny dimensions for inexpensive tests. |
| Mixing | Frozen parameters, gradients to features, interval/null behavior, no learned-module or optimizer reset. |
| Generator VJP | Direct/staged gradients agree with dropout; recomputation restores RNG progression. Toy decoder, not full CUDA VAE. |
| Full state | Toy AdamW update equality after save/load; saved refresh boundary; Torch RNG; strict mismatch rejection. |
| Distributed | Two spawned Gloo CPU workers: DDP + accumulation equals global-batch JR1/pooled gradients; mixing broadcast; ZeRO-1 save/load and subsequent optimizer step. NCCL/CUDA remains untested. |
| Native libraries | Tiny real Diffusers SDXL UNet/VAE and SD3 dual-attention Transformer forwards/backwards; DINOv3 native 12-layer B/16-width model with narrowed FFNs and an image-space second derivative; fixed/dynamic native sigma-1 schedule checks. Randomly initialized test models, not pretrained inference. |
| Inference/export | Injected small backend verifies deterministic per-image seed behavior; strict flat legacy keys; native small UNet safetensors export and file hashes. No selected paper generator converted. |
| Data/cache | RGB processing, SDXL white alpha compositing, deterministic M crop, direct tar offsets, asset mutation detection, safetensors cache mismatch rejection, resumed batch order. |
| Evaluation | Count/protocol/selection checks, GenEval six-task aggregation/coverage, and native small ImageReward BLIP BERT forward through the relocated-helper compatibility adapter. No full metric values computed. |

Additional executed checks: ten CLI `--help` entry points, both recipe `--validate-config` commands, compilation of all Python modules, Ruff static checks/format validation, editable core/evaluation installation and `pip check`. The missing OpenAI CLIP dependency and ImageReward's moved Transformers helpers were found by actual imports and corrected. Evaluation weights are still needed to validate complete scoring.

The original local SDXL and research snapshot trees were compared against **4,578 pre-implementation file hashes**. No research source or experimental artifact was modified, deleted or added by this implementation. One snapshot `.DS_Store` Finder metadata file differed; it was left untouched. The unrelated parent repository was not fetched, reset, committed or pushed.

A targeted source/docs/config scan found no private home/mount paths, credential assignments or private-key blocks. This is a scoped hygiene check, not a comprehensive security certification. The exposed W&B key identified by the earlier audit was not copied; the owner must revoke/rotate it separately.

## Required next validation with real assets

1. **Approve asset identities.** Record selected historical generator hashes, base/DINO revisions, prepared training-pool provenance, fixed COCO10K order/reference hashes, and each metric model/conversion. Do not substitute different data and call it a paper reproduction.
2. **CUDA smoke for both families.** On target hardware, run at 1024px through at least two lazy-R1 events. Check finite G/D/penalty values, gate gradients, nonzero learned updates, no updates to DINO/mixing outside refresh, and record peak memory/throughput. Exercise direct/staged VJP on a feasible identical batch and compare gradients within a documented BF16 tolerance.
3. **Distributed continuation.** Use actual NCCL world sizes and sharded optimizers. In a short test-only copy of the recipe, set a small refresh/save interval, cross the event, save and compare resumed next-update losses/weights/RNG to uninterrupted execution. Preserve the main recipes. Confirm full 5,000-step counter semantics without needing a complete retraining run.
4. **Historical generator comparison.** Strictly import the selected generator-only dictionaries, match native text/VAE precision and identical input noise, and compare latent/RGB outputs to the research backend. Do not try to use renamed/corrected critic state as a historical full resume.
5. **Metric comparison and selection.** Compare retained scorers on fixed sample images against the reference implementation, especially HF CLIP/HPS conversions, aesthetic predictor identity, and HPS/PickScore scales. Run all planned 500-step FID candidates; select once; score all remaining metrics and GenEval from that same checkpoint. Record the complete selection table and score outputs.
6. **Owner approval before publication.** Resolve integrated source licensing and model/data/checkpoint redistribution; retain required notices. Rotate the exposed W&B credential. Review the prepared repository and real-asset validation record. Public distribution requires a separate decision after these checks.

The initial implementation preparation is complete. It is suitable for review and real-asset integration testing, **not yet a cleared public release or demonstrated reproduction of the paper's scores**. Full retraining is not required to review this implementation.
