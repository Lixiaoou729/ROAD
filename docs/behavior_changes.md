# Implementation decisions and fidelity boundaries

This release implements the intended final ROAD method. It is not a promise of bitwise equivalence to an unidentified historical run. The October 19 refinement used source inspection only; no tests or runtime validation were executed.

## Mixing: retained, not reordered

The recurrence remains `mixed[l] = CCM[l](f[l])`, `out[L] = CSM[L](mixed[L])`, and `out[l] = CSM[l](out[l+1] + mixed[l])`. Moving the same-level skip outside CSM would change the result.

Evidence in the supplied references:

- Current technical-report `sections/preliminaries.tex`, lines 45–51, applies CSM to the sum. `sections/appendix_mixing.tex`, lines 90–103, includes CSM in the diagonal block `(CSM_l ... CSM_m) CCM_m`.
- The SDXL research `pg_d2o/pg_modules/projector.py`, lines 1290–1300, uses the same recurrence. `TokenCSMProjector`, lines 430–476, treats CLS independently of patches.
- ProjectedGAN `blocks.py`, lines 214–247, also adds before upsampling/projection. ROAD's equal-resolution token maps do not need that upsampling.

Both CCM and CSM average independent biased 1×1/3×3/5×5 patch convolutions, with zero padding `kernel_size // 2`. Each has a separate biased CLS linear. Actual DINO register tokens are removed before mixing; the critic receives patches, CLS, then four zero dummy registers. Dummy outputs are excluded from adversarial logits/R1. These operations, biases and ordering are unchanged in this refinement.

Refresh still redraws only Linear/Conv2d leaves. Each rank consumes its own RNG draws and rank-zero mixing tensors are broadcast. A new saved boundary counter prevents double refresh. Module construction order from the initial restructuring remains; matching an old seed does not recreate old random weights, and no attempt was made to emulate every historical RNG draw.

## Training decisions

| Area | Decision |
|---|---|
| Recipes | Keep the reconstructed final recipes: XL effective batch 64/JR1 0.0075; M batch 32/JR1 0.00075; lazy interval 16; refresh 5,000. The original launcher names do not prove the exact selected run or dataset identity. |
| Spectral normalization | Retain one power iteration per logical G/D pair outside checkpointed forwards, rather than forward/recompute-dependent updates. Only SN-bearing layers are placed in eval mode. This avoids state mutation during backward recomputation but is an intentional numerical departure from some research paths. |
| Cross-attention | Retain the initial release's nonzero output projection with zero residual gate, preventing the dead double-zero branch, and spectral normalization of the output projection. These earlier corrections differ from some historical implementations and still need real-model validation. |
| Architecture | Fixed RMSNorm Transformer heads, cosine attention, learned RoPE, depths 4/3/2/1 and four DINO taps. The active SD3.5-M LayerNorm text adapter and pretrained native norms remain. No additional scientific ablations were introduced. |
| Data | Keep step-addressed global permutations and family transforms; add spawn workers and bounded prefetch without changing sample or augmentation seeds. Unlike historical stream loaders, incomplete global batches are dropped. Prepared RGB/paper-pool identity remains external. |
| Text cache | Read individual rows through safetensors mmap after a sequential integrity scan; stop loading entire 64-caption shards for sparse samples. Cache building transfers each encoded batch to CPU immediately. Native encoder precision is unchanged. |
| Resume | Schema 2 adds NumPy RNG and refresh state, file integrity checks and collective I/O error propagation. Numerical identity excludes relocatable paths and operational settings. Old full states are not silently migrated. |
| Inference | Use the exported scheduler and recorded precision; preserve native XL FP32/M BF16 text encoding even when generator compute precision is overridden. Export metadata omits private run paths. |

## Evaluation decisions

- **Crop alignment:** use the research PIL helper's rounded center offset, including odd-edge round-to-even cases. The previous floor offset can select different reference pixels.
- **`release_v2`:** CPU FP32 per-image seeds `base_seed + global_image_index`, default base 10; HPS/PickScore report 100× normalized cosine. This preserves the first release's explicit reporting convention, not a claim that it is the table-generation convention.
- **`reference_code`:** COCO uses full canonical CPU FP32 batches seeded by batch index, without a base-seed offset; its historical batch size must be supplied. GenEval uses default seeds 42–45 for four slots, reused for each prompt. HPS/PickScore use each checkpoint's learned `exp(logit_scale)` multiplier, matching the inspected adapters.
- The seed references are SD3.5-M `sd35_coco_eval_common.py` (legacy batch mode, lines 1264–1294) and `eval_sd35_geneval.py`/`sd35_geneval_common.py` (CPU slot seeds). Preference-scale evidence is research `main/coco_eval_clip/preference_metrics.py`, including learned multipliers at lines 323–324 and 636. They do **not** identify which options/assets produced each paper table. The profile must not be described as paper-equivalent without that record.
- Pixel conversion is recorded explicitly: the research M helper (`sd35_coco_eval_common.py:1325–1326`) rounds after clipping/scaling, so `reference_code` M uses rounding; the inspected XL evaluator and `release_v2` retain truncation. The shared reference noise profile follows the M helper, not a certification of the historical XL CUDA RNG stream.
- FID uses explicit file lists and identical generated/reference preprocessing; content-addressed reference statistics replace temporary 512px PNG copies. Cache keys include reference bytes, Inception, preprocessing source, versions, arithmetic settings and batch size.
- Version-2 generation completion records seal protocol/image manifests. FID selection has its own identity, independent of whether other metrics were requested. Optional selection guards enforce the selected checkpoint, and for COCO the identical generated image set.

Neither profile is numerically validated in this round. Asset conversion parity, exact reference-set identity, historical batch size and score scaling remain requirements for comparison with paper values. See [evaluation](evaluation.md) and [validation requirements](validation.md).
