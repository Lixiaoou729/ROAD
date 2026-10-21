# Training and checkpoints

## Numerical contract

A completed step is one accumulated G optimizer update followed by one accumulated D optimizer update. G minimizes `mean(softplus(-D(fake)))`; D minimizes `mean(softplus(-D(real))) + mean(softplus(D(fake)))` plus the real-image penalty. D receives newly drawn fakes from the updated generator. There is no DMD auxiliary term, CFG or EMA.

The YAML loader rejects unknown keys, nonfinite coefficients, invalid booleans and inconsistent effective batches. `world_size * microbatch * accumulation` must equal `global_batch`. Use the supplied complete family recipes rather than constructing a partial SD3.5-M recipe from generic training defaults. The 100,000-step budget is configurable; it does not identify the historical best checkpoint.

The three-stage generator VJP obtains image-space gradients, differentiates the VAE, then recomputes the generator for backward. RNG replay prevents recomputation from consuming extra draws. `generator_vjp: direct` is an operational gradient-comparison path, not another objective. Generator/VAE/feature/critic activation checkpointing is retained. Critic math attention supports higher-order derivatives; native generator attention stays under Diffusers. ZeRO-1 shards generator AdamW state, and DDP reduces learned-model gradients.

Spectral-norm power iteration advances once before each logical G/D pair. Only SN-bearing layers have their training flag disabled during forward/recomputation; learned weights still receive gradients. Estimates stay fixed through the accumulated updates. This differs intentionally from historical forward-dependent updates; see [behavior decisions](behavior_changes.md).

## Regularization and refresh

For a full logit vector of length n, JR1 estimates `||J_d(x)||_F²/n` using one independent Rademacher probe and the squared image gradient of `sum(probe*d)/sqrt(n)`. Pooled R1 differentiates `mean(d)`. Both include patch and CLS outputs; dummy-register outputs are excluded. Four heads each emit 196 patch logits and one CLS logit, for 788 in total.

The default lazy penalty occurs on updates 16,32,… . Every microbatch contributes `gamma * 16 / 2` times its batch-mean penalty, divided by accumulation. Gamma zero avoids the penalty graph. Defaults are gamma 0.0075 for SDXL and 0.00075 for SD3.5-M.

Refresh occurs after updates 5,000,10,000,…, before saving. Each rank redraws the frozen mixing using its training RNG, then rank-zero tensors are broadcast. The saved `last_refresh_step` makes a repeated boundary call a no-op. Learned modules and optimizer state survive refresh. `null` disables it; nonpositive intervals are rejected.

## Data workers and distributed execution

Sampling is addressed by completed step, epoch permutation and rank. The incomplete global batch at each epoch boundary is dropped. Each sample has its own augmentation seed. Workers receive explicit sample/seed tasks and read CPU image tensors and text-cache rows. Prefetch does not advance the saved training position; resume discards unconsumed work and reconstructs the next task. A private loader RNG keeps worker startup out of the training RNG stream.

`data_workers: 4`, `prefetch_factor: 2` and `pin_memory: true` are per-rank defaults. Set workers to zero for synchronous loading. Spawn avoids forking live CUDA/NCCL state. One iterator serves the entire run; workers are not recreated each epoch. Host-memory demand includes prefetched microbatches plus the current accumulated step. Tune workers/prefetch for your node's CPU, RAM and storage; throughput has not been measured for this revision.

All ranks require identical immutable assets and a shared checkpoint/output filesystem. Rank zero hashes large model/data/cache assets once; other ranks rely on that shared-storage contract. Configuration/code identities are compared across ranks. Ordinary loader, checkpoint and logging errors are propagated collectively; process death and unmatched/NCCL failures still require the launcher to terminate the job. Logs contain globally averaged G/D losses.

## Checkpoints

**Full training state, schema 2:** `state-0005000/` contains the generator, critic, mixing and refresh counter, both optimizers, completed-step counter, per-rank Python/NumPy/Torch/current-device CUDA RNG, file hashes and numerical identity. ZeRO state is consolidated collectively. The sampler position follows from the completed step.

**Generator export, schema 2:** `generator-0000500/` contains native Diffusers safetensors/config, scheduler config and `road.json`. Metadata records family, step, base-asset identity, precision, VAE/text policies and file hashes. It omits full private training configuration and paths; a hash links to the run identity. Local `_name_or_path` bookkeeping is removed from saved native configs. Inference loads the exported scheduler and native text encoders without importing critic/DINO training components.

Writes use `.partial` staging and refuse existing destinations. Hashes detect incomplete or altered artifacts; they are not signatures proving a file is trustworthy. Only load trusted checkpoints. Failed partial output is retained for inspection, never treated as complete.

```bash
torchrun --standalone --nproc_per_node=8 -m road.train_sdxl --config configs/sdxl.yaml --resume outputs/sdxl/state-0005000
```

Full resume requires identical numerical configuration, model/data/cache contents, ROAD source, recorded package versions and world size. Paths are replaced by content identities. Output path, total budget, checkpoint/export/log cadence, worker count, prefetch and pinning can change. Increase `training.steps` to extend a run. Other changes require a fresh experiment. If the output directory contains steps later than the resumed state, choose a new output path rather than mixing histories. `resolved_config.json` and `run_identity.json` record the continuation settings.

Schema-1 release full states and original research full states are intentionally rejected by the new trainer: their state/refresh/identity contract differs. Resume those with their original implementation. Generator-only schema-1 exports remain readable when their native asset identities match. CUDA continuation and export parity require the separate validation phase.

## Historical generator import

`road.import_generator` accepts a flat generator-only `.bin`/`.pt` tensor dictionary or safetensors, constructs the native model from the specified base config and loads with `strict=True`. It never strips prefixes, ignores missing keys or guesses nested structures:

```bash
python -m road.import_generator --family sdxl --base-model assets/sdxl --state-dict assets/selected-generator.bin --completed-steps 5000 --vae-dtype bf16 --output outputs/imported-sdxl
```

This example requires your own verified generator file. Supply its actual step and VAE policy; filenames do not establish them. Conversion packages generator weights for inference; it does not restore historical critic/optimizer state or establish numerical parity. It does not require historical weights for fresh release training.
