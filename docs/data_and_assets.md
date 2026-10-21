# Data and assets

## Native model snapshots

Provide complete local snapshots, including configuration, tokenizer, text encoder, generator, scheduler and VAE files:

| Local directory | Upstream model |
|---|---|
| `assets/sdxl` | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| `assets/sd35m` | [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) |
| `assets/dinov3-vitb16` | [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m), Transformers format |

Accept the applicable upstream access terms yourself. Pin and record the exact downloaded revisions. Do not put access tokens into YAML, scripts or Git. Training/inference load local files only and hash relevant snapshot contents. Use one consistent snapshot for text caching, training and inference; replacing even a tokenizer invalidates the fingerprint. DINO is 768-dimensional, 12 layers, patch size 16, with four registers; taps are blocks 3/6/9/12 (one-based).

The release does not redistribute pretrained assets. Base checkpoint, DINO and downstream checkpoint licenses are separate from source-code rights.

## Training pool

The paper specifies the fixed 500K LAION-Aesthetics 5.5+ pool distributed by DMD2, after unsafe/low-resolution filtering. The inspected reference launcher names mention 480K; this alone cannot establish identity with the paper pool. Do not replace the pool with a new random LAION sample and label that a reproduction.

The [DMD2 data location](https://huggingface.co/tianweiy/DMD2/tree/main/data/laion_vae_latents) is a source pointer, **not a complete RGB preparation recipe**: latent-only records cannot be silently treated as original images. The release consumes prepared image/caption pairs. An approved manifest tying those pairs to the paper pool, the exact original decoding recipe where latent inputs were used, and redistribution rights remain owner-supplied requirements. Arbitrary legally obtained pairs can exercise training but establish a different experiment.

For prepared tar shards, pair `sample.jpg`/`.png`/`.webp` with `sample.txt`, or `sample.json` containing `caption` or `text`. Use unique sample stems within each shard. TXT takes precedence over JSON regardless of archive order. Duplicate image/caption entries, images without string captions and empty shards are rejected rather than silently changing the pool. For the SDXL dataset, run:

```bash
python -m road.prepare_data --shards data/sdxl/shards/*.tar --output data/sdxl/train.jsonl
```

This writes ordered JSONL records with `id`, `caption`, `image` (relative shard path) and `member`, plus `train.assets.json` containing SHA-256 hashes. Uncompressed tars include byte offsets for direct reads; compressed archives work but random access is slower. Data are never extracted or rewritten. Index the SD3.5-M source shards separately into `data/sd35m/train.jsonl`; do not silently substitute the already cropped SDXL pool. Training verifies the sidecar and source bytes before starting. IDs incorporate shard hashes; the index is not proof of membership in the paper pool.

SDXL inputs must already be 512×512; the reader applies EXIF orientation, composites alpha onto white, converts to RGB and scales to [-1,1], with no random flip. SD3.5-M converts to RGB, resizes the shorter edge to 1024 with Lanczos, then applies seeded random crop and flip. DINO normalizes before a 224px bicubic antialiased resize. These are model-specific paths; do not share already-transformed tensors between families.

Advanced file manifests can use `{"id":"...","caption":"...","image":"relative/image.png"}`. Supply the same sidecar schema (`count`, `manifest_sha256`, and `shards`, mapping every referenced file to its SHA-256). The `shards` field also covers ordinary image files. Content is assumed immutable during a running job.

## Text caching and storage

Run `road.text_cache` with the matching family/base snapshot, manifest and empty output directory. It writes bounded safetensors shards and a final `metadata.json`; a partial cache is rejected. Captions are not rewritten. SDXL uses both CLIP penultimate hidden states and the second pooled projection; SD3.5-M preserves dual CLIP + T5 concatenation, 77 T5 tokens, 154 total tokens and native pooled projections. The SD3.5-M critic's active 4096→1024 LayerNorm/linear adapter remains trainable.

Cache generation uses FP32 SDXL encoders and BF16 SD3.5-M encoders; training casts loaded tensors to the configured compute dtype. At 500K samples, token storage alone is approximately **315 GB for SDXL FP32** and **631 GB for SD3.5-M BF16**, before metadata/pooled tensors. Budget disk space explicitly. Workers mmap only the requested tensor rows, then copy them into CPU batches. Cache construction transfers each encoder batch to CPU before accumulating a shard. A full sequential hash scan runs once on rank zero before workers start; at these sizes startup I/O is substantial. The cache must remain immutable on the shared filesystem for the job duration.

The deterministic indexed loader uses spawn workers, bounded prefetch and optional pinned memory. `training.data_workers: 0` selects synchronous loading. Sample and crop/flip seeds come from the step-addressed sampler, not worker RNG. Prefer uncompressed tar shards for direct byte-range reads; compressed archives may repeatedly decompress and remain a throughput bottleneck. Local SSD/shared high-throughput storage helps, but worker count and prefetch must be measured in the separate testing phase. No new throughput claim is made.
