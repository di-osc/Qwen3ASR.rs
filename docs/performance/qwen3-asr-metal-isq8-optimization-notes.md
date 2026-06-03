# Qwen3-ASR Metal ISQ8 Optimization Notes

Date: 2026-06-03

This document records the recent Metal decode optimization work against
`mistral.rs`, with separate notes for changes that helped, changes that did not
help, and the remaining performance gap.

## Benchmark Scope

Primary target:

- Model: `Qwen/Qwen3-ASR-0.6B`
- Device: Apple Metal
- Runtime dtype: BF16
- Main optimization target: ISQ 8-bit (`auto8`)
- Workload: text decoder decode speed after audio encoder/prefill
- Fixture: `fixtures/audio/asr_en_16k.wav`
- vASR command:

```bash
cargo run --release -p vasr-models \
  --example bench_transcribe \
  --features 'metal-paged-attn timing audio-loading' \
  -- /Users/wangmengdi/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0 \
  fixtures/audio/asr_en_16k.wav \
  5 64 bf16 auto8
```

Reference command for mistral.rs paged attention:

```bash
./target/release/mistralrs bench \
  -m Qwen/Qwen3-0.6B \
  --isq 8 \
  --paged-attn on \
  --pa-context-len 4096 \
  --prompt-len 0 \
  --depth 214 \
  --gen-len 64 \
  --iterations 5 \
  --warmup 1
```

Reference command for mistral.rs default eager path:

```bash
./target/release/mistralrs bench \
  -m Qwen/Qwen3-0.6B \
  --isq 8 \
  --prompt-len 0 \
  --depth 214 \
  --gen-len 64 \
  --iterations 5 \
  --warmup 1
```

## Current Results

Best clean ISQ8 comparison observed before machine-level Metal slowdown:

| Runtime | Mode | Decode speed |
| --- | --- | ---: |
| vASR | BF16 + `auto8` + paged-attn | `202-205 tok/s` hot |
| mistral.rs | `--isq 8 --paged-attn on` | `202.8 +/- 0.5 tok/s` |
| mistral.rs | `--isq 8` default eager | `231.9 +/- 1.3 tok/s` |

Later validation after the machine/GPU slowed down:

| Runtime | Mode | Decode speed |
| --- | --- | ---: |
| vASR | BF16 + `auto8` + paged-attn | `170-176 tok/s` |
| mistral.rs | `--isq 8 --paged-attn on` | `175.6 +/- 3.6 tok/s` |
| vASR | BF16 + paged-attn | `109-126 tok/s` |
| mistral.rs | `--dtype bf16 --paged-attn on` | `122.3 +/- 4.8 tok/s` |

Interpretation:

- The ISQ8 paged-attn path is effectively at mistral.rs forced paged-attn level.
- mistral.rs default eager remains faster than both paged paths.
- Later absolute numbers dropped for both projects, so current-session numbers
  must be compared only against same-session reference runs.

## Effective Optimizations

### 1. Decode-One QKV Shape Fast Path

Files:

- `vasr_models/src/model/thinker_text.rs`

Change:

- For `seq_len == 1`, reshape Q/K/V directly as:
  - Q: `(batch, num_attention_heads, 1, head_dim)`
  - K/V: `(batch, num_key_value_heads, 1, head_dim)`
- Avoid the slower `(batch, seq, heads, dim) -> transpose(1, 2)` path.

Observed effect:

- ISQ8 decode improved from roughly `152 tok/s` to `168-171 tok/s`
  before later RoPE work.

Why it helps:

- Decode is dominated by one-token steps; avoiding per-layer transpose/view
  churn reduces CPU/GPU scheduling and layout overhead.
- This mirrors mistral.rs Qwen3 decode-one layout.

### 2. Metal mRoPE Single-Token Fast Path

Files:

- `vasr_models/src/model/rope/mrope.rs`
- `vasr_models/src/model/thinker_text.rs`

Change:

- Added `MultimodalRotaryEmbedding::forward_first_modality`.
- For interleaved mRoPE with `seq_len == 1`, avoid full multimodal position
  construction and use the first modality directly.
- Replaced Candle tensor operations for decode-one RoPE with
  `mistralrs_quant::rotary::apply_rotary_qk`.

Observed effect:

- Main jump: approximately `170 tok/s` to `202-205 tok/s` ISQ8 hot
  decode before machine-level Metal slowdown.

Why it helps:

- The previous RoPE path built `cat`, `narrow`, `mul`, `add`, and rotate-half
  tensors for every layer on every single-token decode step.
- The Metal rotary kernel reduces the per-layer decode-one overhead
  substantially.

### 3. Paged Decode Metadata Reuse

Files:

- `vasr_models/src/model/paged_kv_cache.rs`
- `vasr_models/src/model/generation.rs`

Change:

- Added `decode_metadata_for_steps(prompt_len, steps, device)`.
- Precomputes per-step slot/context metadata instead of rebuilding tensors for
  each token inside the decode loop.

Observed effect:

- Small but measurable improvement.
- `decode_metadata_ms` became `0.000` in the benchmark output.

Why it helps:

- Removes repeated host-side metadata construction and Tensor creation from the
  per-token decode loop.

### 4. ISQ8 Fused Gate/Up MLP Path

Files:

- `vasr_models/src/model/isq_linear.rs`
- `vasr_models/src/model/thinker_text.rs`

Change:

- Added a Metal `AFQ` gate/up fused path for SiLU-family MLPs.
- Keeps dense fallback for unsupported layers or devices.

Observed effect:

- Helped close the ISQ8 decode gap after QKV/RoPE improvements.
- Useful specifically for the `auto8` target.

Why it helps:

- Reduces separate quantized linear launches and intermediate tensor movement
  around the MLP gate/up pair.

### 5. Mistral-Compatible ISQ Selection

Files:

- `vasr_models/src/model/isq_linear.rs`

Change:

- Added `auto`, `auto8`, `auto6`, `auto4`, and explicit aliases such as
  `q8_0`, `q6_k`, `q4_k`.
- For this phase, `auto8` is the important production path.

Observed effect:

- Not a raw kernel speedup by itself, but it makes benchmark/config behavior
  predictable and comparable with mistral.rs.

### 6. Paged Attention Path as Default Dense Single-Sequence Decode

Files:

- `vasr_models/src/model/generation.rs`
- `vasr_models/src/model/paged_kv_cache.rs`
- `vasr_models/src/model/thinker_text.rs`

Change:

- Dense single-sequence decode uses paged-attn unless
  `VASR_DISABLE_PAGED_ATTN=1` is set.

Observed effect:

- This is currently the only path that reaches mistral.rs forced paged-attn
  performance level.

Why it helps:

- Avoids the old dynamic KV cache append/cat cost.
- Uses the same family of paged-attention kernels as mistral.rs.

### 7. Preallocated Normal KV Cache for Non-Paged Fallback

Files:

- `vasr_models/src/model/kv_cache.rs`

Change:

- Replaced per-step `Tensor::cat` cache growth with preallocated backing tensors.
- Appends new K/V with `slice_set`.
- Grows in `CACHE_GROW_SIZE = 256` token chunks.

Observed effect:

- Non-paged BF16 fallback improved from roughly `40-45 tok/s` to roughly `50-53 tok/s` in some
  runs.

Why it helps:

- Avoids copying the entire historical K/V cache on every generated token.
- This moves our fallback cache design closer to mistral.rs `SingleCache`.

## Ineffective or Negative Optimizations

### 1. Replacing Argmax with mistral.rs TopK Kernel

Files:

- `vasr_models/src/model/metal_argmax.rs`

Change:

- Replaced custom two-stage argmax with
  `mistralrs_quant::metal_kernels::topk_logits_packed` at `k=1`.

Observed effect:

- No meaningful decode speedup.
- `decode_argmax_ms` mostly represented GPU synchronization/waiting for earlier
  work, not actual argmax compute.

Conclusion:

- Good for implementation alignment, but not a primary performance lever.

### 2. Disabling Paged Attention

Command:

```bash
VASR_DISABLE_PAGED_ATTN=1 cargo run --release -p vasr-models \
  --example bench_transcribe \
  --features 'metal-paged-attn timing audio-loading' \
  -- MODEL_DIR fixtures/audio/asr_en_16k.wav 5 64 bf16 auto8
```

Observed effect:

- Old non-paged path was roughly `40-45 tok/s`.
- After preallocated cache and repeated-K/V SDPA, it reached roughly
  `50-53 tok/s` in some runs.
- Much slower than paged-attn and mistral.rs default eager.

Conclusion:

- Keep paged-attn as the production default.
- Non-paged eager needs a deeper cache/attention rewrite before it is useful.

### 3. Calling Candle SDPA with Unrepeated GQA K/V

Files:

- `vasr_models/src/model/attention.rs`
- `vasr_models/src/model/thinker_text.rs`

Change:

- Tried to pass grouped K/V directly to Candle SDPA, similar to how mistral.rs
  calls `Sdpa.run_attention`.

Observed effect:

- Output became incorrect/garbled.

Reason:

- Our current direct Candle SDPA usage does not handle grouped-query K/V in the
  same way as mistral.rs' `Sdpa` wrapper.

Conclusion:

- We must either repeat K/V before Candle SDPA or port the relevant mistral.rs
  SDPA wrapper behavior more directly.

### 4. Candle SDPA After `repeat_kv`

Files:

- `vasr_models/src/model/attention.rs`
- `vasr_models/src/model/thinker_text.rs`

Change:

- Repeated K/V first, then tried Candle accelerator SDPA.

Observed effect:

- Correct output.
- Non-paged BF16 fallback improved only modestly, roughly from `40-45 tok/s`
  to around `50 tok/s` in some runs.
- Still far slower than mistral.rs default eager.

Reason:

- Repeating K/V before SDPA adds copy/layout overhead.
- This does not reproduce mistral.rs' efficient GQA-aware path.

Conclusion:

- Useful as a fallback improvement, but not enough to match mistral.rs default
  eager.

### 5. Parallel Benchmark Runs

Observed effect:

- Running vASR BF16 and ISQ8 benchmarks in parallel caused Metal/GPU contention.
- Numbers dropped sharply and were not reliable.

Conclusion:

- Run Metal benchmarks sequentially.
- Re-run mistral.rs in the same session when absolute numbers look suspicious.

## Current Gap Analysis

### ISQ8 Paged-Attn

Status:

- Essentially matched with mistral.rs forced paged-attn.

Remaining gap:

- Mostly measurement variance and machine Metal load.

### BF16 Paged-Attn

Status:

- Close to mistral.rs forced paged-attn.

Remaining gap:

- Small variance plus possible per-run Metal synchronization differences.

### Default Eager Path

Status:

- Still significantly behind mistral.rs default eager.

Primary cause:

- mistral.rs uses a more complete `NormalCache`/`SingleCache` and `Sdpa`
  abstraction.
- Our fallback still needs either K/V repeat or a less optimized manual
  matmul/softmax path.

Next useful work:

1. Port a closer equivalent of mistral.rs `Sdpa.run_attention` for GQA.
2. Port more of mistral.rs `SingleCache` semantics, including cache capacity,
   snapshots, and append behavior.
3. Avoid explicit `repeat_kv` in eager decode.
4. Add a dedicated text-only Qwen3 decoder benchmark that removes audio encoder
   and ASR prompt noise, so eager-path regressions are easier to isolate.

## Verification Commands

Commands used during this optimization pass:

```bash
cargo fmt --all -- --check
cargo check -p vasr-models --features metal-paged-attn
cargo test -p vasr-models kv_cache --features metal-paged-attn
cargo test -p vasr-models isq_linear --features metal-paged-attn
git diff --check
```

For release decode speed:

```bash
cargo run --release -p vasr-models \
  --example bench_transcribe \
  --features 'metal-paged-attn timing audio-loading' \
  -- MODEL_DIR fixtures/audio/asr_en_16k.wav 5 64 bf16 auto8
```

For non-paged fallback testing:

```bash
VASR_DISABLE_PAGED_ATTN=1 cargo run --release -p vasr-models \
  --example bench_transcribe \
  --features 'metal-paged-attn timing audio-loading' \
  -- MODEL_DIR fixtures/audio/asr_en_16k.wav 5 64 bf16
```
