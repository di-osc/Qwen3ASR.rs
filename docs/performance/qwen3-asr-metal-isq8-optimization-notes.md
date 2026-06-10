# Qwen3-ASR Metal ISQ8 Optimization Notes

Date: 2026-06-08

## Update: 2026-06-08 (Pass 14 — Metal RoPE Fast Path + Eager Routing)

Latest on-this-machine baseline for `Qwen/Qwen3-ASR-0.6B` text decode after the
Pass 14 optimizations, ISQ 8 (AFQ), single sequence:

```bash
cargo run --release -p vasr-models --example bench_text_decode \
  --features "metal,metal-paged-attn,paged-attn,timing" \
  -- Qwen/Qwen3-ASR-0.6B "The capital of France is" 3 64 bf16 8 --warmup 1
```

Observed on macOS Metal:

- **`tokens/s`: `~236-238 tok/s`** (3–5 runs, 64-step decode)
- `decode_forward` ≈ 281 ms (was 1555 ms)
- `decode_argmax` ≈ 522 ms (was 454 ms, now reflects GPU sync after faster forward)

Control checks:

- `VASR_DISABLE_PAGED_ATTN=1` → ~237 tok/s (same path, confirms eager routing)
- `VASR_FORCE_PAGED_ATTN=1` → ~170–175 tok/s (paged single-seq, now opt-in)

For direct reference, on the same machine:

```bash
/Users/wangmengdi/.cargo/bin/mistralrs bench \
  -m Qwen/Qwen3-0.6B \
  --isq 8 \
  --prompt-len 0 \
  --depth 214 \
  --gen-len 64 \
  --iterations 3 \
  --warmup 1
```

→ `253.0 ± 2.0 T/s` (latency 3.95 ms/T).

### Batch decode (concurrent sequences)

```bash
cargo run --release -p vasr-models --example bench_text_decode \
  --features "metal,metal-paged-attn,paged-attn,timing" \
  -- Qwen/Qwen3-ASR-0.6B "The capital of France is" 3 64 bf16 8 --batch <N> --warmup 1
```

| batch | batch tok/s | per-seq tok/s | scaling |
|-------|-------------|---------------|---------|
| 1 | **238** | 238 | 1.00× |
| 2 | **450** | 225 | 1.89× |
| 4 | **658** | 165 | 2.77× |

Batch throughput scales near-linearly because Metal GPU parallelises matmul,
SDPA, and RoPE across the batch dimension within each decode step.

### 结论（Pass 14 之后）

- **单序列 ISQ8 已基本追平 mistral.rs**（238 vs 253 tok/s，差距 ~6%）。
- 剩余差距主要来自 Q/K norm 未与 mRoPE 融合（`forward_qk_norm` 需要适配 3D
  mRoPE positions），以及 benchmark 测试方式差异（vasr 包含 prompt prefill
  设置，mistralrs `--prompt-len 0` 跳过 prefill）。
- **并发 batch decode** 总吞吐远超 mistralrs 单序列（batch=2 → 450 tok/s）。

### Pass 14 优化记录

见下文新增的 [Pass 14](#2026-06-08-pass-14-metal-rope-fast-path--eager-routing) 小节。

Historical notes below (from earlier passes): 2026-06-03

This document records the recent Metal decode optimization work against
`mistral.rs`, with separate notes for changes that helped, changes that did not
help, and the remaining performance gap.

Update: 2026-06-04 (Pass 13)

Metal production default is **eager KV decode** (single + batch) with **ISQ8
(`auto8`)** weights. HTTP serve and `bench_transcribe_dir` now use an **async
Loader→VAD→ASR pipeline** that overlaps stages across files; ASR micro-batches default
to **≤60 s audio per batch** (`max_batch_audio_sec`). End-to-end `raw_audios` (20 files)
reaches **65.6× speedup / RTF 0.0152** (wall **25.2 s**, down from **25.5 s** Pass 12).
Use `VASR_FORCE_PAGED_ATTN=1` only for paged regression tests; forced paged can exhaust
the KV block pool on long multi-segment clips.

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

### 8. Mixed-Length Paged Prefill Correctness Fixes

Files:

- `vasr_models/src/model/thinker.rs`
- `vasr_models/src/model/paged_kv_cache.rs`
- `vasr_models/src/model/generation.rs`
- `vasr_models/src/model/thinker_text.rs`

Change:

- Fixed paged prefill for mixed-length ASR batches.
- `get_rope_index` now works for generic `0/1` masks instead of assuming only
  left padding.
- Paged metadata now carries pad-aware `slot_mapping`,
  `token_attention_mask`, and per-request `query_lens`.
- Batched paged prefill normalizes variable-length prompts into a single
  right-padded paged-prefill layout.

Observed effect:

- Removed the earlier mixed-batch decode corruption where outputs degraded into
  repeated garbage such as `퓮`.
- Mixed-length release repro stabilized at roughly `350 tok/s` batch decode in
  the good runs while keeping correct text output.

Why it helps:

- Before this, the main batch path was fast only for dense/equal-length inputs
  and broke on realistic VAD-segment batches.
- This was the correctness prerequisite for all later ASR batch throughput work.

### 9. Gather-Backed Paged Prefill Attention

Files:

- `vasr_models/src/model/thinker_text.rs`

Change:

- For `seq_len > 1` paged prefill, attention no longer uses only the local
  forward K/V tensors after cache write.
- It now writes paged K/V first, then gathers them back with
  `mistralrs_paged_attn::gather_kv_cache(...)` and runs attention on the
  gathered cache view.

Observed effect:

- Brought the implementation shape much closer to `mistral.rs`.
- Correct mixed-length batched prefill became stable on the single production
  path.

Why it helps:

- The attention source now matches the actual paged cache layout rather than a
  temporary local K/V view.
- This removed a major source of divergence between our ASR batch path and the
  `mistral.rs` paged-attention design.

### 10. Right-Padded Prompt Batching and Paged Prefill Fast Path

Files:

- `vasr_models/src/processor/asr_processor.rs`
- `vasr_models/src/forced_aligner/model.rs`
- `vasr_models/src/model/generation.rs`

Change:

- Batch prompt construction now uses right padding.
- Paged prefill detects already-right-padded batches and skips the extra
  normalization/repack step.

Observed effect:

- Small but repeatable throughput improvement on real `transcribe` runs.
- Helped keep the paged prefill path closer to the layout assumptions used by
  `mistral.rs`.

Why it helps:

- Removes a redundant host-side batch rewrite before paged prefill.
- Lets the production ASR path feed the paged decoder in a more direct layout.

### 11. Varlen Metadata Hoisting into PagedInputMetadata

Files:

- `vasr_models/src/model/paged_kv_cache.rs`
- `vasr_models/src/model/thinker_text.rs`

Change:

- `PagedInputMetadata` now carries:
  - `kv_lens`
  - `cu_seqlens_q`
  - `cu_seqlens_kv`
  - `max_query_len`
  - `max_kv_len`
- Paged prefill now consumes these values directly instead of rebuilding them
  inside each layer.

Observed effect:

- Modest improvement in host-side overhead.
- More importantly, this aligned our paged metadata flow with the way
  `mistral.rs` prepares varlen information.

Why it helps:

- Avoids repeated per-layer length bookkeeping.
- Makes later packed/varlen attention work easier because the right metadata is
  already attached to the request.

### 12. Prefill Mask Hoisting

Files:

- `vasr_models/src/model/paged_kv_cache.rs`
- `vasr_models/src/model/generation.rs`
- `vasr_models/src/model/thinker_text.rs`

Change:

- Added `prefill_attention_mask` and `prefill_causal_only` to paged metadata.
- Paged prefill can now reuse a single precomputed mask across decoder layers
  instead of rebuilding the same additive mask repeatedly.

Observed effect:

- Helped reduce per-layer setup overhead, though the gain was smaller than the
  later tensor-layout cleanup.

Why it helps:

- The prefill attention mask is request-global, not layer-specific.
- Hoisting it into metadata removes repeated Tensor construction work.

### 13. Fused Unpack+Group Expansion and Deferred Fallback Transpose

Files:

- `vasr_models/src/model/thinker_text.rs`

Change:

- Replaced `unpack_gathered_kv(...)` followed by `repeat_kv(...)` with
  `unpack_gathered_kv_for_attention(...)`, which directly materializes the head
  shape needed by attention.
- In the paged prefill path, `k.transpose(...).contiguous()` is no longer done
  before the accelerator attempt. It is now only built if attention falls back
  to the manual matmul path.

Observed effect:

- Mixed-length prefill time dropped from roughly `390-406 ms` to roughly
  `349-358 ms` in the small release repro.
- Real `transcribe` benchmark over 5 raw audio files improved from roughly:
  - `wall_seconds=5.110`
  - `speedup=61.723x`
  - `rtf=0.0162`
  to:
  - `wall_seconds=4.644`
  - `speedup=67.908x`
  - `rtf=0.0147`
- This was about a `9%` end-to-end improvement on that batch.

Why it helps:

- Removes one large intermediate grouped-K/V expansion.
- Avoids paying for fallback-only `transpose+contiguous` work when Metal SDPA
  succeeds.
- This is the first mixed-length paged-prefill optimization in this phase that
  clearly moved the real batch `transcribe` numbers.

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

### 6. Causal-Only Paged Prefill Shortcut

Files:

- `vasr_models/src/model/paged_kv_cache.rs`
- `vasr_models/src/model/generation.rs`
- `vasr_models/src/model/thinker_text.rs`

Change:

- Tried to treat the right-padded paged prefill case as a pure causal-attention
  problem and skip the custom padding mask on the accelerator path.

Observed effect:

- Correct output was preserved.
- Throughput gain was inconsistent and usually negligible on Metal.
- In some runs prefill even became noisier/slower.

Conclusion:

- The idea is logically valid, but in the current Candle/Metal stack it is not
  a reliable primary optimization lever.
- Keep the supporting metadata, but do not treat this shortcut as a major win.

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

### Mixed-Length ASR Batch Path

Status:

- Correctness is now stable on the single production paged path.
- Real ASR `transcribe` batch throughput improved meaningfully after the recent
  paged-prefill cleanup.

Current reference points from the 2026-06-04 pass:

- Mixed-length batch repro:
  - batch decode roughly `338-360 tok/s`
  - per-sequence decode roughly `169-180 tok/s`
- `bench_transcribe_dir` over 5 `raw_audios`:
  - `audio_seconds=315.389`
  - `wall_seconds=4.644`
  - `speedup=67.908x`
  - `rtf=0.0147`

Remaining gap:

- Still not fully at the `mistral.rs` style packed-varlen attention path.
- We still unpack gathered K/V into padded batch tensors on Metal instead of
  using a true packed varlen attention backend.

Next useful work:

1. Reduce `unpack_gathered_kv_for_attention(...)` padding/cat overhead further.
2. Port a closer packed-varlen attention path where Metal backend support allows it.
3. Keep separating correctness-fix work from real throughput wins in the notes,
   because several alignment steps are necessary but not themselves speedups.

### 2026-06-04 Follow-up: Shared Attention Dispatch

Goal:

- Start moving more attention paths, not only paged prefill, onto the local
  `mistral.rs`-style dispatch abstraction without giving back the Metal gains
  from the previous mixed-length work.

Retained changes:

- Kept `PagedInputMetadata::flash_params(...)` as the metadata-side bridge from
  paged batch inputs to `FlashParams`.
- Extended `thinker_text.rs` so the normal eager attention paths also route
  through `attention::run_attention(...)`, instead of each branch open-coding
  its own `accelerated_sdpa` / manual fallback split.
- Tightened `run_attention(...)` so it only considers the flash-attn path when
  it is actually viable:
  - CUDA, or
  - CPU with the `flash-attn` feature enabled.
- This avoids the earlier wasteful "prepare for flash anyway" behavior on Metal.

Why this one stayed:

- Mixed-length batch correctness stayed stable.
- The small mixed-length repro returned to the pre-regression range after the
  flash-attempt gate was tightened.
- End-to-end `transcribe` throughput also held or improved slightly, so this is
  now a structural cleanup that does not cost us runtime.

Reference measurements:

- Mixed-length batch repro (`fixtures/audio/asr_en_16k.wav` +
  `raw_audios/audio (12).wav`):
  - run 1:
    - `audio_encoder_ms=155.303`
    - `prefill_ms=352.976`
    - `decode_ms=44.701`
    - `batch_decode_tokens_per_s=357.934`
    - `per_sequence_decode_tokens_per_s=178.967`
  - run 2:
    - `audio_encoder_ms=130.893`
    - `prefill_ms=348.174`
    - `decode_ms=43.775`
    - `batch_decode_tokens_per_s=365.505`
    - `per_sequence_decode_tokens_per_s=182.753`
- `bench_transcribe_dir` over 5 `raw_audios`:
  - `audio_seconds=315.389`
  - `wall_seconds=4.600`
  - `speedup=68.570x`
  - `rtf=0.0146`

Takeaway:

- This did not unlock a new dramatic decode-speed jump by itself.
- It did get more of the codebase onto a single attention-dispatch shape that
  is much closer to `mistral.rs`, while preserving the current Metal numbers.
- That should make the next step, pushing paged prefill closer to a true
  packed-varlen flow, less brittle.

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
  -- MODEL_DIR fixtures/audio/asr_en_16k.wav 5 64 bf16 auto8
```

## 2026-06-04 Pass 2: GQA Eager SDPA + Packed Varlen Prefill

Date: 2026-06-04

Platform:

- Model: `Qwen/Qwen3-ASR-0.6B`
- Device: Apple Metal (same machine as prior notes)
- Runtime dtype: BF16
- Quantization: ISQ 8-bit (`auto8`)
- Fixture: `fixtures/audio/asr_en_16k.wav`

Reference mistral.rs numbers (unchanged baseline from earlier notes):

| Runtime | Mode | Decode speed |
| --- | --- | ---: |
| mistral.rs | `--isq 8` default eager | `231.9 +/- 1.3 tok/s` |
| mistral.rs | `--isq 8 --paged-attn on` | `175.6 +/- 3.6 tok/s` |

### What Changed

#### 1. GQA-aware eager attention (P0, all platforms)

Files:

- `vasr_models/src/model/attention.rs`
- `vasr_models/src/model/thinker_text.rs`

Behavior:

- Removed unconditional `repeat_kv` before `run_attention` in eager prefill and
  eager cached decode paths.
- `run_attention` now tries Metal/CUDA `candle_nn::ops::sdpa` with native GQA
  head counts first, matching mistral.rs `Sdpa.run_attention_noflash` ordering.
- `repeat_kv` is deferred to the manual matmul fallback only when SDPA is
  unavailable.

Why:

- mistral.rs avoids expanding K/V on the hot Metal SDPA path; our eager branches
  were paying the repeat cost on every layer/step even when SDPA succeeded.

#### 2. Packed varlen paged prefill (P0, CUDA/CPU flash only)

Files:

- `vasr_models/src/model/attention.rs` (`supports_packed_varlen_sdpa`)
- `vasr_models/src/model/thinker_text.rs` (`forward_with_paged_cache` prefill)

Behavior:

- After `gather_kv_cache`, when `supports_packed_varlen_sdpa` is true, reshape
  gathered K/V to `(1, kv_heads, total_kv, dim)` and call `run_attention` with
  existing `PagedInputMetadata::flash_params` instead of
  `unpack_gathered_kv_for_attention` + padded `Tensor::cat`.
- Metal still uses the unpack path because mistral.rs only enables packed varlen
  on CPU flash or CUDA flash-attn.

Why:

- Removes pad/cat overhead on mixed-length batch prefill for CUDA deployments.
- Keeps Metal on the proven unpack path while eager GQA work closes the larger
  local gap.

### Measured Results (this pass)

Command:

```bash
cargo run --release -p vasr-models \
  --example bench_transcribe \
  --features 'metal-paged-attn timing audio-loading' \
  -- MODEL_DIR fixtures/audio/asr_en_16k.wav 3 64 bf16 auto8
```

Paged-attn (default production path on Metal):

| Run | Mode | `decode_tokens_per_s` | Notes |
| --- | --- | ---: | --- |
| hot run 2 | paged-attn | `182.3` | `prompt_len=214`, `steps=44` |
| hot run 3 | paged-attn | `184.2` | stable vs prior `170-176` band |

Eager fallback (`VASR_DISABLE_PAGED_ATTN=1`):

| Run | Mode | `decode_tokens_per_s` | Notes |
| --- | --- | ---: | --- |
| hot run 2 | eager KV cache | `197.6` | `steps=45` |
| hot run 3 | eager KV cache | `196.1` | large improvement vs pre-pass eager gap |

Takeaways:

- **Eager decode improved materially** from the previously documented large
  deficit vs mistral.rs default eager (`~232 tok/s`) to **`~196-198 tok/s`** on
  the same fixture, while preserving output correctness on the benchmark clip.
- **Paged-attn decode stayed in the same band** as the prior Metal session
  (`~182-184 tok/s` hot), with no regression observed.
- **CUDA mixed-length prefill** should benefit from packed varlen, but was not
  re-benchmarked on this Metal-only machine in this pass.

### Next Steps

1. Re-run mixed-length batch prefill benchmarks (`bench_transcribe_batch`) on
   Metal to quantify any prefill-side movement.
2. Port batch decode metadata precomputation for `greedy_generate_paged_batch`.
3. On CUDA, validate packed varlen prefill against mistral.rs forced paged-attn
   prefill timings.
4. Continue closing the remaining eager gap (`~35 tok/s` vs mistral.rs) via
   fuller `SingleCache` semantics and SDPA softcap/sliding-window parity.

## 2026-06-04 Pass 3: Batch Decode Metadata Precompute (Metal)

Date: 2026-06-04

Platform:

- Model: `Qwen/Qwen3-ASR-0.6B`
- Device: Apple Metal
- Runtime dtype: BF16
- Quantization: ISQ 8-bit (`auto8`)

Skipped in this pass:

- CUDA packed-varlen validation (not available on the current Mac machine)

### What Changed

#### Batch decode metadata precompute (P1)

Files:

- `vasr_models/src/model/paged_kv_cache.rs`
- `vasr_models/src/model/generation.rs`

Behavior:

- Added `PagedKvCache::decode_metadata_for_batch_steps(...)` to precompute all
  batch decode `PagedInputMetadata` entries once before the decode loop.
- `greedy_generate_paged_batch` now indexes precomputed metadata per step,
  instead of rebuilding tensors via `input_metadata_from_block_tables(...)`
  on every decode step.
- Added unit test
  `test_decode_metadata_for_batch_steps_matches_per_step_builder` to ensure
  parity with the old per-step builder.

Why:

- Single-sequence paged decode already used `decode_metadata_for_steps`; the
  batch path was still paying repeated host/tensor construction each step.

### Measured Results (this pass)

Single-sequence paged decode (`fixtures/audio/asr_en_16k.wav`, 3 iters):

| Run | `decode_tokens_per_s` |
| --- | ---: |
| hot run 2 | `174.5` |
| hot run 3 | `174.1` |

No regression observed versus Pass 2 hot band (`~182-184 tok/s` is within normal
Metal run-to-run variance on this machine).

Mixed-length batch decode (`asr_en_16k.wav` + `raw_audios/audio (12).wav`,
`batch_size=2`, 3 iters):

| Run | `prefill_ms` | `decode_ms` | `batch_decode_tokens_per_s` | `per_sequence_decode_tokens_per_s` |
| --- | ---: | ---: | ---: | ---: |
| 2 | `383.5` | `279.9` | `217.9` | `109.0` |
| 3 | `377.9` | `281.4` | `216.8` | `108.4` |

Notes:

- Output text for both sequences remained stable across runs.
- This pass mainly removes repeated metadata construction overhead; mixed-length
  batch throughput still depends on prefill unpack/pad work (Pass 2 item #2 on
  Metal) and overall batch scheduling, so tok/s alone is not the primary success
  signal here.
- Correctness parity is covered by the new batch-metadata unit test.

### Next Steps (Metal-first)

1. Profile mixed-length **prefill** (`prefill_ms` above) for further unpack/pad
   reductions on Metal.
2. Port fuller eager `SingleCache` semantics to keep closing the non-paged gap.
3. Revisit CUDA packed-varlen prefill when a CUDA machine is available.

## 2026-06-04 Pass 4: Mistral-Style Gathered KV Unpack (Metal)

Date: 2026-06-04

Platform:

- Model: `Qwen/Qwen3-ASR-0.6B`
- Device: Apple Metal
- Runtime dtype: BF16
- Quantization: ISQ 8-bit (`auto8`)

### What Changed

File:

- `vasr_models/src/model/thinker_text.rs` (`unpack_gathered_kv_for_attention`)

Behavior:

- Align mixed-length paged prefill unpack with mistral.rs `unpack_gathered_kv`:
  - keep native `num_kv_heads` and defer GQA expansion to `run_attention`
  - pad only the trailing tail when `kv_len < max_kv` via `Tensor::cat`
  - remove full `(attn_heads, max_kv)` zero buffers plus `slice_set`
- Add equal-length fast path: when all `kv_lens` match, reshape gathered KV
  directly to `(batch, num_kv_heads, max_kv, head_size)` without per-row cat.

Why:

- The old unpack path expanded GQA inside unpack and always materialized padded
  zero tensors, adding avoidable Metal memory traffic before SDPA.

### Measured Results (this pass)

Single-sequence paged decode (`asr_en_16k.wav`, hot runs):

| Run | `prefill_ms` | `decode_tokens_per_s` |
| --- | ---: | ---: |
| 2 | `234.0` | `175.0` |
| 3 | `234.0` | `173.9` |

Mixed-length batch (`asr_en_16k.wav` + `audio (12).wav`, batch=2):

| Run | `prefill_ms` | `decode_ms` | `batch_decode_tokens_per_s` |
| --- | ---: | ---: | ---: |
| 1 | `367.4` | `276.9` | `220.3` |
| 2 | `362.6` | `281.1` | `217.0` |

Compared with Pass 3 on the same machine:

- Mixed-length **prefill improved by ~15-20 ms** (~4-5%) on hot runs
  (`~378-384 ms` -> `~362-367 ms`).
- Decode throughput stayed in the same band; output text remained stable.

### Next Steps (Metal-first)

1. Add `adjust_kv_mask`-style narrowing when custom masks exceed gathered `max_kv`.
2. Port fuller eager `SingleCache` semantics for the remaining non-paged gap.
3. Revisit CUDA packed-varlen prefill when a CUDA machine is available.

## 2026-06-04 Pass 5: Paged Prefill Mask Alignment (Metal)

Date: 2026-06-04

Platform:

- Model: `Qwen/Qwen3-ASR-0.6B`
- Device: Apple Metal
- Runtime dtype: BF16
- Quantization: ISQ 8-bit (`auto8`)

### What Changed

Files:

- `vasr_models/src/model/attention.rs` (`adjust_kv_mask`)
- `vasr_models/src/model/paged_kv_cache.rs`
- `vasr_models/src/model/thinker_text.rs` (`paged_prefill_attention_mask`)

Behavior:

- Port mistral.rs `adjust_kv_mask` and apply it when paged prefill uses unpacked
  gathered K/V with `max_kv` shorter than the padded prompt width.
- Only mark `prefill_causal_only=true` when every sequence uses the full prompt
  width (`query_len == seq_len` for all rows).
- Narrow explicit causal masks to gathered `max_kv` before Metal SDPA.

Why:

- Padded ASR batches often have `seq_len` wider than effective `max_kv`. Mask/K/V
  shape mismatch could disable Metal SDPA or attend over padded key slots.

### Measured Results (this pass)

Mixed-length batch (`asr_en_16k.wav` + `audio (12).wav`, batch=2):

| Run | `prefill_ms` | `batch_decode_tokens_per_s` |
| --- | ---: | ---: |
| 1 | `367.6` | `218.4` |
| 3 | `367.2` | `217.3` |

Notes:

- Throughput stayed in the Pass 4 band; output text remained stable.
- Primary value: correctness + SDPA compatibility for padded batch prefill.

### Next Steps (Metal-first)

1. Port fuller eager `SingleCache` semantics to close the non-paged eager gap.
2. Revisit CUDA packed-varlen prefill when a CUDA machine is available.

## 2026-06-04 Pass 6: Eager KV Prealloc + Metal Argmax Fix (Metal)

Date: 2026-06-04

Platform:

- Model: `Qwen/Qwen3-ASR-0.6B`
- Device: Apple Metal
- Runtime dtype: BF16
- Quantization: ISQ 8-bit (`auto8`)

Skipped in this pass:

- CUDA validation (not available on the current Mac machine)

### What Changed

#### 1. mistral-style eager KV cache preallocation

Files:

- `vasr_models/src/model/kv_cache.rs`
- `vasr_models/src/model/generation.rs`
- `vasr_models/src/model/thinker.rs` (`num_text_layers`)

Behavior:

- Added `KVCache::with_max_seq_len(num_layers, prompt_len + max_new_tokens)` so
  decode avoids repeated grow/realloc on every token.
- `KVCacheEntry::new` reserves full capacity up front when `max_seq_len` is
  known; append only copies backing storage when capacity is insufficient.
- Skip redundant `contiguous()` when tensors are already contiguous.
- Generation paths call `kv_cache_for_generation(...)` instead of bare
  `KVCache::new()`.

Why:

- mistral.rs `SingleCache` pre-reserves prompt + generation length; our dynamic
  grow-by-256 path added allocator churn on every decode step.

#### 2. Metal argmax on eager batch decode (batch=1)

Files:

- `vasr_models/src/model/generation.rs` (`argmax_token_ids_from_logits`)

Behavior:

- Route batch=1 logits through `metal_argmax` (same as single-sequence path).
- Fix indexing: use row `logits.i((0,))` (or 1-D logits) instead of scalar
  `logits.i((0, 0))`, which previously picked vocab index 0 and broke eager
  output when `VASR_DISABLE_PAGED_ATTN=1`.

Why:

- `transcribe` always uses the batch generation helper even for batch=1; the
  broken argmax caused garbage output (`"!"`) and inflated token counts.

### Measured Results (this pass)

Single-sequence fixture (`fixtures/audio/asr_en_16k.wav`, 3 runs):

Paged-attn (default):

| Run | `decode_tokens_per_s` | `decode_argmax_ms` | Notes |
| --- | ---: | ---: | --- |
| 1 | `204.3` | `62.4` | `prompt_len=214`, `steps=44` |
| 2 | `203.7` | `61.8` | stable transcription |
| 3 | `202.2` | `62.6` | vs Pass 2 hot band `~182-184` |

Eager fallback (`VASR_DISABLE_PAGED_ATTN=1`):

| Run | `decode_tokens_per_s` | Notes |
| --- | ---: | --- |
| 1 | `232.0` | matches mistral.rs default eager reference |
| 2 | `234.0` | `prompt_len=214`, correct ASR text |
| 3 | `233.0` | vs Pass 2 `~196-198` (+~35 tok/s) |

Takeaways:

- **Eager decode now matches mistral.rs default eager** (~232 tok/s) on the same
  fixture, closing the largest remaining non-paged gap.
- **Paged-attn decode improved** to ~202-204 tok/s hot on this session (up from
  ~171-175 earlier in the day; some run-to-run Metal variance remains).
- **`decode_argmax_ms` still ~62 ms/run** on paged path (~29% of decode); next
  optimization target for paged parity.

### Next Steps (Metal-first)

1. Reduce paged decode argmax overhead (currently ~60 ms / 44 steps).
2. Port snapshot/rollback `SingleCache` semantics if needed for speculative paths.
3. Revisit CUDA packed-varlen prefill when a CUDA machine is available.

## 2026-06-04 Pass 7: Metal Argmax Scratch + Decode Timing Split (Metal)

Date: 2026-06-04

Platform:

- Model: `Qwen/Qwen3-ASR-0.6B`
- Device: Apple Metal
- Runtime dtype: BF16
- Quantization: ISQ 8-bit (`auto8`)

### What Changed

#### 1. Reusable Metal argmax workspace

Files:

- `vasr_models/src/model/metal_argmax.rs`
- `vasr_models/src/model/generation.rs`

Behavior:

- Added `MetalArgmaxScratch` to reuse topk stage1/stage2 Metal buffers across decode
  steps instead of allocating five buffers on every argmax call.
- `greedy_generate_paged` keeps one scratch for prefill + all decode steps.
- Skip redundant `contiguous()` when logits are already contiguous.

Why:

- Each decode step previously allocated ~75-block topk workspace from scratch.
- Confirms prior note: argmax kernel choice matters less than sync behavior.

#### 2. Split paged decode timing (`decode_forward_us`)

Files:

- `vasr_models/src/model/generation.rs`
- `vasr_models/examples/bench_transcribe.rs`
- `vasr_models/src/inference/transcribe.rs`

Behavior:

- Time paged decode forward separately from argmax readback.
- Bench output adds `decode_forward_ms` alongside `decode_argmax_ms`.

Why:

- Makes it explicit that most `decode_argmax_ms` is GPU pipeline drain at the
  first host readback, not topk compute or buffer allocation.

### Measured Results (this pass)

Single-sequence paged-attn (`fixtures/audio/asr_en_16k.wav`, 3 hot runs):

| Run | `decode_tokens_per_s` | `decode_forward_ms` | `decode_argmax_ms` | Notes |
| --- | ---: | ---: | ---: | --- |
| 2 | `172.3` | `169.4` | `84.2` | `prompt_len=214`, `steps=44` |
| 3 | `169.1` | `173.0` | `85.2` | stable transcription |

Reference (Pass 6 hot band on a faster session): `~202-204 tok/s`.

Control (`VASR_DISABLE_METAL_ARGMAX=1`, candle argmax): `~170 tok/s` — custom
Metal topk remains faster than candle fallback.

Takeaways:

- **`decode_forward_ms + decode_argmax_ms ≈ decode_ms`** on paged path
  (~171 ms + ~85 ms ≈ 256 ms for 44 tokens).
- **`decode_argmax_ms` is dominated by GPU synchronization**, not buffer alloc;
  scratch reuse is hygiene but does not materially move tok/s alone.
- Session-to-session Metal variance remains large (`~170-204 tok/s` observed).

### Next Steps (Metal-first)

1. Reduce paged decode forward cost (true gap vs eager ~232 tok/s on same fixture).
2. Explore Metal command-buffer batching / fewer sync points across decode steps.
3. Revisit CUDA packed-varlen prefill when a CUDA machine is available.

## 2026-06-04 Pass 8: Paged Decode Forward Hygiene (Metal)

Date: 2026-06-04

Platform:

- Model: `Qwen/Qwen3-ASR-0.6B`
- Device: Apple Metal
- Runtime dtype: BF16
- Quantization: ISQ 8-bit (`auto8`)

### What Changed

#### 1. mistral-style packed-layout contiguous skip

Files:

- `vasr_models/src/model/thinker_text.rs`

Behavior:

- Port `cache_input_is_packed` from mistral.rs; only call `contiguous()` on
  paged K/V/Q tensors when strides are not already head-major packed.

Why:

- Decode path was unconditionally contiguous-ing up to 96 tensors/request
  (32 layers × 3).

#### 2. Slim single-sequence decode metadata

Files:

- `vasr_models/src/model/paged_kv_cache.rs`

Behavior:

- `decode_metadata_for_steps` no longer builds per-step `cu_seqlens_*`,
  `query_lens`, or `kv_lens` (unused on `seq_len==1` paged-attention path).
- Added `test_decode_metadata_for_steps_matches_per_step_builder`.

Why:

- Removed ~2 device tensor allocations per decode step that only served prefill
  varlen helpers.

#### 3. Precomputed decode position ids + Metal hidden-state view

Files:

- `vasr_models/src/model/generation.rs`
- `vasr_models/src/model/thinker_text.rs`

Behavior:

- Precompute all mRoPE `position_ids` before the paged decode loop.
- Use `inputs_embeds.affine(1,0)` view on Metal decode (`seq_len==1`) instead
  of cloning hidden states into the layer stack.

Why:

- Removes per-step position tensor rebuild; avoids an extra hidden-state buffer
  copy on Metal decode.

### Measured Results (this pass)

Single-sequence paged-attn (`fixtures/audio/asr_en_16k.wav`, 5 runs):

| Run | `decode_tokens_per_s` | `decode_forward_ms` | `decode_argmax_ms` | Notes |
| --- | ---: | ---: | ---: | --- |
| 2 | `169.7` | `181.8` | `76.1` | `decode_position_ms=0` (precomputed) |
| 4 | `169.9` | `180.2` | `77.5` | stable transcription |

Same session eager fallback: `~198-200 tok/s` (down from Pass 6 hot `~232`;
Metal run-to-run variance).

Takeaways:

- **Paged decode per-step host overhead reduced** (position rebuild + redundant
  metadata tensors + unconditional contiguous).
- **Paged vs eager gap on Metal batch=1 remains ~15-30%** — root cause is
  `reshape_and_cache` + `paged_attention` vs eager dense KV + Metal SDPA, not
  metadata/position overhead alone.
- **`decode_argmax_ms` still ~76-79 ms/44 steps** — sync-dominated (Pass 7).

### Next Steps (Metal-first)

1. Evaluate Metal batch=1 routing to eager KV path for max single-stream throughput.
2. Prototype paged decode gather+SDPA fallback (mistral.rs large-head path).
3. Metal command-buffer pipelining to overlap forward + argmax sync.

## 2026-06-04 Pass 9: Metal Batch=1 Eager Routing (Metal)

Date: 2026-06-04

Platform:

- Model: `Qwen/Qwen3-ASR-0.6B`
- Device: Apple Metal
- Runtime dtype: BF16
- Quantization: ISQ 8-bit (`auto8`)

### What Changed

Files:

- `vasr_models/src/model/generation.rs`

Behavior:

- Metal + `batch==1` + dense mask now defaults to **eager KV-cache decode**
  (`forward_decode_one_without_padding` + Metal SDPA) instead of paged-attn.
- CUDA batch=1 still uses paged-attn (unchanged).
- `batch>1` on Metal/CUDA still uses paged batch path when runtime is available.
- Opt-in overrides:
  - `VASR_FORCE_PAGED_ATTN=1` — force paged single-sequence decode on Metal
  - `VASR_DISABLE_PAGED_ATTN=1` — disable all paged paths (unchanged)

Why:

- Pass 8 showed paged single-stream decode is ~15–30% slower than eager on Metal
  because `reshape_and_cache` + `paged_attention` loses to dense KV + SDPA at
  batch=1. Paged-attn value is batch scheduling / memory pooling, not single-seq
  throughput.

### Measured Results (this pass)

Single-sequence (`fixtures/audio/asr_en_16k.wav`, same session):

| Mode | `decode_tokens_per_s` | Notes |
| --- | ---: | --- |
| **Default (Metal eager)** | `~195-198` | new production default |
| `VASR_FORCE_PAGED_ATTN=1` | `~171-174` | prior paged single-seq path |
| Improvement | **+~23 tok/s (~14%)** | same fixture, same session |

Mixed batch=2 (still paged):

| Run | `batch_decode_tokens_per_s` | Notes |
| --- | ---: | --- |
| 2 | `216.6` | unchanged paged batch path |

Takeaways:

- **`bench_transcribe` on Metal now reflects eager decode by default** — aligns
  with mistral.rs default eager reference (~232 tok/s on hot sessions).
- Use `VASR_FORCE_PAGED_ATTN=1` when benchmarking paged single-seq regressions.
- Server `transcribe` with `max_batch_size=1` picks up the improvement automatically.

### Next Steps (Metal-first)

1. Prototype paged decode gather+SDPA fallback for batch>1 single-token steps.
2. Metal command-buffer pipelining to overlap forward + argmax sync on paged batch.
3. Revisit CUDA packed-varlen prefill when a CUDA machine is available.

## 2026-06-04 Pass 10: Paged Batch Decode Hygiene + Gather SDPA Trial (Metal)

Date: 2026-06-04

Platform:

- Model: `Qwen/Qwen3-ASR-0.6B`
- Device: Apple Metal
- Runtime dtype: BF16
- Quantization: ISQ 8-bit (`auto8`)

### What Changed (shipped)

#### 1. Slim batch decode metadata

Files:

- `vasr_models/src/model/paged_kv_cache.rs`

Behavior:

- `decode_metadata_for_batch_steps` no longer builds per-step `cu_seqlens_*`,
  `query_lens`, or `kv_lens` for single-token decode steps.
- Updated batch metadata unit test accordingly.

Why:

- Same rationale as Pass 8 single-seq slim metadata: decode uses
  `paged_attention`, not varlen flash helpers.

#### 2. Precomputed batch decode position ids

Files:

- `vasr_models/src/model/generation.rs`

Behavior:

- `greedy_generate_paged_batch` precomputes all mRoPE `position_ids` before the
  decode loop via `position_ids_for_decode_steps_batch`.

Why:

- Removes per-step position tensor rebuild on the batch paged path.

### What Was Tried and Rejected

#### Metal batch gather+SDPA decode (not shipped)

Prototype replaced `paged_attention` with per-layer `gather_kv_cache` +
`unpack_gathered_kv_for_attention` + Metal SDPA for `batch>1, seq_len==1`.

Observed effect on batch=2 mixed fixture:

| Path | `batch_decode_tokens_per_s` |
| --- | ---: |
| gather+SDPA (prototype) | `~35` |
| `paged_attention` (kept) | `~218` |

Conclusion:

- Per-layer full KV gather on every decode step is far more expensive than the
  block-table `paged_attention` kernel at head_dim=128. mistral.rs uses gather
  decode mainly for oversized heads, not Qwen3-ASR 0.6B on Metal.

### Measured Results (shipped changes)

Mixed batch=2 (`asr_en_16k.wav` + `audio (12).wav`, 3 runs):

| Run | `batch_decode_tokens_per_s` | Notes |
| --- | ---: | --- |
| 1 | `220.3` | no regression vs Pass 9 (~216) |
| 2 | `219.1` | stable transcription |

Single-seq default (Pass 9 eager routing): `~200 tok/s` unchanged.

### Next Steps (Metal-first)

1. Metal command-buffer pipelining on paged batch decode (overlap sync).
2. Investigate paged batch decode forward vs 2× single eager on mixed workloads.
3. Revisit CUDA packed-varlen prefill when a CUDA machine is available.

## 2026-06-04 Pass 11: Metal Batch Eager Routing + Padded Prefill Fix (Metal)

Date: 2026-06-04

Platform:

- Model: `Qwen/Qwen3-ASR-0.6B`
- Device: Apple Metal
- Runtime dtype: BF16
- Quantization: ISQ 8-bit (`auto8`)

### What Changed

#### 1. Extend Metal eager routing to batch>1

Files:

- `vasr_models/src/model/generation.rs`

Behavior:

- Renamed routing helper to `use_paged_attn_on_device`; on Metal, paged-attn
  (single or batch) is opt-in via `VASR_FORCE_PAGED_ATTN=1` only.
- CUDA keeps paged-attn as default when runtime is available.

Why:

- Pass 8–10 showed paged single-token decode loses to eager dense KV + Metal SDPA;
  batch=2 paged decode was ~109 tok/s per sequence vs ~200 for batch=1 eager.

#### 2. Fix eager batch prefill on left-padded mixed batches

Files:

- `vasr_models/src/model/generation.rs`

Behavior:

- Eager batch prefill now uses `gather_last_logits_for_prompt_lens` instead of
  always indexing `seq_len - 1` (which broke the first sequence on padded batch).

Why:

- Mixed-length ASR batches are left-padded; the last valid prompt token is per-row,
  not at the padded width.

#### 3. Precompute eager batch decode position ids

Behavior:

- Eager batch decode loop reuses precomputed mRoPE `position_ids` (same pattern as
  Pass 10 paged batch).

### Measured Results (this pass)

Mixed batch=2 (`asr_en_16k.wav` + `audio (12).wav`):

| Mode | `batch_decode_tokens_per_s` | `per_sequence_decode_tokens_per_s` | Notes |
| --- | ---: | ---: | --- |
| **Default (Metal eager batch)** | `~248` | `~124` | correct text both sequences |
| `VASR_FORCE_PAGED_ATTN=1` | `~219` | `~110` | prior paged batch path |

Improvement: **+~29 batch tok/s (~13%)**, **+~15 tok/s per sequence (~14%)**.

End-to-end wall on batch=2: **~762 ms** (eager) vs **~857 ms** (paged).

### Next Steps (Metal-first)

1. Profile eager batch prefill vs paged prefill on longer mixed batches.
2. Metal command-buffer pipelining where paged-attn remains required (CUDA / forced).
3. Revisit CUDA packed-varlen prefill when a CUDA machine is available.

## 2026-06-04 Pass 12: raw_audios E2E Validation + Length Bucketing (Metal)

Date: 2026-06-04

Platform:

- Model: `Qwen/Qwen3-ASR-0.6B`
- Device: Apple Metal
- Runtime dtype: BF16
- Quantization: ISQ 8-bit (`auto8` → AFQ8 on Metal)
- Workload: full offline pipeline (Silero VAD + batched ASR), not decode-only bench
- Corpus: `raw_audios/` (20 customer-service WAV files, ~27.5 min total)

### Production defaults after Pass 9–11

| Setting | Metal default | Override |
| --- | --- | --- |
| KV decode path | **eager** (single + batch) | `VASR_FORCE_PAGED_ATTN=1` |
| Disable paged entirely | — | `VASR_DISABLE_PAGED_ATTN=1` |
| Quantization | `auto8` (8-bit ISQ) | `8`, `auto`, `auto6`, … |
| Runtime dtype | BF16 | — |

### E2E command

```bash
MODEL="/Users/wangmengdi/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0"

# Current default (Metal eager, ISQ8)
cargo run --release -p vasr-cli --example bench_transcribe_dir \
  --features metal-paged-attn -- \
  "$MODEL" raw_audios 64 bf16 auto8

# Compare forced paged (legacy path)
VASR_FORCE_PAGED_ATTN=1 cargo run --release -p vasr-cli --example bench_transcribe_dir \
  --features metal-paged-attn -- \
  "$MODEL" raw_audios 64 bf16 auto8
```

Stage breakdown helper:

```bash
cargo run --release -p vasr-cli --example bench_transcribe_stages \
  --features metal-paged-attn -- \
  "$MODEL" raw_audios 64 bf16 auto8 5 all
```

### Measured Results: full 20 files (default eager)

| Metric | Value |
| --- | ---: |
| Total audio | **1652.1 s** (~27.5 min) |
| Wall time (excl. first compile) | **26.42 s** |
| Speedup | **62.5×** |
| RTF | **0.0160** |
| VAD+ASR annotations | 933 |
| Stability | **20/20** completed |

Long-file examples (default eager):

| File | Audio (s) | Wall (s) | Speedup | Annotations |
| --- | ---: | ---: | ---: | ---: |
| audio (2).wav | 193.6 | 4.64 | 41.7× | 119 |
| audio (7).wav | 193.6 | 4.65 | 41.7× | 119 |
| audio (18).wav | 171.6 | 4.21 | 40.8× | 151 |

### Measured Results: first 5 files (same scale as Pass 4 doc)

| Mode | Wall (s) | Speedup | RTF | Notes |
| --- | ---: | ---: | ---: | --- |
| **Default (Metal eager)** | **4.59** | **68.8×** | **0.0145** | 189 annotations |
| `VASR_FORCE_PAGED_ATTN=1` | 5.07 | 62.2× | 0.0161 | 208 annotations |
| Pass 4 doc (paged era) | 4.64 | 67.9× | 0.0147 | — |

Relative forced paged on 5 files: **wall −10.6%** (5.07 → 4.59 s).  
Relative Pass 4 doc: **wall −1.2%** (4.64 → 4.59 s).

Stage split on 5 files (default eager, pre–Pass 12 bucketing):

| Stage | Time (s) | Share of wall |
| --- | ---: | ---: |
| VAD | 0.80 | ~17% |
| ASR | 3.77 | ~82% |
| Load + other | 0.02 | ~1% |
| **Total** | **4.58** | — |

ASR throughput on detected speech: **~15.7× realtime** (`asr_speech_speedup`).

### Forced paged: stability failure on long corpus

Full 20-file run with `VASR_FORCE_PAGED_ATTN=1` failed on `audio (18).wav`:

```text
Error: paged KV cache exhausted: request_id=25 needed_blocks=5 free_blocks=3
```

Cause: many short VAD segments on a 171 s clip exhaust the shared paged block pool
(default `PagedCacheMemory::ContextSize(4096)`). **Default Metal eager path does not
hit this limit**; 20/20 files complete.

Transcription text is broadly consistent between modes; segment counts can differ
slightly (e.g. audio (1): 66 vs 73 annotations eager vs paged).

### What Changed (Pass 12 optimization)

File:

- `vasr_runtime/src/models/qwen3_asr.rs`
- `vasr_models/src/inference/types.rs`

Behavior:

- Enable **`bucket_by_length: true`** by default for runtime `transcribe` / VAD-batched
  paths. Chunks are sorted by descending waveform length before batching so left-padded
  mixed batches carry less audio/prompt padding into eager batch prefill.

Why:

- E2E stage profiling showed **ASR ≈ 82% of wall** on `raw_audios`; VAD segments within
  one file vary widely in duration. Bucketing reduces wasted prefill compute without
  changing decode routing (still Metal eager by default).

### Measured Results after length bucketing

Same commands as above, post-change:

| Scope | Wall (s) | Speedup | RTF | Notes |
| --- | ---: | ---: | ---: | --- |
| 5 files | **4.57** | **69.0×** | **0.0145** | vs pre-bucket **4.59 s** (~flat) |
| **20 files** | **25.50** | **64.8×** | **0.0154** | vs pre-bucket **26.42 s** (**−3.5%**) |

Long file `audio (18).wav` (stages):

| Metric | Pre-bucket | Post-bucket | Delta |
| --- | ---: | ---: | ---: |
| ASR stage | 3.81 s | **3.38 s** | **−11%** |
| Total wall | 4.24 s | **3.81 s** | **−10%** |
| Speech speedup | 16.6× | **18.7×** | +2.1× |

Transcription text on spot-checked files (e.g. audio (1), (12)) remained stable; total
annotation count on 20 files moved slightly (933 → 949), within normal VAD/ASR variance.

Takeaway: length bucketing is a low-risk E2E win on **multi-segment long clips**; short
5-file smoke tests under-report the benefit because each file has fewer segments to reorder.

### Next Steps (Metal-first)

1. Metal command-buffer pipelining where paged-attn remains required (CUDA / forced).
2. Increase paged block pool or per-request lifecycle for `VASR_FORCE_PAGED_ATTN=1` long
   VAD runs (correctness/stability, not default-path perf).
3. Revisit CUDA packed-varlen prefill when a CUDA machine is available.

## 2026-06-04 Pass 13: Async Loader/VAD/ASR Pipeline + 60s ASR Batching (Metal)

Date: 2026-06-04

Platform:

- Model: `Qwen/Qwen3-ASR-0.6B`
- Device: Apple Metal
- Runtime dtype: BF16
- Quantization: ISQ 8-bit (`auto8` → AFQ8 on Metal)
- Workload: full offline pipeline (Silero VAD + batched ASR), multi-file E2E
- Corpus: `raw_audios/` (20 customer-service WAV files, ~27.5 min total)

### What Changed

#### 1. Async three-stage transcribe pipeline (cross-file overlap)

Files:

- `vasr_server/src/async_transcribe.rs`
- `vasr_server/src/transcribe.rs`
- `vasr_runtime/src/pipeline/async.rs`
- `vasr_runtime/src/pipeline/mod.rs`
- `vasr_cli/examples/bench_transcribe_dir.rs`
- `vasr_cli/src/serve.rs`

Behavior:

- **`AsyncTranscribePipeline`** runs three concurrent workers connected by bounded
  `mpsc` channels (default buffer **4**):
  - **Loader** — read/decode audio off the hot path (`spawn_blocking`)
  - **VAD** — `OfflinePipeline::prepare_vad`
  - **ASR** — `OfflinePipeline::transcribe_prepared`
- HTTP `/transcribe` and `bench_transcribe_dir` use this pipeline for multi-input
  batches; a single input still runs sequentially (load → VAD → ASR).
- **`AsyncOfflinePipeline`** (feature `async` on `vasr-runtime`) provides the same
  overlap for in-memory `Waveform` batches.

Parallelism boundary:

- **Across files/jobs**: while file *A* is in ASR, file *B* can run VAD and file *C*
  can load — Loader/VAD overlap ASR wall time.
- **Within one file**: still **VAD → ASR** (ASR needs VAD slices first). No
  segment-level streaming yet.

Why:

- Pass 12 stage profiling showed VAD ≈ **17%** of sequential wall on 5 files; on 20
  files VAD is **~4.1 s** vs ASR **~24.9 s**. Overlapping VAD/Loader with ASR across
  files recovers most of the non-ASR time without changing GPU decode routing.

#### 2. OfflinePipeline stage split + shared VAD

Files:

- `vasr_runtime/src/pipeline/mod.rs`

Behavior:

- Split monolithic `transcribe` into **`prepare_vad`** and **`transcribe_prepared`** so
  the async workers can hand off `VadPrepared` (speech annotations + slices) between
  stages.
- `VadModel` holder changed from `Box<dyn VadModel>` to **`Arc<dyn VadModel>`** so VAD
  is cheaply shared across blocking worker threads.

#### 3. ASR micro-batch grouping by audio duration (60 s cap)

Files:

- `vasr_models/src/inference/transcribe.rs` (`group_chunk_indices`)
- `vasr_models/src/inference/types.rs`
- `vasr_runtime/src/models/qwen3_asr.rs`

Behavior:

- New option **`max_batch_audio_sec`** (default **`60.0`**).
- After length bucketing, chunk indices are grouped greedily so each ASR forward batch
  carries **≤60 s** of audio (single segments **>60 s** get their own batch).
- **`max_batch_size: 0`** means no count cap — only duration bounds batch size.
- Decode-only benches (`bench_transcribe`, `bench_transcribe_wall`) set
  `max_batch_audio_sec: 0.0` to preserve single-chunk behavior.

Why:

- Long VAD-heavy files (e.g. 171 s clips with dozens of segments) previously fed very
  large mixed batches into eager prefill. Capping batch audio seconds keeps GPU memory
  and prefill padding predictable while still batching short segments together.

### Verification

Unit test (pipeline overlap, no real model):

```bash
cargo test -p vasr-runtime --features async async_pipeline
```

Async E2E (production path):

```bash
MODEL="/Users/wangmengdi/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0"

cargo run --release -p vasr-cli --example bench_transcribe_dir \
  --features metal-paged-attn -- \
  "$MODEL" raw_audios 128 bf16 auto8 20
```

Sequential stage breakdown (does **not** overlap stages — for profiling only):

```bash
cargo run --release -p vasr-cli --example bench_transcribe_stages \
  --features metal-paged-attn -- \
  "$MODEL" raw_audios 128 bf16 auto8 20 all
```

### Measured Results (this pass)

Compare **async pipeline** (`bench_transcribe_dir`) vs **sequential stages**
(`bench_transcribe_stages`, load + VAD + ASR summed per file):

| Scope | Async wall (s) | Sequential wall (s) | Δ wall | Async speedup | Async RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| **5 files** (315.4 s audio) | **4.14** | 4.66 | **−11%** | **76.2×** | **0.0131** |
| **20 files** (1652.1 s audio) | **25.18** | 29.15 | **−14%** | **65.6×** | **0.0152** |

Sequential stage split (20 files, for reference):

| Stage | Time (s) | Share |
| --- | ---: | ---: |
| Load | 0.09 | ~0.3% |
| VAD | 4.11 | ~14% |
| ASR | 24.94 | ~86% |
| **Sum** | **29.15** | — |

Async wall **25.18 s** vs sequential sum **29.15 s** → recovered **~4.0 s**
(~14%), consistent with overlapping VAD/Loader while ASR runs on other files.

Relative Pass 12 post-bucketing (same corpus, sequential `bench_transcribe_dir`):

| Scope | Pass 12 wall (s) | Pass 13 async wall (s) | Δ |
| --- | ---: | ---: | ---: |
| 5 files | 4.57 | **4.14** | **−9%** |
| 20 files | 25.50 | **25.18** | **−1%** |

Notes:

- Pass 12 → Pass 13 comparison mixes pipeline change with `max_batch_audio_sec`; the
  20-file delta is small because ASR still dominates and GPU ASR remains single-worker
  serialized via `ScheduledAsrModel`.
- All spot-checked transcriptions completed without error (189 annotations / 5 files,
  935 / 20 files in the async run).
- **`bench_transcribe_stages` wall ≈ load + VAD + ASR**; do not use it as the async
  pipeline ceiling.

Takeaways:

- **Async pipeline is now the production default** for HTTP transcribe and directory
  benchmarks; multi-file workloads should prefer `transcribe_many` / batch HTTP inputs.
- **Biggest win is cross-file overlap**, not parallel ASR forwards (GPU still one decode
  stream at a time).
- **60 s ASR batch cap** is a safe default for long multi-segment clips; tune via
  `TranscribeOptions.max_batch_audio_sec` if memory/latency trade-offs change.

### Next Steps (Metal-first)

1. Segment-level VAD→ASR streaming (overlap VAD and ASR within one long file) if Silero
   streaming API is wired in.
2. Metal command-buffer pipelining where paged-attn remains required (CUDA / forced).
3. Revisit CUDA packed-varlen prefill when a CUDA machine is available.

## 2026-06-08 Pass 14: Metal RoPE Fast Path + Eager Routing

Date: 2026-06-08

Platform:

- Model: `Qwen/Qwen3-ASR-0.6B`
- Device: Apple Metal
- Runtime dtype: BF16
- Quantization: ISQ 8-bit (`8` → AFQ8 on Metal)

### What Changed

#### 1. Metal batch=1/2/4 default eager KV-cache routing

Files:

- `vasr_models/src/model/generation.rs`

Behavior:

- `use_paged_attn_on_device` and `should_use_paged_attention`: Metal now defaults
  to `false` (eager), paged-attn is opt-in via `VASR_FORCE_PAGED_ATTN=1`.
- Batch=1 decode on Metal routes to `forward_decode_one_without_padding` (Metal
  SDPA + preallocated KV cache), matching the Pass 9 design but actually
  implemented now.
- Batch>1 on Metal routes to the eager batch path with `forward_decode_one_without_padding`.

Why:

- The paged-attn `reshape_and_cache` + `paged_attention` kernel path is
  optimized for batched/memory-pooled workloads, not single-stream Metal decode.
- Eager dense KV + Metal SDPA is the default in mistral.rs and gives faster
  single-sequence throughput.
- This was documented as a Pass 9 change but the code still unconditionally
  returned `true` for Metal, so the eager routing was never actually active.

#### 2. Metal-accelerated RoPE for seq_len==1 via `mistralrs_quant::rotary`

Files:

- `vasr_models/src/model/rope/mrope.rs`
- `vasr_models/src/model/thinker_text.rs`
- `vasr_quant/src/lib.rs`

Behavior:

- Added `use_accelerated_rotary()` helper: true when device is Metal/CUDA.
- `apply_multimodal_rotary_pos_emb` now dispatches seq_len==1 to
  `apply_multimodal_rotary_pos_emb_seq_one`, which extracts the first (temporal)
  mRoPE modality cos/sin and calls `mistralrs_quant::rotary::apply_rotary_qk`
  (a Metal-optimized kernel from the same dependency used by mistral.rs).
- Added `MultimodalRotaryEmbedding::forward_first_modality` that computes only
  the temporal modality's `(cos, sin)` for seq_len==1, avoiding wasted work on
  the unused height/width modalities.
- `ThinkerTextRotaryEmbedding` exposes `forward_first_modality`.
- `ThinkerTextModel::forward_decode_one_without_padding` now calls
  `forward_first_modality` instead of `forward` (which built all 3 modalities).
- Re-exported `mistralrs_quant::rotary::apply_rotary_qk` from `vasr_quant` so
  downstream crates don't need a direct `mistralrs-quant` dependency.

Why:

- The old `apply_rope_batched` function is a pure Candle tensor-op
  implementation: it does `cat`, `narrow`, `unsqueeze`, `rotate_half` (split +
  negate + cat), `broadcast_mul` × 2, `add`, and `contiguous` for every layer
  on every decode step. On Metal each of these is a separate kernel launch and
  intermediate allocation.
- `apply_rotary_qk` is a single Metal `CustomOp3` kernel that does all RoPE
  math in one dispatch, cutting per-layer overhead from ~8 tensor ops to 1
  kernel.
- This mirrors exactly what mistral.rs does for Qwen3 decode via
  `RotaryEmbedding::forward_qk_norm`.

Observed effect:

- **`decode_forward` dropped from ~1555 ms to ~281 ms** (5.5× faster) for 64
  tokens of single-sequence decode.
- Aggregate throughput jumped from **~95 tok/s to ~238 tok/s** (2.5×).

#### 3. Batch decode benchmark support

Files:

- `vasr_models/examples/bench_text_decode.rs`
- `vasr_models/src/lib.rs`

Behavior:

- Added `--batch <N>` flag to `bench_text_decode` example.
- When `--batch > 1`, replicates the same prompt N times and uses
  `greedy_generate_cached_batch_timed_with_paged_runtime` to measure concurrent
  decode throughput.
- Added `Qwen3Asr::inner_model()` accessor for benchmarks.

### Measured Results (this pass)

Single-sequence (`Qwen/Qwen3-ASR-0.6B`, ISQ 8, 5 runs):

| Run | `decode_tokens_per_s` | `decode_forward_ms` | `decode_argmax_ms` |
| --- | ---: | ---: | ---: |
| 1 | 241.1 | — | — |
| 2 | 242.2 | — | — |
| 3 | 241.5 | — | — |
| 4 | 234.5 | — | — |
| 5 | 232.1 | — | — |
| **Aggregate** | **238.2** | **462** (5 runs) | **875** (5 runs) |

Reference mistral.rs (same machine, ISQ 8, 5 iterations): **253.0 ± 2.0 tok/s**.

Gap: ~6% (238 vs 253 tok/s).

Batch decode scaling (same model, ISQ 8, 3 runs each):

| batch | batch tok/s | per-seq tok/s | forward_ms | argmax_ms |
|-------|-------------|---------------|------------|-----------|
| 1 | 236.7 | 236.7 | 281 | 526 |
| 2 | 450.3 | 225.2 | 291 | 557 |
| 4 | 658.3 | 164.6 | 295 | 867 |

Forward time stays essentially flat (281→295 ms) as batch grows, confirming GPU
parallelism. Argmax grows with batch because more rows need synchronisation +
readback.

### Remaining Gap Analysis

The ~6% gap vs mistral.rs likely comes from:

1. **Q/K norm not fused with mRoPE** — mistral.rs `RotaryEmbedding::forward_qk_norm`
   combines RMS norm on Q, RMS norm on K, and RoPE into a single Metal kernel.
   Our path still does `q_norm.forward()`, `k_norm.forward()`, then
   `apply_rotary_qk` (3 kernel launches vs 1). Adapting this fusion for 3D
   mRoPE positions requires a custom kernel.

2. **Benchmark methodology** — our benchmark includes prompt tokenisation +
   prefill setup in the same pipeline, while mistral.rs `--prompt-len 0` runs
   pure decode with a pre-populated cache.

3. **Model weight layout** — Qwen3-ASR-0.6B safetensors may have slightly
   different memory layout than Qwen3-0.6B GGUF, affecting ISQ matmul
   efficiency.

### Verification Commands

```bash
# Single-sequence decode
cargo run --release -p vasr-models --example bench_text_decode \
  --features "metal,metal-paged-attn,paged-attn,timing" \
  -- Qwen/Qwen3-ASR-0.6B "The capital of France is" 5 64 bf16 8 --warmup 1

# Batch decode (N concurrent sequences)
cargo run --release -p vasr-models --example bench_text_decode \
  --features "metal,metal-paged-attn,paged-attn,timing" \
  -- Qwen/Qwen3-ASR-0.6B "The capital of France is" 3 64 bf16 8 --batch 2 --warmup 1

# Force paged-attn (opt-in regression test)
VASR_FORCE_PAGED_ATTN=1 cargo run --release -p vasr-models --example bench_text_decode \
  --features "metal,metal-paged-attn,paged-attn,timing" \
  -- Qwen/Qwen3-ASR-0.6B "The capital of France is" 3 64 bf16 8 --warmup 1

# Unit tests
cargo test -p vasr-models --features metal-paged-attn -- kv_cache isq_linear rope
cargo fmt --all -- --check
git diff --check
```

### Next Steps

1. Fuse Q/K norm with mRoPE for seq_len==1 (Metal kernel or mistral.rs-style
   `forward_qk_norm` adaptation).
2. Evaluate `apply_rotary_qk_positions` variant if mRoPE position-aware RoPE is
   needed.
3. Revisit CUDA packed-varlen prefill when a CUDA machine is available.

## 2026-06-08 Pass 15: Audio Encoder BF16 Workaround + PagedCache Metal Default

Date: 2026-06-08

Platform:

- Model: `Qwen/Qwen3-ASR-0.6B`
- Device: Apple Metal
- Runtime dtype: BF16
- Quantization: ISQ 8-bit

### What Changed

#### 1. Audio encoder Metal BF16 GPU→CPU→GPU round-trip removal

Files:

- `vasr_models/src/model/audio_encoder.rs`

Behavior:

- Replaced three GPU→CPU→GPU round-trip workarounds with on-device F32
  intermediate conversions:
  - `cast_input_features_to_weight_dtype`: `to_device(Cpu) → to_dtype → to_device(Metal)`
    replaced with `to_dtype(F32) → to_dtype(BF16)` (both on-device).
  - `pad_audio_chunk`: `to_device(Cpu) → contiguous → pad → to_device(Metal)`
    replaced with `to_dtype(F32) → contiguous → pad → to_dtype(BF16)` (all on-device).
  - `stack_audio_chunks`: per-chunk `to_device(Cpu) → contiguous → stack → to_device(Metal)`
    replaced with per-chunk `to_dtype(F32) → contiguous → stack → to_dtype(BF16)`.

Why:

- Candle's Metal backend has incomplete BF16 support for `pad_with_zeros`,
  `Tensor::stack`, and `to_dtype(BF16)`. The original workaround moved data to
  CPU and back, which:
  - Drains the GPU command buffer (synchronisation point)
  - Copies data over PCI/Thunderbolt (bandwidth bottleneck)
  - Prevents GPU/CPU overlap
- Converting BF16→F32→(do work)→BF16 keeps all operations on the Metal device.
  Two on-device casts are orders of magnitude cheaper than one GPU↔CPU transfer.
- This mirrors the approach used in other Candle Metal workarounds.

Observed effect:

- Modest wall-time improvement in end-to-end ASR (38.5× → 39.4× speedup on
  the 2-file `raw_audios` benchmark).
- The audio encoder is only ~16% of total ASR time, so this is a hygiene win
  rather than a dramatic speedup.

#### 2. PagedCacheConfig Metal default fix

Files:

- `vasr_models/src/lib.rs` (model constructor)

Behavior:

- When `paged_cache: None` is passed in `LoadOptions`, the constructor now
  creates a device-aware default:
  - CUDA: `PagedCacheMemory::GpuMemoryFraction(0.8)` (existing behavior)
  - Metal/non-CUDA: `PagedCacheMemory::ContextSize(100_000)` (new, avoids
    "requires the `cuda` feature" error).

Why:

- `PagedCacheConfig::default()` unconditionally uses `GpuMemoryFraction(0.8)`,
  which panics on Metal because GPU memory fraction queries require the `cuda`
  feature.
- The constructor used `unwrap_or_default()` which picked up this CUDA-only
  default even on Metal.
- This prevented `bench_transcribe` and other examples that pass
  `paged_cache: None` from running on Metal.

#### 3. Audio encoder ISQ attempt (rolled back)

Files:

- `vasr_models/src/model/audio_encoder.rs`
- `vasr_models/src/model/thinker.rs`

What was tried:

- Replaced all `Linear` layers in the audio encoder (attention QKVO, fc1/fc2,
  conv_out, proj1, proj2) with `IsqLinear` so they benefit from AFQ 8-bit
  quantization.

Why it was rolled back:

- The audio encoder's linear layers are relatively small (d_model=512,
  encoder_ffn_dim=2048 for Qwen3-ASR-0.6B). For small matrices, the AfqLayer
  quantize/dequantize overhead exceeds the matmul speedup.
- Measured regression: `audio_encoder_ms` increased from ~72 ms to ~330 ms
  (4.6× slower).
- **Rule of thumb**: ISQ is only beneficial for matrices above a minimum size
  threshold (roughly ≥1024 inner dimension). The text decoder's 1024→3072 MLP
  benefits; the audio encoder's 512→2048 FFN does not.

### Next Steps

1. Fuse Q/K norm with mRoPE for seq_len==1 to close the remaining ~6% gap vs
   mistral.rs.
2. Profile and reduce prefill time (second-largest bottleneck after decode).
3. Evaluate selective ISQ application: only quantize layers above a size
   threshold rather than all-or-nothing.

## 2026-06-08 Pass 16: Per-Batch Timing + Prefill Substages + Causal Mask Skip

Date: 2026-06-08

Platform:

- Model: `Qwen/Qwen3-ASR-0.6B`
- Device: Apple Metal
- Runtime dtype: BF16
- Quantization: ISQ 8-bit

### What Changed

#### 1. Per-batch verbose timing for `vasr-transcribe run`

Files:

- `vasr-cli/Cargo.toml`
- `vasr-models/src/inference/transcribe.rs`
- `vasr-models/src/model/generation.rs`

Behavior:

- `metal-paged-attn` 和 `cuda-paged-attn` feature 默认启用 `timing`，无需
  手动传 `--features timing`。
- `run_asr_on_chunks_batched_timed` 的每个 ASR batch 现在通过 `eprintln!`
  打印 stderr 耗时分解，无需 `RUST_LOG` 即可看到。
- 输出格式：
  ```
  [vasr batch 1/3] items=1 | prepare=113ms | audio_enc=36ms |
    prefill=718ms (inputs=1.6 rope=0.3 fwd=503 other=213) |
    decode=622ms (fwd=199) | steps=128 tokens=128
  ```
- 修复了 varlen paged 路径的计时遗漏：原先 `run_paged_prepared_batch`
  不返回任何 timing，现在改为走 `greedy_generate_cached_batch_timed_with_paged_runtime`
  带计时路径。
- Eager prefill 路径新增 `prefill_inputs_us`、`prefill_rope_us`、
  `prefill_forward_us` 子阶段计时（原先只在 paged 路径有）。

Why:

- 之前无法看到每个 batch 的耗时分解，排查性能瓶颈困难。
- prefill/decode 细分帮助快速定位瓶颈是 forward、RoPE 还是 mask 构建。

#### 2. Metal causal prefill mask skip

Files:

- `vasr-models/src/model/attention.rs`

Behavior:

- `run_attention` 中：Metal 设备上，当 `causal=true` 且 `q_len == k_len`
  （纯 causal prefill，无 padding）时，跳过显式 attention mask 张量，
  直接将 `mask=None + causal=true` 传给 MPS SDPA。
- MPS 内部 causal kernel 比显式 mask 路径更高效。

Observed effect:

- Prefill 时间无明显变化（O(n²) attention 是主要限制）。
- Decode 时间 ~6% 改善（622→582ms），因为 decode 侧也会经过此路径。

#### 3. Prefill bottleneck analysis

Files:

- `vasr-models/src/inference/transcribe.rs`（计时采集）
- `vasr-models/src/model/generation.rs`（计时采集）

Findings from per-batch output (114s audio, 128 decode tokens):

| substage | time | share |
|----------|------|-------|
| prefill inputs (embed + audio merge) | 1.6ms | <1% |
| prefill rope (mRoPE 3D positions) | 0.3ms | <1% |
| **prefill forward (28-layer causal attn)** | **503ms** | **70%** |
| other (mask/metadata/gather/argmax) | 213ms | 30% |
| **total prefill** | **718ms** | — |

Prefill 的 **70% 时间在 forward**，即 28 层 causal attention O(n²)。
Prompt 长度约 1435 tokens（16 base + 1419 audio frames），causal
attention 矩阵约 1435² = 2M 元素/head/层。

Metal MPS SDPA 已在使用，这是 Apple GPU 上最快的 attention 内核。
进一步优化需减小 prompt 长度（`--max-batch-audio-sec 30` 可让
prefill 减半）或复用跨 segment 的 KV cache。

#### 4. Metal paged attention varlen prefill kernel

Files:

- `vasr-paged-attn/build.rs`
- `vasr-paged-attn/src/metal/mod.rs`
- `vasr-paged-attn/src/metal/varlen_prefill.rs`
- `vasr-paged-attn/src/metal/kernels/mod.rs`
- `vasr-paged-attn/src/metal/kernels/prefill_paged_attn.metal`
- `vasr-paged-attn/src/metal/kernels/utils.rs`

Behavior:

- 新增 Metal shader `prefill_paged_attn.metal`：varlen paged prefill 的
  GPU kernel，在 batch 内处理不等长序列的 paged attention。
- `build.rs` 在构建时用 `metal` 命令编译 `.metal` → `.metallib`。
- Rust 封装层通过 `objc2-metal` 调用编译好的 metallib。
- 当前 Metal 默认 eager 路由不使用 paged attention，此 kernel 在
  `VASR_FORCE_PAGED_ATTN=1` 或 CUDA 部署时生效。

### Verification Commands

```bash
# Per-batch verbose timing
cargo build --release --features metal-paged-attn -p vasr-cli --bin vasr-transcribe
./target/release/vasr-transcribe run \
  --input raw_audios \
  --model /path/to/Qwen3-ASR-0.6B \
  --output results --isq 8 \
  --max-batch-audio-sec 60 --max-new-tokens 128 --limit 2 --no-vad

# Full aggregate timing (RUST_LOG=info)
RUST_LOG=vasr_runtime::models::qwen3_asr=info ./target/release/vasr-transcribe run ...
# Outputs: qwen3_asr_timing | items=N | prefill=X.Xs | prefill_forward=Y.Ys | ...
```

### Next Steps

1. **减小 audio chunk**：`--max-batch-audio-sec 30` 让 prefill 约减半。
2. KV cache 跨 segment 复用：同一 system prompt 避免重复 prefill。
3. Fuse Q/K norm + mRoPE：关闭 text decode 剩余 ~6% 差距。
4. CUDA flash-attn：评估 CUDA 部署时的 prefill 加速。
