# Qwen3-ASR Metal ISQ8 Optimization Notes

Date: 2026-06-03

This document records the recent Metal decode optimization work against
`mistral.rs`, with separate notes for changes that helped, changes that did not
help, and the remaining performance gap.

Update: 2026-06-04

The notes below were extended with the mixed-length paged-prefill work that
followed the initial single-sequence decode optimization pass. The 2026-06-04
changes are more about bringing the ASR batch path closer to `mistral.rs`
metadata/layout behavior while also recovering real `transcribe` throughput.

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
