# Qwen3-ASR CUDA Graph Raw Audios Bottleneck Notes

Date: 2026-06-05

This document records the CUDA `/transcribe` benchmark against the 20-file
`raw_audios` workload, the CUDA graph optimization applied in this pass, and the
remaining bottlenecks observed from timing instrumentation.

## Benchmark Scope

Primary target:

- Model: local `Qwen3-ASR-0.6B`
  (`/home/featurize/data/Qwen/Qwen3-ASR-0___6B`)
- Device: NVIDIA GeForce RTX 3060, 12 GiB
- Runtime dtype: BF16
- Server binary: `target/release/vasr`
- Workload: one HTTP `/transcribe` request containing 20 wav files from
  `raw_audios`
- Audio total duration: `1652.0855 s`
- CUDA graph prewarm: `graphs=12`, `max_batch=20`
- VAD: FSMN VAD on CUDA unless noted otherwise

Server command:

```bash
source ./setup.sh
RUST_LOG=info ./target/release/vasr serve transcribe \
  --model /home/featurize/data/Qwen/Qwen3-ASR-0___6B \
  --host 127.0.0.1 \
  --port 18080 \
  --device cuda \
  --dtype bf16 \
  --max-batch-size 20 \
  --max-batch-audio-sec 180 \
  --vad-model /home/featurize/work/Qwen3ASR.rs/.cache/fsmn-vad
```

Benchmark command:

```bash
python3 scripts/bench_transcribe_http.py \
  --base-url http://127.0.0.1:18080 \
  --audio-dir raw_audios \
  --limit 20 \
  --wait-health-sec 10
```

## Current Results

Warm and hot `/transcribe` runs before the waveform-equivalent feature
extraction optimization:

| Run | Items | Bad | Wall | Throughput | Audio | Speedup | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 0 | `9.535 s` | `2.097 items/s` | `1652.09 s` | `173.26x` | `0.00577` |
| 2 hot | 20 | 0 | `8.841 s` | `2.262 items/s` | `1652.09 s` | `186.87x` | `0.00535` |
| 3 hot | 20 | 0 | `8.663 s` | `2.309 items/s` | `1652.09 s` | `190.70x` | `0.00524` |

Hot average:

- Wall: approximately `8.75 s`
- Throughput: approximately `2.29 items/s`
- Speedup: approximately `189x`
- RTF: approximately `0.00530`

This is still above the expected target of roughly `3 s` for the 20-file batch.

Latest A/B after the waveform-equivalent feature extraction optimization:

| Setting | Bad | Client wall | Pipeline wall | ASR total | Prepare feat | Decode | Text diff vs `waveform-full` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `waveform` default, optimized | 0 | `9.188 s` | `9.18 s` | `8.197 s` | `1.625 s` | `2.874 s` | `0 / 20` |
| `waveform-full` old path | 0 | `9.817 s` | `9.81 s` | `8.890 s` | `2.179 s` | `2.867 s` | baseline |

Interpretation:

- The safe default optimization removed about `0.55 s` from CPU feature
  extraction and about `0.63-0.69 s` from end-to-end/server ASR time on this
  run.
- Transcripts matched exactly against the old full waveform-padding path for
  all 20 returned items.
- The overall request is still around `9 s`, so the 3-second target remains
  blocked by the GPU forward/synchronization portions rather than only by
  feature extraction.

Latest timing-probe run with prefill substage instrumentation:

```text
pipeline | batch=20 | returned=20 | audio=1652.09s | spent=9.24s | speed=178.71x | rtf=0.006 | bad=0

qwen3_asr_timing |
  items=90
  chunks=90
  batches=7
  total=8.298s
  prepare=1.714s
  prepare_feat=1.678s
  audio_encoder=1.324s
  prefill=2.070s
  prefill_inputs=0.064s
  prefill_rope=0.005s
  prefill_metadata=0.000s
  prefill_mask=0.001s
  prefill_forward=1.998s
  prefill_gather=0.000s
  prefill_decode_setup=0.014s
  prefill_argmax=0.184s
  decode=2.876s
  decode_graph_replay=0.163s
  decode_forward=0.163s
  decode_argmax=2.707s
  decode_steps=218
  generated_tokens=1354
```

Interpretation:

- Prefill is dominated by the model forward itself (`prefill_forward=1.998s`),
  not by mRoPE, metadata construction, mask construction, or gathering last
  logits.
- The first post-prefill argmax/sync is visible but smaller (`0.184s`) than the
  token-by-token decode sync (`2.707s`).
- This makes flash attention or other prefill-forward kernel work the relevant
  prefill path; CPU-side prefill setup is not a meaningful target on this
  workload.

## Effective Optimization In This Pass

### CUDA graph replay padding stays on device

Files:

- `vasr_models/src/model/cuda_graph.rs`

Change:

- Reworked CUDA graph replay padding for active decode batches.
- Replaced host materialization in `pad_decode_batch_to_max` and
  `pad_metadata_block_tables`:
  - old path: `to_vec*` on GPU tensors, rebuild `Vec`, then `Tensor::from_vec`
  - new path: `pad_with_zeros`, `Tensor::ones`, `broadcast_as`, and `Tensor::cat`
- Preserved the existing dummy metadata semantics:
  - inactive batch rows use token id `0`, slot `0`, context length `1`
  - block-table row padding repeats each active row's last block id
  - dummy block-table rows are all zeros

Why it helps:

- It follows the same broad shape as `xinfer`: replay inputs live as stable CUDA
  tensors and are updated without pulling metadata back to the host.
- It removes unnecessary CPU/GPU synchronization from the graph replay setup
  path.

Observed effect:

- This change improves the replay setup path, but the current benchmark shows
  CUDA graph forward itself is already small. In hot runs, decode graph replay is
  tens of milliseconds, while argmax synchronization is roughly `2.4 s`.
- Therefore this optimization is correct but not sufficient for the 3-second
  target.

Verification:

```bash
source ./setup.sh
cargo test -p vasr-models cuda_graph::tests --features cuda
cargo check -p vasr-models --features cuda
cargo fmt --check
```

### Feature extraction padding mode A/B and safe default optimization

Files:

- `vasr_models/src/processor/feature_extractor.rs`
- `vasr_models/src/processor/asr_processor.rs`
- `vasr_models/src/inference/transcribe.rs`
- `vasr_runtime/src/models/qwen3_asr.rs`

Change:

- Split `prepare` timing into:
  - `prepare_norm`
  - `prepare_tok_lookup`
  - `prepare_feat`
  - `prepare_tok_expand`
  - `prepare_pad`
- Added an explicit feature extraction padding mode:

```bash
VASR_FEATURE_PADDING_MODE=waveform       # default, optimized and output-equivalent
VASR_FEATURE_PADDING_MODE=waveform-full  # old full waveform-padding path for A/B
VASR_FEATURE_PADDING_MODE=feature        # experimental faster path
```

- `waveform` path: preserve the old zero-padded waveform boundary behavior for
  real frames, but skip FFT/log-mel work for frames that will be fully masked
  out later.
- `waveform-full` path: pad every waveform in the ASR micro-batch to the
  longest waveform, then run CPU log-mel feature extraction for every padded
  sample. This is the previous behavior and is kept for A/B verification.
- `feature` path: run CPU log-mel feature extraction on each real waveform, then
  right-pad the resulting feature rows and feature masks to the batch max frame
  length.

Why it helps:

- The audio encoder consumes `feature_lens` and narrows each sample to its real
  feature length before `forward_one`.
- The optimized `waveform` path computes only the real prefix frames, but uses
  the batch `padded_len` as the reflected source length and returns zero for
  source samples beyond the real waveform. That preserves the old full-padding
  context for boundary frames.
- Padding feature rows after extraction preserves the batch tensor shape needed
  by `stack_features`, while avoiding FFT/log-mel work on artificial fully
  masked waveform tails.

Observed result:

| Setting | Bad | Wall | ASR total | Prepare | Prepare feat | Decode steps | Generated tokens | Text diff vs waveform |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous hot | 0 | `8.663 s` | `8.013 s` | `1.985 s` | not split | `218` | `1354` | baseline |
| `waveform` mode | 0 | `9.299 s` | `8.385 s` | `2.152 s` | `2.127 s` | `218` | `1354` | baseline |
| `feature` mode | 0 | `8.583 s` | `7.644 s` | `1.614 s` | `1.593 s` | `204` | `1323` | `7 / 20` |
| optimized `waveform` | 0 | `9.188 s` | `8.197 s` | `1.654 s` | `1.625 s` | `218` | `1354` | `0 / 20` |
| `waveform-full` | 0 | `9.817 s` | `8.890 s` | `2.205 s` | `2.179 s` | `218` | `1354` | baseline |

Interpretation:

- `prepare` was almost entirely CPU feature extraction.
- The safe optimized `waveform` mode reduced `prepare_feat` by about `0.55 s`
  versus `waveform-full` in the latest A/B, with exact text parity on all 20
  returned items.
- Feature-level padding reduced `prepare_feat` by about `0.53 s` versus the
  earlier measured waveform-padding A/B run.
- `/transcribe` benchmark correctness stayed parse-clean (`bad=0`).
- A/B transcript comparison for the earlier `feature` mode found text
  differences in `7 / 20` returned items. Therefore `feature` mode is not safe
  as the default.
- The default is now the optimized `waveform` path, while `waveform-full` remains
  available for old-path verification and `feature` remains an opt-in
  experiment.

Verification:

```bash
source ./setup.sh
cargo test -p vasr-models feature_extractor::tests --features cuda,timing
cargo test -p vasr-models asr_processor::tests --features cuda,timing
cargo test -p vasr-models generation::tests --features cuda,timing
cargo test -p vasr-models cuda_graph::tests --features cuda,timing
cargo check -p vasr-runtime --features cuda,timing
cargo build --release -p vasr-cli --bin vasr --features cuda,timing
python3 scripts/bench_transcribe_http.py \
  --base-url http://127.0.0.1:18080 \
  --audio-dir raw_audios \
  --limit 20 \
  --wait-health-sec 10
```

For the A/B runs, set `VASR_FEATURE_PADDING_MODE` on the server process before
starting `vasr serve transcribe`.

## Additional Instrumentation In This Pass

### Decode pre-argmax synchronization probe

Files:

- `vasr_models/src/model/generation.rs`
- `vasr_models/src/inference/transcribe.rs`
- `vasr_runtime/src/models/qwen3_asr.rs`

Change:

- Added an opt-in timing probe controlled by:

```bash
VASR_TIMING_SYNC_BEFORE_ARGMAX=1
```

- When enabled, the decode loop calls `device.synchronize()` immediately before
  argmax and records the time as `decode_pre_argmax_sync`.
- Default behavior is unchanged when the environment variable is unset.

Why it matters:

- CUDA graph replay and forward launches are asynchronous.
- The old `decode_argmax` bucket included the first forced synchronization caused
  by `argmax(...).to_vec1::<u32>()`.
- Without this probe, `decode_argmax=~2.4s` looked like an argmax reduction issue.

Observed result with the probe enabled:

```text
pipeline | batch=20 | returned=20 | audio=1652.09s | spent=9.81s | speed=168.46x | rtf=0.006

qwen3_asr_timing |
  items=90
  chunks=90
  batches=7
  total=8.794s
  prepare=2.127s
  audio_encoder=1.384s
  prefill=2.105s
  decode=2.861s
  decode_graph_replay=0.146s
  decode_forward=0.146s
  decode_pre_argmax_sync=2.674s
  decode_argmax=0.036s
  decode_steps=218
  generated_tokens=1354
```

Interpretation:

- The batch argmax reduction and tiny token-id host copy are not the dominant
  cost by themselves (`0.036 s` with the stream already synchronized).
- The apparent `decode_argmax=~2.4s` bottleneck is mostly the synchronization
  point where CPU decode logic waits for pending CUDA graph/forward work.
- A standalone custom CUDA argmax still leaves this synchronization boundary in
  place, which explains why that experiment regressed.
- The useful next-token optimization is to keep argmax, EOS masking, and next
  input token update on device, then avoid per-step host synchronization where
  possible.

Default-path check with the probe unset:

```text
pipeline | batch=20 | returned=20 | audio=1652.09s | spent=8.86s | speed=186.56x | rtf=0.005

qwen3_asr_timing |
  total=8.070s
  prepare=2.042s
  audio_encoder=1.264s
  prefill=2.046s
  decode=2.441s
  decode_graph_replay=0.029s
  decode_pre_argmax_sync=0.000s
  decode_argmax=2.409s
  decode_steps=218
  generated_tokens=1354
```

This confirms the probe does not add synchronization unless explicitly enabled.

## Bottleneck Breakdown

Representative hot run before the latest waveform-equivalent feature extraction
optimization, with VAD enabled and `--max-batch-audio-sec 180`:

```text
pipeline | batch=20 | returned=20 | audio=1652.09s | spent=8.66s | speed=190.71x | rtf=0.005

qwen3_asr_timing |
  items=90
  chunks=90
  batches=7
  total=8.013s
  prepare=1.985s
  stack=0.018s
  audio_encoder=1.287s
  prefill=2.232s
  decode=2.430s
  decode_graph_replay=0.027s
  decode_forward=0.027s
  decode_argmax=2.400s
  decode_steps=218
  generated_tokens=1354
```

### 1. Decode synchronization at next-token selection

Current observation:

- `decode_graph_replay`: about `0.027 s`
- `decode_argmax`: about `2.400 s`
- With `VASR_TIMING_SYNC_BEFORE_ARGMAX=1`:
  - `decode_pre_argmax_sync`: `2.674 s`
  - `decode_argmax`: `0.036 s`

Why this is the main decode bottleneck:

- CUDA graph replay is already fast.
- The decode loop still crosses back to CPU every token step to obtain next token
  ids and update EOS state.
- The host readback forces the CPU to wait for pending CUDA work. The measured
  cost is mostly that wait, not the argmax reduction after synchronization.
- The next step input token tensor is then rebuilt from a CPU `Vec<u32>`.

Likely optimization:

- Keep `[batch, vocab] -> next_ids`, EOS masking, and next graph input update on
  device.
- Avoid per-step host synchronization for all next ids.
- Use a minimal completion signal only when needed to stop the loop.

Expected value:

- The largest single visible decode win is the roughly `2.4-2.7 s` currently
  spent at the next-token synchronization boundary.
- Removing it alone would still leave the hot run around `6 s`, so it must be
  combined with prefill/prepare work to approach `3 s`.

### 2. Prefill time

Current observation:

- `prefill`: about `2.2-2.4 s` across the VAD-enabled runs.
- With prefill substage instrumentation:
  - `prefill_forward`: `1.998 s`
  - `prefill_inputs`: `0.064 s`
  - `prefill_argmax`: `0.184 s`
  - mRoPE, metadata, mask, and gather are all near-zero at this scale.

Why it matters:

- VAD produces around `90` ASR segments/chunks for the 20 input files.
- Each micro-batch still pays text/audio prompt prefill cost before decode.
- CUDA graph only covers single-token decode, not prefill.

Likely optimization:

- Profile paged prefill specifically on CUDA. The current evidence points at the
  model forward kernels, not CPU setup.
- Revisit packed/varlen prefill so mixed-length VAD segments avoid extra
  padding/unpack work.
- Reuse or precompute masks/metadata more aggressively across layers and
  micro-batches.
- Evaluate `--flash-attn` for CUDA prefill once correctness and build support are
  confirmed for this model path.

Expected value:

- Prefill is another roughly `2.3 s` of the hot run. It must move substantially
  for a `3 s` end-to-end target.
- Setup-only changes can at best recover a few tens of milliseconds; material
  gains need the forward kernels to move.

### 3. Prepare time

Current observation:

- Optimized default waveform mode:
  - `prepare`: `1.654 s`
  - `prepare_feat`: `1.625 s`
- Old `waveform-full` mode:
  - `prepare`: `2.205 s`
  - `prepare_feat`: `2.179 s`
- Experimental feature-padding mode:
  - `prepare`: about `1.60 s`
  - `prepare_feat`: about `1.59 s`
  - transcript text differs from waveform mode in `7 / 20` items

Why it matters:

- This is comparable to prefill and decode argmax.
- It happens before audio encoder/prefill and is paid across the 90 prepared
  chunks.
- It is almost entirely CPU log-mel feature extraction, not tokenizer/template
  construction.

Likely optimization:

- Preserve waveform-padding output behavior by default; the current default now
  does this while skipping fully masked padded frames.
- Further reduce CPU feature extraction cost with cached/reused FFT plans,
  faster STFT/log-mel implementation, or GPU feature extraction.

Expected value:

- The safe default optimization already captured about `0.55 s`.
- The remaining default `prepare_feat=1.625 s` is still meaningful, but it is now
  smaller than decode synchronization and prefill. Further wins probably require
  a faster STFT/log-mel implementation or moving feature extraction to GPU.

### 4. Audio encoder time

Current observation:

- `audio_encoder`: about `1.3-1.4 s` in VAD-enabled runs.

Why it matters:

- It is not the largest bottleneck, but after decode argmax and prefill are
  improved it will become a visible part of the remaining budget.

Likely optimization:

- Keep current batching, but profile encoder feature lengths and CUDA kernel
  occupancy.
- Check whether all VAD segments are batched with efficient shape buckets.
- Revisit audio tower flash-attn/varlen behavior on CUDA if prefill work is not
  enough.

## Negative Or Neutral Experiments

### Custom CUDA argmax with per-step CPU readback

Change tested:

- Added a CUDA `[batch, vocab] -> [batch]` argmax kernel for F32/F16/BF16 logits.
- Replaced the decode loop's batch `logits.argmax(1).to_vec1::<u32>()` with the
  custom kernel followed by `DtoH` of only `batch` token ids.

Observed result:

| Setting | Wall | Throughput | Speedup | RTF | Decode argmax |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline hot | `8.663 s` | `2.309 items/s` | `190.70x` | `0.00524` | `2.400 s` |
| Custom CUDA argmax | `9.971 s` | `2.006 items/s` | `165.69x` | `0.00604` | `2.795 s` |

Interpretation:

- The bottleneck is not the argmax reduction kernel alone.
- The pre-argmax synchronization probe later confirmed that the per-step host
  readback absorbs pending graph/forward work.
- A standalone custom argmax plus `DtoH` is not enough; the next-token update and
  EOS state must stay on device to remove the synchronization boundary.

### Device-side next token without device-side EOS

Change tested:

- Kept next-token ids as CUDA tensors between decode steps.
- Deferred CPU readback until the end by concatenating generated token tensors.
- Did not yet move `finished` / EOS early-stop bookkeeping to CUDA, so the loop had
  to run to `max_new_tokens` for every batch.

Observed result:

| Setting | Wall | Throughput | Speedup | RTF | Decode | Decode steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline hot | `8.663 s` | `2.309 items/s` | `190.70x` | `0.00524` | `2.430 s` | `218` |
| Device token, no EOS state | `30.201 s` | `0.662 items/s` | `54.70x` | `0.01828` | `23.523 s` | `1792` |

Interpretation:

- Avoiding per-step readback is directionally right, but running every micro-batch
  to the full token limit is far too expensive.
- `decode_steps` grew from about `224` to `1792` because seven ASR micro-batches
  all ran the full decode budget.
- A viable GPU-side decode path must also keep active/finished flags and EOS
  masking on device, while still providing a cheap way to detect global completion.

### Dynamic active-row compaction

Reference:

- `osc_transformers` removes finished sequences from the running queue at the
  scheduler level.

Change tested:

- After each decode argmax, compacted the active batch to rows that had not
  emitted EOS.
- Selected the matching `position_ids`, `slot_mapping`, `block_tables`, and
  `context_lens` rows before CUDA graph replay.
- Initial experiment hit a Candle contiguity failure on broadcast
  `position_ids`:

```text
index-select only supports contiguous tensors
```

- The experiment was fixed by making the selected tensors contiguous before
  `index_select`.

Observed result:

| Setting | Bad | Wall | Decode | Graph replay | Argmax | Decode steps | Generated tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline hot | 0 | `8.663 s` | `2.430 s` | `0.027 s` | `2.400 s` | `218` | `1354` |
| Active compaction run 1 | 0 | `9.678 s` | `2.995 s` | `0.512 s` | `2.456 s` | `210` | `1353` |
| Active compaction hot | 0 | `8.944 s` | `2.556 s` | `0.026 s` | `2.506 s` | `210` | `1353` |

Interpretation:

- Correctness was OK after the contiguity fix (`bad=0`).
- The loop saved only `8` decode iterations (`218 -> 210`) on this workload.
- Extra metadata/position selection work and changed graph bucket behavior did
  not produce an end-to-end win.
- This optimization was not kept in the default hot path. Stable fixed-batch
  replay with EOS fill remains better aligned with the current CUDA graph capture
  strategy.

### Disabling max batch audio seconds

Command difference:

```bash
--max-batch-audio-sec 0
```

Observed result:

| Setting | Wall | Speedup | RTF | ASR batches | ASR total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `180` hot | `8.663 s` | `190.70x` | `0.00524` | `7` | `8.013 s` |
| `0` | `9.126 s` | `181.02x` | `0.00552` | `5` | `8.221 s` |

Interpretation:

- Reducing ASR micro-batch count from `7` to `5` did not improve end-to-end
  throughput.
- The current `max_batch_audio_sec` cap is not the main bottleneck.

### Disabling VAD

Command difference:

```bash
--no-vad
```

Observed result:

| Setting | Wall | Speedup | RTF | ASR items | ASR batches | ASR total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VAD enabled hot | `8.663 s` | `190.70x` | `0.00524` | `90` | `7` | `8.013 s` |
| `--no-vad` | `35.140 s` | `47.01x` | `0.02127` | `15` | `8` | `34.918 s` |

Additional observation:

```text
CUDA decode graph key disabled after decode error:
CUDA graph capture failed for batch=4 block_table_cols=64:
CUDA graph capture warmup forward failed:
DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")
```

Interpretation:

- VAD is not the cause of the current slowdown.
- Without VAD, long raw chunks make prepare/prefill/decode much more expensive.
- The default VAD segmentation is necessary for this workload.
- Long raw chunks can hit CUDA graph capture/fallback cases that should be
  investigated separately, but they are not on the default fast path.

### Flash-attn feature wiring and build attempt

Change:

- Added Cargo feature passthroughs:
  - `vasr-runtime/flash-attn = ["cuda", "vasr-models/flash-attn"]`
  - `vasr-cli/flash-attn = ["cuda", "vasr-runtime/flash-attn"]`
- This aligns the existing `vasr serve --flash-attn` runtime flag with an
  actually buildable CLI feature.

Build command attempted:

```bash
source ./setup.sh
cargo build --release -p vasr-cli --bin vasr --features cuda,timing,flash-attn
```

Observed result:

- The initial missing-feature error was fixed.
- The build entered `candle-flash-attn` and launched many `nvcc` compilations for
  BF16/FP16 flash forward kernels across head dimensions and causal/non-causal
  variants, for example:
  - `flash_fwd_hdim128_bf16_causal_sm80.cu`
  - `flash_fwd_hdim256_bf16_sm80.cu`
  - `flash_fwd_hdim224_bf16_sm80.cu`
  - `flash_fwd_hdim128_fp16_sm80.cu`
- After several minutes it was still compiling additional kernel variants. The
  experiment was terminated manually to avoid spending the rest of the iteration
  on first-time kernel compilation.

Interpretation:

- Flash-attn remains a plausible prefill/audio-encoder optimization because the
  new breakdown shows prefill is forward-kernel dominated.
- It should be evaluated in a separate run where the `candle-flash-attn` kernels
  are allowed to precompile fully and the build artifact is reused for A/B.
- It is not yet proven as a runtime optimization for this workload.

## Current Root Cause Summary

The 20-file CUDA `/transcribe` run is not limited by HTTP overhead or CUDA graph
forward replay. The current bottleneck stack is:

1. Per-step CPU synchronization during next-token selection in decode.
2. CUDA prefill across many VAD segments.
3. CPU log-mel feature extraction inside `prepare`.
4. Audio encoder work, secondary today but likely important after the above move.

The applied CUDA graph padding optimization removes avoidable host materialization
from replay setup, but the timing data shows the larger gap to a `3 s` target is
elsewhere.

The active-row compaction experiment suggests that scheduler-style row removal is
not a P0 optimization for this 20-file workload: there is not enough tail
divergence to offset the extra per-step tensor selection work.

## Recommended Next Passes

### P0: Device-side next-token update plus EOS state

Goal:

- Remove the per-step next-token host synchronization from the hot decode loop
  without forcing every request to run to `max_new_tokens`.

Implementation direction:

- Add a CUDA kernel or reuse an existing CUDA reduction primitive to compute
  next ids for each batch row.
- Store next ids in a persistent device tensor.
- Feed CUDA graph replay from device-side token buffers instead of rebuilding
  `Tensor::from_vec(tokens_in, ...)` from CPU every step.
- Maintain device-side `finished` flags and EOS masking so finished rows feed the
  EOS fill token and the loop can stop near the current `~224` decode-step count,
  not the full `1792` worst-case budget observed in the naive device-token
  experiment.
- Use a narrow completion signal readback, or an async/captured reduction, instead
  of reading all next ids every step.
- Do not rely on active-row compaction alone; it saved too few decode steps on
  the measured workload.

Validation:

- Compare generated token ids against the current CPU argmax path on a small
  deterministic fixture.
- Re-run the 20-file `raw_audios` benchmark and verify the per-step next-token
  synchronization drops materially while `decode_steps` stays close to the
  baseline.

### P1: Validate and further reduce CPU feature extraction

Goal:

- Reduce the default `prepare_feat=~2 s` CPU cost without changing transcripts.

Implementation direction:

- Keep `waveform` as the default padding mode.
- Use `VASR_FEATURE_PADDING_MODE=feature` only as an experiment/reference point.
- Implement a waveform-equivalent extractor that computes only real frames while
  preserving zero-padding tail context, or evaluate a GPU/faster CPU log-mel path.

Validation:

- Re-run the same HTTP benchmark and compare transcript text, `prepare_feat`,
  wall time, and `bad` count.

### P1: CUDA paged prefill optimization

Goal:

- Reduce `prefill=~2.3 s` for mixed-length VAD segment batches.

Implementation direction:

- Profile current paged prefill path on CUDA.
- Revisit packed varlen prefill and reduce pad/unpack overhead.
- Test CUDA flash attention for prefill where compatible.

Validation:

- Compare prefill timings and transcript parity on the 20-file workload.

### P2: Audio encoder profiling

Goal:

- Keep audio encoder from becoming the next blocker after decode and prefill
  improvements.

Implementation direction:

- Record feature length distribution for VAD segments.
- Bucket encoder inputs by shape when it improves occupancy.
- Compare encoder timings with and without flash-attn/varlen paths.
