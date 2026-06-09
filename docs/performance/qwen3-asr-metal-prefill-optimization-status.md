# Qwen3-ASR Metal Prefill Optimization Status

Date: 2026-06-09
Last updated: 2026-06-09 08:55:00 CST

## Scope

This document consolidates the recent Qwen3-ASR prefill-related optimization work on
macOS Metal, including:

- the machine/hardware used for all measurements in this note,
- the current measured runtime behavior,
- what changes have already landed,
- which changes actually reduce prefill compute,
- which changes only improve surrounding overhead,
- and what still remains to reach a true varlen-prefill implementation.

This note is intentionally prefill-centric. Decode-only optimizations are included only
when they materially affect the same code paths or the interpretation of timing output.

## Test Machine

All measurements below were taken on the local macOS Metal machine used in this repo.

### Hardware

- CPU: `Apple M4 Pro`
- GPU: `Apple M4 Pro` integrated GPU
- GPU cores: `20`
- Metal support: `Metal 4`
- Unified memory: `51539607552` bytes (`48 GiB`)

### Display probe output used for hardware confirmation

```text
Graphics/Displays:

    Apple M4 Pro:

      Chipset Model: Apple M4 Pro
      Type: GPU
      Bus: Built-In
      Total Number of Cores: 20
      Vendor: Apple (0x106b)
      Metal Support: Metal 4
```

## Current Measurement Command

### VAD-enabled raw audios timed run

```bash
./target/release/vasr-transcribe run \
  --input raw_audios \
  --model /Users/wangmengdi/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0 \
  --output /tmp/vasr-bench-vad-out \
  --isq 8 \
  --max-batch-audio-sec 60 \
  --max-new-tokens 128 \
  --limit 20
```

## Current Measured Results

## VAD-enabled run (`raw_audios`, 20 files)

Observed final summary:

```text
Done: files=20 bad=0 audio_seconds=1652.085 wall_seconds=24.310 speedup=67.960 rtf=0.0147
```

Representative batches from this run:

```text
[vasr batch 1/20] items=13 | prepare=50.1ms | audio_enc=55.7ms | prefill=899.4ms (inputs=2.1 rope=0.9 fwd=580.4 other=315.9) | decode=319.3ms (fwd=79.0) | steps=38 tokens=60
[vasr batch 2/20] items=7 | prepare=46.2ms | audio_enc=21.8ms | prefill=660.5ms (inputs=0.3 rope=0.2 fwd=435.7 other=224.3) | decode=1362.9ms (fwd=268.3) | steps=152 tokens=166
[vasr batch 1/20] items=14 | prepare=30.8ms | audio_enc=31.9ms | prefill=867.3ms (inputs=0.5 rope=0.4 fwd=551.2 other=315.2) | decode=1517.5ms (fwd=302.2) | steps=167 tokens=190
[vasr batch 2/20] items=5 | prepare=52.7ms | audio_enc=31.3ms | prefill=756.7ms (inputs=0.5 rope=0.3 fwd=498.6 other=257.3) | decode=1648.1ms (fwd=245.2) | steps=140 tokens=139
[vasr batch 3/20] items=9 | prepare=55.0ms | audio_enc=27.9ms | prefill=778.1ms (inputs=76.9 rope=0.2 fwd=453.1 other=247.9) | decode=196.6ms (fwd=46.3) | steps=24 tokens=39
[vasr batch 1/9] items=4 | prepare=55.6ms | audio_enc=27.0ms | prefill=681.5ms (inputs=0.4 rope=0.2 fwd=449.9 other=231.0) | decode=945.4ms (fwd=239.5) | steps=136 tokens=133
```

Interpretation:

- VAD materially improves end-to-end behavior by creating more segment-level batching.
- However, prefill is still expensive even with VAD: many batches remain in the
  `~680-900ms` range, and `prefill_forward` remains the dominant term.
- This means the current bottleneck is still real prefill compute, not merely outer
  pipeline inefficiency.

## Latest Experimental Findings

This section records the most recent prefill experiments after the baseline above.

### 1. Prefix-KV reuse for the short text prefix is not enough

Implementation status:

- A batch-level/common-prefix KV reuse path was implemented for the shared text prefix
  before the audio placeholder.
- The implementation was then refined so the first batch no longer pays an extra
  prefix-only prefill cost; instead, the first full prefill result is sliced into a
  reusable prefix template.

Measured result:

```text
Done: files=20 bad=0 audio_seconds=1652.085 wall_seconds=25.976 speedup=63.601 rtf=0.0157
```

Interpretation:

- Removing the extra first-batch prefix prefill overhead worked.
- However, the reusable text prefix in the current Qwen3-ASR prompt shape is too short.
- The saved prefill work is smaller than the extra cache-copy / branching overhead, so
  this is not a net win on this workload.

### 2. Fully forcing the current Metal paged path is not acceptable

Experiment:

- Forced the current Metal paged path with `VASR_FORCE_PAGED_ATTN=1`.

Observed behavior:

- Some prefill batches became much smaller.
- Decode behavior regressed badly, including abnormally large decode step counts and
  token counts.

Interpretation:

- The current Metal paged/varlen prefill path has real prefill potential.
- But the full paged generation stack is not suitable as the default path on this
  workload because it harms decode behavior.

### 3. Hybrid experiment: paged/varlen prefill + eager decode

Implementation status:

- Added an experimental path gated behind:

```bash
VASR_ENABLE_METAL_PAGED_PREFILL_HYBRID=1
```

- This path:
  - runs batch prefill through the paged/varlen machinery,
  - reconstructs an eager `KVCache` from the paged cache,
  - then continues with the existing eager decode loop.

Current status:

- This is still experimental and is not enabled by default.
- Decode correctness is now preserved only for the current restricted hybrid path
  described below.

#### Root cause investigation result

The original hybrid path produced wrong text even on a one-file/two-step reproduction.

That investigation isolated the failure to two Metal-side fast paths used during paged
prefill:

- the Metal paged varlen prefill kernel,
- and the fallback path's packed-varlen SDPA branch.

In the current experimental hybrid mode, both of these paths are bypassed, and the
hybrid route now uses the safer paged-prefill fallback behavior before reconstructing
the eager decode KV cache.

This means:

- the hybrid design itself is viable,
- the main issue was not the paged-prefill concept,
- the issue was the interaction of those two fast paths with the current batched paged
  prefill semantics.

#### One-token isolation test

To separate "prefill quality" from "multi-step decode handoff", a one-token run was
measured with `--max-new-tokens 1`.

Baseline command shape:

```bash
./target/release/vasr-transcribe run \
  --device metal \
  --input raw_audios \
  --model /Users/wangmengdi/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0 \
  --output /tmp/vasr-bench-vad-out-1tok-base \
  --isq 8 \
  --max-batch-audio-sec 60 \
  --max-new-tokens 1 \
  --limit 20
```

Baseline result:

```text
Done: files=20 bad=0 audio_seconds=1652.085 wall_seconds=25.153 speedup=65.680 rtf=0.0152
```

Hybrid command shape:

```bash
VASR_ENABLE_METAL_PAGED_PREFILL_HYBRID=1 \
./target/release/vasr-transcribe run \
  --device metal \
  --input raw_audios \
  --model /Users/wangmengdi/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0 \
  --output /tmp/vasr-bench-vad-out-1tok-hybrid \
  --isq 8 \
  --max-batch-audio-sec 60 \
  --max-new-tokens 1 \
  --limit 20
```

Hybrid result:

```text
Done: files=20 bad=0 audio_seconds=1652.085 wall_seconds=12.488 speedup=132.296 rtf=0.0076
```

Interpretation:

- This is the strongest evidence so far that the paged/varlen prefill itself is
  useful on Metal for this workload.
- The hybrid path works very well when the task is reduced to prefill plus only the
  first decode token.
- Therefore the current blocker is specifically multi-step decode continuation after
  paged prefill, not the prefill optimization itself.

#### Current corrected hybrid result

After restricting the experimental hybrid mode to avoid the two incorrect fast paths
above, the normal VAD benchmark was rerun with only this single gate enabled:

```bash
VASR_ENABLE_METAL_PAGED_PREFILL_HYBRID=1 \
./target/release/vasr-transcribe run \
  --device metal \
  --input raw_audios \
  --model /Users/wangmengdi/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0 \
  --output /tmp/vasr-bench-vad-out-hybrid-final \
  --isq 8 \
  --max-batch-audio-sec 60 \
  --max-new-tokens 128 \
  --limit 20
```

Observed final summary:

```text
Done: files=20 bad=0 audio_seconds=1652.085 wall_seconds=21.077 speedup=78.382 rtf=0.0128
```

Interpretation:

- This is the first confirmed end-to-end Metal prefill optimization in this note that
  produces a real net improvement on the full VAD benchmark without harming decode
  correctness on the validated repro cases.
- Relative to the current baseline (`24.310s`), the corrected hybrid path reduces total
  wall time by about `13.3%`.
- The remaining optimization work should now focus on recovering more of the fast-path
  prefill benefit without reintroducing the correctness regressions seen earlier.

#### Current detailed timing breakdown

For the corrected hybrid path, a verbose run was captured with:

```bash
VASR_ENABLE_METAL_PAGED_PREFILL_HYBRID=1 \
./target/release/vasr-transcribe run \
  -v \
  --device metal \
  --input raw_audios \
  --model /Users/wangmengdi/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0 \
  --output /tmp/vasr-bench-vad-out-hybrid-metrics \
  --isq 8 \
  --max-batch-audio-sec 60 \
  --max-new-tokens 128 \
  --limit 20
```

The final pipeline summary from that run was:

```text
pipeline | batch=20 | returned=20 | audio=1652.09s | spent=21.08s | speed=78.38x | rtf=0.013 | bad=0
Done: files=20 bad=0 audio_seconds=1652.085 wall_seconds=21.077 speedup=78.382 rtf=0.0128
```

The run produced 5 `qwen3_asr_timing` summaries for the ASR sub-batches inside the full
pipeline. Aggregating those 5 summaries gives the current stage-level totals below.

##### Aggregated ASR-stage totals inside the full 20-file run

| Stage | Total time | Notes |
| --- | ---: | --- |
| `prepare` | `0.750s` | Batch-side processor work before the model |
| `stack_features` | `0.014s` | Feature stacking / transfer into model input tensors |
| `audio_encoder` | `0.410s` | Audio tower forward |
| `prefill` | `10.864s` | Total prefill phase |
| `prefill_forward` | `8.756s` | Core prefill model forward inside prefill |
| `decode` | `8.712s` | Total decode phase |
| `decode_forward` | `1.637s` | Core token forward inside decode |
| `decode_argmax` | `7.050s` | Token selection / host-visible argmax cost |

##### Current token throughput

These values are aggregated from the same 5 `qwen3_asr_timing` summaries:

- `prefill_tokens = 20104`
- `generated_tokens = 1013`
- `prefill tok/s = 20104 / 10.864s = 1850.3 tok/s`
- `decode tok/s = 1013 / 8.712s = 116.3 tok/s`

Interpretation:

- Prefill is now clearly faster than decode in raw token throughput.
- Even after the hybrid prefill improvement, the largest decode-side cost is not the
  decode forward itself, but `decode_argmax`.
- On the prefill side, the dominant cost remains `prefill_forward`, not input
  preparation, RoPE, or metadata.

##### Representative per-sub-batch timing summaries

These are direct `qwen3_asr_timing` samples from the same verbose run:

```text
qwen3_asr_timing | items=20 | chunks=20 | batches=2 | total=3.070s | prepare=0.077s | stack=0.002s | audio_encoder=0.068s | prefill=1.550s | prefill_tokens=2346 | prefill_tok_s=1513.4 | prefill_forward=1.397s | decode=1.365s | decode_tok_s=172.2 | decode_forward=0.332s | decode_steps=192 | generated_tokens=235

qwen3_asr_timing | items=20 | chunks=20 | batches=4 | total=6.764s | prepare=0.191s | stack=0.003s | audio_encoder=0.097s | prefill=2.790s | prefill_tokens=5119 | prefill_tok_s=1835.0 | prefill_forward=2.331s | decode=3.673s | decode_tok_s=94.5 | decode_forward=0.596s | decode_steps=343 | generated_tokens=347

qwen3_asr_timing | items=9 | chunks=9 | batches=3 | total=3.035s | prepare=0.142s | stack=0.002s | audio_encoder=0.064s | prefill=1.745s | prefill_tokens=3588 | prefill_tok_s=2055.9 | prefill_forward=1.286s | decode=1.076s | decode_tok_s=132.9 | decode_forward=0.257s | decode_steps=148 | generated_tokens=143
```

These samples show the current range:

- `prefill tok/s` is roughly `1.5k - 2.1k tok/s`
- `decode tok/s` is roughly `95 - 172 tok/s`
- `prefill_forward` still dominates the prefill phase
- `decode_argmax` remains a large fraction of decode wall time

## Changes Already Landed

This section records the prefill-related code changes already made in the local repo.

### 1. Eager attention input materialization before Metal/CUDA attention

Files:

- `vasr-models/src/model/thinker_text.rs`

Change:

- Before entering `run_attention(...)` on the eager path, Q/K/V now go through the
  accelerator packing check and `contiguous()` materialization when needed.

Why it matters:

- This aligns the input layout discipline more closely with `mistralrs`' Metal fast
  paths, which explicitly materialize kernel inputs before dispatch.
- This does not change algorithmic complexity, but reduces layout-related overhead.

### 2. Prefill flash metadata shape was unified across paged and eager paths

Files:

- `vasr-paged-attn/src/flash.rs`
- `vasr-paged-attn/src/paged_kv_cache.rs`
- `vasr-models/src/model/attention.rs`
- `vasr-models/src/model/thinker_text.rs`

Change:

- Added reusable `FlashParams` / `FlashKMeta` construction helpers.
- `PagedInputMetadata` now carries prebuilt `prefill_flash_params`.
- Eager prefill now also constructs the same general flash-style metadata shape.

Why it matters:

- This is not yet a true speedup by itself.
- It is important because it removes a dataflow blocker for implementing true packed
  varlen prefill later.
- It makes the prefill metadata shape much closer to `mistralrs`.

### 3. `run_attention(...)` was refactored into a clearer dispatch shape

Files:

- `vasr-models/src/model/attention.rs`

Change:

- Split common decisions into helpers such as:
  - causal inference,
  - flash eligibility,
  - SDPA mask preparation.
- Added a separate `run_attention_noflash(...)` entrypoint.

Why it matters:

- This is structural groundwork.
- It makes it much easier to insert a dedicated varlen prefill backend without leaving
  all dispatch logic tangled in one function.

### 4. Batch-local prompt token caching

Files:

- `vasr-models/src/processor/asr_processor.rs`

Change:

- Within one prepare batch, identical prompt strings are tokenized once and reused.

Why it matters:

- This reduces `tokenize_expand_us`.
- It is not a prefill-forward optimization, but it removes repeated prompt-side work in
  VAD-heavy and repeated-context workloads.

### 5. Prepared batch rebatching by effective prompt length before generation

Files:

- `vasr-models/src/inference/transcribe.rs`

Change:

- `PreparedInputs` are regrouped into micro-batches by effective prompt length.
- Each micro-batch is cropped to real token length and then padded only to its own local
  maximum length.
- Output ordering is restored after generation.
- Timed and non-timed transcribe paths were both updated.

Why it matters:

- This is the first landed change in this phase that directly reduces padded prefill work.
- It does not yet eliminate all prefill padding, because the actual eager attention backend
  is still fundamentally dense/padded.
- It does, however, reduce wasted prompt tokens that were previously carried from a wider
  prepare batch into generation.

### 6. VAD path regression fix after rebatching

Files:

- `vasr-models/src/inference/transcribe.rs`

Change:

- Fixed a length-mismatch regression caused by mixed-width rebatching feeding unequal
  `input_ids` rows into a path that still requires equal-width rows per generation call.
- The current implementation now always repads within each micro-batch to a local max.

Why it matters:

- Restores correctness for VAD-enabled evaluation.
- Confirms the current rebatching logic is functioning on real segmented workloads.

### 7. Experimental hybrid paged-prefill to eager-decode handoff

Files:

- `vasr-models/src/model/generation.rs`
- `vasr-models/src/model/kv_cache.rs`

Change:

- Added an experimental Metal-only path that:
  - runs paged batch prefill,
  - reconstructs eager `KVCache` tensors from paged cache blocks,
  - then resumes the existing eager decode loop.
- The path is gated behind `VASR_ENABLE_METAL_PAGED_PREFILL_HYBRID=1`.

Why it matters:

- This is the first experiment in the repo that tries to combine:
  - real varlen/paged prefill savings,
  - with unchanged eager decode kernels.
- It is still experimental and gated behind `VASR_ENABLE_METAL_PAGED_PREFILL_HYBRID=1`.
- The original version exposed correctness problems, but the current restricted version
  now has a validated full-benchmark win.
- In its current form, the hybrid mode deliberately avoids the unsafe Metal paged
  fast-path variants and keeps the decode kernels unchanged.

## What Actually Speeds Up Prefill vs. What Does Not

### Changes that can reduce real prefill work

- Rebatching by effective prompt length before generation.
- Any future packed varlen prefill path using real `cu_seqlens_q/cu_seqlens_kv`.
- Any future Metal backend that computes only on true non-padded Q/K/V tokens.

### Changes that help, but do not reduce real prefill FLOPs

- Prompt token caching.
- Better Q/K/V materialization before dispatch.
- Flash metadata unification.
- Attention dispatch refactors.

These changes reduce overhead or remove future blockers, but they do not by themselves
remove padded attention math.

## What Has NOT Been Achieved Yet

The repo does **not** yet have a full `mistralrs`-style packed varlen eager prefill backend
on Metal.

More specifically:

- Eager prefill still ultimately runs as dense attention over padded batch shapes.
- The existing varlen support is partial and stronger on the paged path than on the normal
  eager prefill path.
- There is not yet a dedicated Metal prefill backend analogous to a true flash-attn-like
  varlen prefill kernel for the default eager path.

This is the main remaining optimization opportunity if the goal is to accelerate prefill
itself rather than merely improving surrounding pipeline overhead.

## Main Technical Conclusion From Current Data

The current numbers show the relevant bottleneck shape under the kept offline path:

### With VAD

- The workload becomes a real multi-item segmented workload.
- Prefill still remains expensive in the `~680-900ms` range for many batches.
- This is the strongest evidence that a true packed varlen prefill backend would be
  worthwhile: these batches have multiple sequences and therefore real padding waste.

## Recommended Next Step

If the goal is a real prefill-speed optimization rather than a parameter workaround,
then the highest-value next step is:

1. implement a true packed varlen prefill attention backend for Metal eager prefill,
2. using the already-landed `FlashParams` / `cu_seqlens_q` / `cu_seqlens_kv` dataflow,
3. and keep dense Metal SDPA as the fallback path.

That would be the first change in this repo that can plausibly deliver a large prefill
speedup without depending on smaller chunk sizes or other workload-shaping shortcuts.
