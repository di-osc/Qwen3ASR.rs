# Qwen3-ASR Metal Prefill Status

Date: 2026-06-10

This note keeps only the current, reproducible Metal prefill status for
`Qwen/Qwen3-ASR-0.6B`. Older one-off experiments and non-reproduced fast numbers
were removed so this file can be used as a practical baseline.

## Machine

- CPU/GPU: Apple M4 Pro, 20 GPU cores
- Unified memory: 48 GiB
- Backend: Metal
- Runtime dtype: BF16
- Quantization: `--isq 8`, resolved to AFQ8 on Metal

## Current Reproducible Baseline

Use this command for the current complete-recognition Metal baseline on
`raw_audios`:

```bash
VASR_LOG='info,vasr_models::inference::transcribe=info' \
./target/release/vasr-transcribe run \
  --device metal \
  --input raw_audios \
  --output /tmp/vasr-repro-metal-default-clean-info \
  --isq 8 \
  --max-batch-audio-sec 60 \
  --max-new-tokens 128 \
  --limit 20
```

Observed result:

```text
Done: files=20 bad=0 audio_seconds=1652.085 wall_seconds=27.610 speedup=59.836 rtf=0.0167
```

Correctness sanity from that run:

- `ISQ selected is afq8 (requested=8, backend=metal)`.
- 20/20 files returned.
- Output scan found no `퓮`, replacement character, `<asr_text>`, or `<|...|>`
  markers.
- Output contained 120 transcript text fields, 111 non-empty, with 2173 total
  non-whitespace transcript characters.

## Current Comparison Points

| Path | Difference | Wall | Status |
| --- | --- | ---: | --- |
| Default eager | `--max-batch-audio-sec 60` | **27.610s** | Fastest complete run reproduced on 2026-06-10 |
| Hybrid prefill | `VASR_ENABLE_METAL_PAGED_PREFILL_HYBRID=1` | 29.007s | Complete output, but slower after the mRoPE fix |
| Larger micro-batches | `--max-batch-audio-sec 180` | 46.162s | Slower; larger batches increase decode argmax cost |

Older hybrid-prefill numbers are intentionally not listed as current results.
After the seq_len==1 mRoPE fix they did not reproduce as the fastest complete
path, and at least one older run generated fewer tokens than current
complete-recognition runs.

## Landed Changes That Matter

- Effective prompt-length rebatching before generation reduces padded prefill
  work per micro-batch.
- VAD speech segments are padded with ASR context, but full-coverage VAD cases
  now run the full waveform instead of slicing away useful context.
- ASR worker micro-batching now preserves pending jobs when segment and raw-job
  messages interleave.
- Default offline ASR options are aligned with the validated baseline:
  `max_new_tokens = 128` and `max_batch_audio_sec = 60`.
- The hybrid paged-prefill to eager-decode handoff remains experimental and is
  not the current default.

## Current Bottleneck

The current complete path is still dominated by generation work:

- Prefill remains expensive because the default eager path is dense over padded
  batch shapes.
- Decode has meaningful overhead from token selection / host-visible argmax,
  especially as the number of concurrent rows grows.
- Reducing `max_batch_audio_sec` below 60 can reduce some per-batch prefill
  pressure, but that is a workload-shaping tradeoff rather than a backend fix.

## Next Useful Work

1. Implement a true packed varlen prefill backend for the Metal eager path.
2. Keep dense Metal SDPA as the fallback path.
3. Reduce decode argmax synchronization/readback overhead.
4. Re-run `raw_audios` after each change and treat only complete, marker-free
   output as a valid performance number.
