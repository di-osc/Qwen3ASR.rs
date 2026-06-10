# Qwen3-ASR Metal ISQ8 Notes

Date: 2026-06-10

This file is the compact Metal ISQ8 reference for the current implementation.
Old pass-by-pass logs, rejected experiments, and stale one-off measurements were
removed. Use this document for the current expected behavior and verification
commands.

## Current Behavior

- `--isq 8` on Metal resolves to AFQ8 through the existing automatic selection
  path.
- The text decoder uses quantized linear layers where AFQ8 is supported.
- The audio encoder is not quantized by default. A previous audio-encoder ISQ
  attempt was slower because those matrices are smaller and the quant/dequant
  overhead dominated.
- Metal defaults to the eager KV-cache decode path. Paged attention is still
  available for regression/experiments through `VASR_FORCE_PAGED_ATTN=1`.
- seq_len==1 mRoPE decode now uses the accelerated Metal/CUDA rotary kernel and
  first-modality semantics.

## Current Correctness Repro

The minimal ASR correctness repro for the mRoPE/AFQ8 path is:

```bash
VASR_DEBUG_RAW_ASR=1 \
VASR_LOG='warn,vasr_transcribe=info,vasr_models::inference::transcribe=debug,vasr_quant::isq_linear=warn' \
cargo run -q -p vasr-transcribe --features metal -- run \
  --input asr_example.wav \
  --output /tmp/vasr-metal-accel-rope \
  --isq 8 \
  --limit 1
```

Expected raw ASR content:

```text
language Chinese<asr_text>甚至出现交易几乎停滞的情况。
```

The run should also log:

```text
ISQ selected is afq8 (requested=8, backend=metal).
```

## Decode Benchmark

Single-sequence text decode command:

```bash
cargo run --release -p vasr-models --example bench_text_decode \
  --features "metal,metal-paged-attn,paged-attn,timing" \
  -- Qwen/Qwen3-ASR-0.6B "The capital of France is" 3 64 bf16 8 --warmup 1
```

Most recent clean reference from this optimization series:

| Runtime | Mode | Decode speed |
| --- | --- | ---: |
| vASR | BF16 + AFQ8 + eager Metal decode | ~236-238 tok/s |
| mistral.rs | `--isq 8`, prompt-len 0 | ~253 tok/s |
| vASR | forced paged Metal decode | ~170-175 tok/s |

Batch decode command:

```bash
cargo run --release -p vasr-models --example bench_text_decode \
  --features "metal,metal-paged-attn,paged-attn,timing" \
  -- Qwen/Qwen3-ASR-0.6B "The capital of France is" 3 64 bf16 8 --batch 2 --warmup 1
```

Reference scaling from the same pass:

| batch | batch tok/s | per-seq tok/s |
| ---: | ---: | ---: |
| 1 | ~238 | ~238 |
| 2 | ~450 | ~225 |
| 4 | ~658 | ~165 |

Batch throughput scales because the Metal GPU parallelizes matmul, SDPA, and
RoPE across rows inside each decode step. Per-sequence throughput drops as
argmax/readback work grows with batch size.

## Important Landed Optimizations

### Decode-One QKV Shape

For `seq_len == 1`, Q/K/V are shaped directly as:

- Q: `(batch, num_attention_heads, 1, head_dim)`
- K/V: `(batch, num_key_value_heads, 1, head_dim)`

This avoids the slower `(batch, seq, heads, dim) -> transpose(1, 2)` path on
every decode step.

### Accelerated mRoPE Decode

For seq_len==1 decode:

- only the first mRoPE modality is used,
- the accelerated `apply_rotary_qk` kernel is used on Metal/CUDA,
- the accelerated path uses NeoX/rotate-half semantics to match the generic
  Qwen3-ASR decode path.

This fixed the earlier wrong-output path while keeping the Metal rotary speedup.

### Eager Metal Decode Default

Metal now defaults to the eager KV-cache decode route. Forced paged attention is
kept as an opt-in regression path:

```bash
VASR_FORCE_PAGED_ATTN=1 cargo run --release -p vasr-models --example bench_text_decode \
  --features "metal,metal-paged-attn,paged-attn,timing" \
  -- Qwen/Qwen3-ASR-0.6B "The capital of France is" 3 64 bf16 8 --warmup 1
```

### Audio Encoder BF16 Workaround

Metal BF16 workarounds now keep intermediate conversions on device where
possible instead of round-tripping through CPU. This is a hygiene improvement
for end-to-end ASR, not the main speed lever.

## What Was Removed From This Doc

- Historical pass logs that no longer describe the current code.
- Failed paged-attention and SDPA experiments.
- One-token decode isolation numbers.
- Removed accelerated-RoPE feature-switch descriptions.
- Non-reproduced or incomplete-output timing claims.

## Verification

```bash
cargo test -p vasr-models --features metal-paged-attn test_metal_accelerated_seq_one -- --nocapture
cargo test -p vasr-models test_seq_one_first_modality_matches_standard_when_positions_match -- --nocapture
cargo check -p vasr-transcribe --features metal
```

For end-to-end validation, use the `raw_audios` baseline documented in
`docs/performance/qwen3-asr-metal-prefill-optimization-status.md`.

## Remaining Work

1. Reduce decode argmax/readback synchronization overhead.
2. Consider fusing Q/K norm with mRoPE for seq_len==1.
3. Implement true packed varlen prefill for the default Metal eager path.
4. Keep AFQ8 automatic selection unchanged while validating any quantization
   changes against full ASR output.
