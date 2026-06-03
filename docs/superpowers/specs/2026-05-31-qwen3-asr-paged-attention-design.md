# Qwen3-ASR Paged Attention Design

## Goal

Use the core inference architecture from `guoqingbao/xinfer` to speed up Qwen3-ASR decoding on Metal, while keeping the existing ASR API, audio encoder, mRoPE behavior, and CPU fallback intact.

## Scope

The first implementation targets Metal decode speed. CPU keeps the current eager attention and dynamic KV cache path as a correctness fallback. CUDA and FlashInfer support are out of scope for this phase, but the interfaces should keep backend-specific logic isolated.

This is not a full `xinfer` engine import. The project will borrow the parts that matter for ASR decode speed:

- `attention-rs` powered paged attention.
- Preallocated, block-addressed KV cache storage.
- `InputMetadata` style prefill/decode metadata.
- A model-side fast path that avoids per-token `Tensor::cat` on KV tensors.

The `xinfer` scheduler, OpenAI-compatible server, prefix-cache eviction policy, tensor parallelism, CUDA graph capture, and request batching engine stay out of scope.

## Architecture

Add a new runtime module for paged decoding rather than replacing the existing generation code in place. The old path remains the fallback and test oracle.

The new path will have four pieces:

1. `PagedKvCache`: owns preallocated per-layer key/value page tensors and block tables.
2. `PagedDecodeMetadata`: builds the slot mapping, block table, context lengths, and position tensors needed by `attention-rs`.
3. `ThinkerTextModel::forward_*_paged`: mirrors the existing text model forward path but sends Q/K/V through paged attention when the device and feature flags support it.
4. `greedy_generate_paged`: generation loop that does prompt prefill, then single-token decode without growing KV tensors by concatenation.

The ASR pipeline continues to prepare input IDs, attention masks, audio features, and mRoPE positions exactly as it does today. Only the text decoder's KV storage and attention execution change.

## Data Flow

Prefill:

1. Processor prepares prompt tokens and audio placeholder expansion.
2. Audio encoder produces audio features.
3. Text embeddings merge audio features into placeholder positions.
4. Paged path allocates enough KV pages for `prompt_len + max_new_tokens`.
5. Prefill writes prompt K/V into paged cache and returns logits for the last prompt token.

Decode:

1. The generation loop embeds the last token.
2. Metadata maps the token to the next cache slot.
3. Each decoder layer writes one new K/V entry into the paged cache.
4. `attention-rs` reads K/V through block tables instead of concatenated tensors.
5. The loop greedily selects the next token until EOS or `max_new_tokens`.

For left-padded batch inputs, phase 1 uses the old path. The first optimization target is the common ASR batch size 1 path with a dense attention mask.

## Dependencies

Add `attention-rs` as an optional dependency under a new feature such as `paged-attn`.

Metal feature wiring should look conceptually like:

```toml
paged-attn = ["dep:attention-rs"]
metal-paged-attn = ["metal", "paged-attn", "attention-rs/metal", "attention-rs/metal-flash"]
```

The exact version must be resolved against the current Candle 0.9 dependency. If `attention-rs` only compiles against the `xinfer` Candle fork, the implementation should stop at a compatibility shim and report that the dependency graph must either move to the fork or use a local adapter.

## Error Handling

The paged path is opt-in and must fail closed:

- If the feature is disabled, use the existing generation path.
- If the device is not Metal, use the existing generation path.
- If cache allocation is too small, return a clear error with required tokens and configured capacity.
- If `attention-rs` rejects a shape, include the layer index, batch size, context length, and head dimensions.
- If numerical parity fails in tests, keep the paged path disabled by default.

## Testing

The old dynamic-cache path remains the parity oracle.

Required tests:

- Unit tests for block allocation, slot mapping, and block table generation.
- Unit tests for prompt length and generated token position metadata.
- A small model-level parity test that compares logits from dynamic cache vs paged cache on CPU-compatible tensors when possible.
- A Metal fixture transcription test comparing output text before and after enabling paged decode.
- A speed smoke test that reports prefill/decode timing but does not assert a fixed percentage.

## Success Criteria

The phase is complete when:

- `cargo test --workspace` passes.
- `cargo test -p vasr-models-qwen3-asr --features metal-paged-attn` passes on Metal.
- `maturin develop --features metal,metal-paged-attn` builds.
- The fixture `fixtures/audio/asr_en_16k.wav` transcribes successfully with the same text as the fallback path.
- Warm Metal inference is measurably faster than the fallback on the same fixture.

## Non-Goals

- Full `xinfer` server or scheduler integration.
- Prefix cache across independent ASR requests.
- Tensor parallelism.
- CUDA graph capture.
- Quantized KV cache.
- Replacing the audio encoder.
