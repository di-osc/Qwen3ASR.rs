# Qwen3-ASR Python Package Design

## Goal

Build a Python package backed by Rust and Candle that can run Qwen3-ASR offline transcription locally. The first version prioritizes a clean Python API and local developer installation; prebuilt wheels, streaming, HTTP serving, and forced-alignment timestamps are intentionally left for later phases.

## Reference

Use `lumosimmo/qwen3-asr-rs` as the main implementation reference for model architecture, audio preprocessing, token placeholder expansion, safetensors loading, and generation flow. The new project should copy the shape of the proven pipeline while keeping the first version much smaller.

The reference project demonstrates these important boundaries:

- `audio`: decode, normalize, resample, chunking.
- `processor`: tokenizer, chat template, feature extraction, audio placeholder expansion.
- `model`: Qwen3-ASR thinker, audio tower, text model, RoPE, KV cache, safetensors loading.
- `inference`: offline transcription, streaming, output parsing.
- `cli`: command-line and serving tools.

The first version of this project should keep only the pieces required for offline transcription and Python binding.

## Scope

In scope for the first implementation:

- A Rust workspace with a core library crate and a PyO3 extension crate.
- A Python package installable through `maturin develop` and buildable through `maturin build`.
- Offline transcription from a local audio file path.
- Loading a model from a HuggingFace model id or local model directory.
- Device selection through Python: `auto`, `cpu`, `cuda`, and `metal`.
- Default dtype selection suitable for local development, with an explicit dtype parameter available.
- A Python API centered on `Qwen3ASR.from_pretrained(...)` and `model.transcribe(...)`.
- Focused unit and smoke tests that do not require downloading the full model by default.

Out of scope for the first implementation:

- Streaming transcription.
- Forced aligner and word timestamps.
- HTTP server or OpenAI-compatible endpoints.
- Browser demo.
- Benchmark matrix and regression gate scripts.
- Publishing prebuilt wheels to PyPI.
- Full parity fixture suite.

## User-Facing API

The first Python API should look like this:

```python
from qwen3_asr_rs import Qwen3ASR

model = Qwen3ASR.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    device="auto",
    dtype="auto",
)

result = model.transcribe("audio.wav", language="Chinese")
print(result.text)
print(result.language)
```

`Qwen3ASR.from_pretrained` accepts:

- `model_id_or_path: str`
- `device: str = "auto"`
- `dtype: str = "auto"`
- `use_flash_attn: bool = False`

`Qwen3ASR.transcribe` accepts:

- `audio: str`
- `language: str | None = None`
- `context: str = ""`
- `max_new_tokens: int = 256`

`TranscriptionResult` exposes:

- `text: str`
- `language: str | None`
- `raw: str`

## Architecture

The workspace has two crates:

- `qwen3_asr_core`: pure Rust library for device selection, audio input, model loading, processor logic, and transcription.
- `qwen3_asr_py`: PyO3 extension module named `qwen3_asr_rs`, wrapping the core library.

The Python package is built with `maturin`. The root `pyproject.toml` points at the PyO3 crate and includes the Python package metadata.

The core crate should be organized for later growth:

- `src/lib.rs`: public exports.
- `src/error.rs`: common error type aliases and context helpers.
- `src/device.rs`: parse Python-facing device and dtype strings into Candle values.
- `src/audio.rs`: first-pass audio file loading abstraction. It may start narrow and grow into decode/resample modules.
- `src/model.rs`: model-loading facade. The first implementation can use a lightweight stub while the crate skeleton and tests settle, then replace it with the Candle implementation ported from the reference project.
- `src/transcribe.rs`: public transcription request and result types.

The PyO3 crate should be thin. It should not contain inference logic; it only validates Python arguments, calls `qwen3_asr_core`, and converts errors/results.

## Device Strategy

`device="auto"` chooses the best available backend in this order:

1. CUDA when the build has the `cuda` feature and CUDA device 0 is available.
2. Metal when the build has the `metal` feature and a Metal device is available.
3. CPU.

`device="cuda"` and `device="metal"` fail clearly if the current build does not include the required feature or the backend cannot be initialized.

`dtype="auto"` maps to:

- `f16` for CUDA or Metal.
- `f32` for CPU.

Explicit dtype values are `f32`, `f16`, and `bf16`. Unsupported values fail before model loading.

## Implementation Phases

Phase 1 creates the package skeleton and stable Python API with tested argument validation. It may include a placeholder transcription backend that returns a clear runtime error until real model loading is wired.

Phase 2 ports the minimal Candle model-loading and offline transcription path from `lumosimmo/qwen3-asr-rs`, including config parsing, tokenizer, feature extraction, safetensors loading, audio placeholder expansion, audio encoder, text decoder, KV cache, and greedy generation.

Phase 3 adds integration tests with a real Qwen3-ASR model behind an ignored or opt-in test flag.

Phase 4 adds packaging automation and platform-specific wheel builds.

## Error Handling

Rust code should use `anyhow::Result` internally and attach context at I/O, parsing, model loading, and inference boundaries.

Python-facing errors should be converted to `PyRuntimeError` with actionable messages. Examples:

- Unknown device string.
- CUDA requested but the extension was built without CUDA support.
- Metal requested but the extension was built without Metal support.
- Model config or safetensors file missing.
- Audio path does not exist or cannot be decoded.

## Tests

Default tests must be fast and offline:

- Rust tests for device string parsing and dtype selection.
- Rust tests for transcription option validation.
- Python tests for importing the package, constructing option objects, and clear error messages.

Heavy tests should be opt-in:

- Download or use a local `Qwen/Qwen3-ASR-0.6B` checkpoint.
- Run a fixture wav file through `Qwen3ASR.transcribe`.
- Assert the output contains expected text.

## Acceptance Criteria

The first implementation is acceptable when:

- `cargo test --workspace` passes.
- `python -m pytest` passes after `maturin develop`.
- `python -c "import qwen3_asr_rs; print(qwen3_asr_rs.__version__)"` works.
- The Python API shape is stable enough for the real Candle inference port.
- Unsupported devices and dtypes fail with clear messages.

