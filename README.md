# Qwen3ASR.rs

Python bindings for Qwen3-ASR inference using Rust and Candle.

The first development milestone provides the Python package shape, device/dtype validation, and a stable API for the upcoming Candle inference port.

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install maturin pytest
maturin develop
pytest
cargo test --workspace
```

## Python API

```python
from qwen3_asr_rs import Qwen3ASR

model = Qwen3ASR.from_pretrained("Qwen/Qwen3-ASR-0.6B", device="auto")
result = model.transcribe("audio.wav", language="Chinese")
print(result.text)
```

Real Candle inference is being ported from `lumosimmo/qwen3-asr-rs` in the next milestone.
