# Qwen3ASR.rs

Python bindings for Qwen3-ASR inference using Rust and Candle.

The package contains its own Rust/Candle runtime crate and exposes a small Python API.

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install maturin pytest
maturin develop
pytest
cargo test --workspace
```

Build with Apple Metal support:

```bash
maturin develop --features metal
```

Build with CUDA support:

```bash
maturin develop --features cuda
```

## Python API

```python
from qwen3_asr_rs import Qwen3ASR

model = Qwen3ASR.from_pretrained("Qwen/Qwen3-ASR-0.6B", device="auto")
result = model.transcribe("audio.wav", language="Chinese")
print(result.text)
```

`device` accepts `auto`, `cpu`, `metal`, or `cuda`. The first real run downloads model files through Hugging Face cache if you pass a model id.
