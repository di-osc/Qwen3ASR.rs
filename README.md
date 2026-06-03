# vASR

vASR is a Rust speech inference framework built around a timeline data model.

The current implementation keeps the existing Qwen3-ASR Candle runtime and wraps it as an
`AsrModel`. New framework crates define audio loading, model capabilities, fasr-compatible
protocol DTOs, and HTTP/WebSocket service entry points.

## Crates

- `vasr-data`: `Timeline`, `Annotation`, `Waveform`, `AudioChunk`, `Transcript`.
- `vasr-audio`: audio source loading into normalized 16 kHz mono waveforms.
- `vasr-runtime`: model capability traits, Qwen3-ASR ASR model wrapper, Silero ONNX VAD, offline/realtime pipelines.
- `vasr-protocol`: fasr-compatible transcribe and realtime DTOs.
- `vasr-server`: axum routes for `/transcribe`, `/inference`, `/v1/realtime`, and `/api-ws/v1/realtime`.
- `vasr-models`: Candle Qwen3-ASR model implementation.

## Development

```bash
cargo test --workspace
```

Build with Apple Metal support:

```bash
cargo test --workspace --features vasr-runtime/metal
```

Build with CUDA support:

```bash
cargo test --workspace --features vasr-runtime/cuda
```

## Serve

Start a CPU service:

```bash
cargo run -p vasr-cli --bin vasr -- serve \
  --model /path/to/Qwen3-ASR-0.6B \
  --vad-model /path/to/silero_vad.onnx \
  --host 127.0.0.1 \
  --port 8000 \
  --device cpu \
  --dtype bf16
```

Start a Metal service:

```bash
cargo run -p vasr-cli --bin vasr --features metal-paged-attn -- serve \
  --model /path/to/Qwen3-ASR-0.6B \
  --host 127.0.0.1 \
  --port 8000 \
  --device metal \
  --dtype bf16 \
  --isq q8_0
```

Build a release binary:

```bash
cargo build --release -p vasr-cli --bin vasr --features metal-paged-attn
./target/release/vasr serve --model /path/to/Qwen3-ASR-0.6B --device metal --dtype bf16 --isq q8_0
```

Silero VAD is enabled by default for offline `/transcribe` annotations and realtime speech
events. Pass `--vad-model /path/to/silero_vad.onnx` to choose an ONNX model explicitly, or
`--no-vad` to disable offline VAD annotations. When `--vad-model` is omitted, vASR searches
common local Silero cache locations under `$HOME/.cache`.

The default Qwen3-ASR weight dtype is `bf16`, including Metal builds. Use `--dtype f16`
as a fallback if a future model path hits a backend kernel limitation.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Offline transcribe:

```bash
curl -X POST http://127.0.0.1:8000/transcribe \
  -H 'content-type: application/json' \
  -d '{
    "inputs": [
      {
        "url": "file:///absolute/path/audio.wav",
        "mono": true,
        "hotword": "OpenAI fasr"
      }
    ]
  }'
```

Realtime WebSocket endpoints:

```text
ws://127.0.0.1:8000/v1/realtime
ws://127.0.0.1:8000/api-ws/v1/realtime
```

The realtime API accepts fasr-style JSON events such as `session.update`,
`input_audio_buffer.append`, `input_audio_buffer.commit`, and `session.finish`.

## Architecture

Offline flow:

```text
TranscribeRequest
  -> vasr-audio AudioLoader
  -> Waveform
  -> vasr-runtime OfflinePipeline
  -> Timeline + Annotation
  -> vasr-protocol TranscribeResponse
```

Realtime flow:

```text
WebSocket PCM16 base64 events
  -> AudioBytesStream
  -> AudioChunk
  -> RealtimePipeline
  -> ServerRealtimeEvent
```
