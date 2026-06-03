# vASR Data Timeline Annotation Design

## Goal

`vasr-data` defines the stable data model for vASR. It should model audio and ASR outputs as time-aligned data, not as a copy of fasr's Python object graph.

## Core Decision

The canonical internal representation is `Timeline + Annotation`.

`Timeline` represents one media item and contains tracks plus annotations. `Annotation` represents one time-bounded fact about that media item: speech, silence, token, segment, speaker, language, hotword, acoustic event, or diagnostic information.

`Transcript` is a derived view over a timeline. Fasr-compatible HTTP and websocket schemas belong in a later protocol layer, not in `vasr-data`.

## Data Boundaries

`vasr-data` owns:

- `time`: strong time types such as `DurationMs`, `SampleIndex`, and `TimeRange`.
- `media`: identifiers, channels, audio format, and audio sources.
- `waveform`: decoded PCM waveform data.
- `token`: token-level recognition output.
- `segment`: text segments and transcript views.
- `timeline`: annotations and timelines.
- `stream`: PCM16 byte buffering into fixed-size audio chunks.

`vasr-data` does not own:

- Candle tensors or model internals.
- HTTP response field names.
- Fasr compatibility typos such as `bad_componet`.
- Pipeline timing/performance fields.

## Type Shape

`TimeRange` uses `DurationMs` fields instead of naked `u64` millisecond values.

`Waveform` stores decoded PCM samples, sample rate, and channel count. It does not store transcript results.

`AudioChunk` carries stream chunks with a `TimeRange`, start/final flags, and waveform samples.

`Annotation` stores an id, time range, source, status, confidence, and `AnnotationPayload`.

`AnnotationPayload` supports:

- `Speech`
- `Silence`
- `Token(Token)`
- `Segment(TextSegment)`
- `Sentence(TextSegment)`
- `Speaker(SpeakerId)`
- `Language(LanguageTag)`
- `Hotword(HotwordMatch)`
- `AcousticEvent(AcousticEvent)`
- `Diagnostic(Diagnostic)`

## Derived Views

`Timeline::transcript()` collects final `Segment` and `Sentence` annotations in time order and returns a simple `Transcript`.

`Transcript` is a convenience view for common ASR users. It is not the source of truth.

## Compatibility Strategy

The current `AudioSpan`, `AudioChannel`, and `Audio` names are transitional. They should be replaced by `TimeRange`, `TextSegment`, `Transcript`, `Timeline`, and `Annotation`.

Future fasr-style schemas should convert from `Timeline` and `Transcript` into protocol DTOs.

## Testing

Tests should cover:

- Time range duration and ordering.
- Waveform PCM16 roundtrip.
- PCM stream chunking with correct time ranges.
- Timeline annotation filtering.
- Transcript derivation from final text annotations only.
