use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use base64::Engine;
use clap::Args;
use serde::Serialize;
use vasr_data::{
    AudioAsset, AudioSource, CerStats, VasrRecord, VasrRecordList, compute_cer, normalize_for_cer,
};
use vasr_protocol::InferencePerformance;
use vasr_server::TranscribeInput;

use crate::serve::{TranscribePipelineArgs, build_async_transcribe_pipeline, validate_pipeline};

#[derive(Debug, Clone, Args)]
pub struct BenchmarkTranscribeArgs {
    #[command(flatten)]
    pub pipeline: TranscribePipelineArgs,

    /// `VasrRecordList` MessagePack file.
    #[arg(short, long, value_name = "PATH")]
    pub input: PathBuf,

    /// Optional JSON report output path.
    #[arg(short, long, value_name = "PATH")]
    pub output: Option<PathBuf>,

    /// Optional ASR language hint.
    #[arg(long, env = "VASR_LANGUAGE")]
    pub language: Option<String>,

    /// Process at most this many records.
    #[arg(long)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkCerSummary {
    pub num_records: usize,
    pub num_evaluated: usize,
    pub num_failed: usize,
    pub num_skipped_empty_reference: usize,
    pub macro_cer: f64,
    pub micro_cer: f64,
    pub total_substitutions: usize,
    pub total_deletions: usize,
    pub total_insertions: usize,
    pub total_reference_chars: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkCerItem {
    pub index: usize,
    pub record_id: String,
    pub reference: String,
    pub hypothesis: String,
    pub cer: Option<f64>,
    pub substitutions: usize,
    pub deletions: usize,
    pub insertions: usize,
    pub reference_chars: usize,
    pub is_bad: bool,
    pub bad_reason: Option<String>,
    pub bad_component: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkCerReport {
    pub summary: BenchmarkCerSummary,
    pub items: Vec<BenchmarkCerItem>,
    pub inference_performance: InferencePerformance,
}

pub async fn run_benchmark(args: BenchmarkTranscribeArgs) -> Result<()> {
    validate_pipeline(&args.pipeline)?;
    if !args.input.exists() {
        bail!("input path does not exist: {}", args.input.display());
    }

    let record_list = VasrRecordList::read_msgpack(&args.input).with_context(|| {
        format!(
            "failed to read VasrRecordList from {}",
            args.input.display()
        )
    })?;
    if record_list.is_empty() {
        bail!("record list is empty: {}", args.input.display());
    }

    let mut records = record_list.records;
    if let Some(limit) = args.limit {
        records.truncate(limit);
    }

    let pipeline = build_async_transcribe_pipeline(&args.pipeline, args.language.clone())?;
    let mut inputs = Vec::with_capacity(records.len());
    let mut record_ids = Vec::with_capacity(records.len());
    let mut references = Vec::with_capacity(records.len());

    for (index, record) in records.iter().enumerate() {
        let (source, record_id) = audio_source_from_record(record)?;
        references.push(reference_text(record));
        record_ids.push(record_id);
        inputs.push(TranscribeInput { index, source });
    }

    tracing::info!(
        target: "vasr_cli::benchmark",
        "Benchmarking {} record(s) from `{}`.",
        records.len(),
        args.input.display()
    );

    let batch_start = Instant::now();
    let outcomes = pipeline.transcribe_many(inputs).await;
    let batch_wall = batch_start.elapsed().as_secs_f64();

    let mut items = Vec::with_capacity(records.len());
    let mut total_audio_seconds = 0.0;
    let mut total_substitutions = 0usize;
    let mut total_deletions = 0usize;
    let mut total_insertions = 0usize;
    let mut total_reference_chars = 0usize;
    let mut macro_cer_sum = 0.0;
    let mut num_evaluated = 0usize;
    let mut num_failed = 0usize;
    let mut num_skipped_empty_reference = 0usize;

    for (index, ((record_id, reference), outcome)) in record_ids
        .into_iter()
        .zip(references)
        .zip(outcomes)
        .enumerate()
    {
        total_audio_seconds += outcome.audio_seconds;
        let normalized_reference = normalize_for_cer(&reference, true);

        let (hypothesis, is_bad, bad_reason, bad_component) = match &outcome.result {
            Ok(timeline) => {
                let hypothesis = timeline.transcript().text;
                (hypothesis, false, None, None)
            }
            Err(error) => (
                String::new(),
                true,
                Some(error.to_string()),
                outcome.bad_component.map(str::to_string),
            ),
        };
        if is_bad {
            num_failed += 1;
        }

        let normalized_hypothesis = normalize_for_cer(&hypothesis, true);
        let (cer, stats) = if normalized_reference.is_empty() {
            num_skipped_empty_reference += 1;
            (None, CerStats::default())
        } else if is_bad {
            (None, CerStats::default())
        } else {
            let stats = compute_cer(&normalized_reference, &normalized_hypothesis);
            total_substitutions += stats.substitutions;
            total_deletions += stats.deletions;
            total_insertions += stats.insertions;
            total_reference_chars += stats.reference_chars;
            macro_cer_sum += stats.cer();
            num_evaluated += 1;
            (Some(stats.cer()), stats)
        };

        items.push(BenchmarkCerItem {
            index,
            record_id,
            reference,
            hypothesis,
            cer,
            substitutions: stats.substitutions,
            deletions: stats.deletions,
            insertions: stats.insertions,
            reference_chars: stats.reference_chars,
            is_bad,
            bad_reason,
            bad_component,
        });
    }

    let macro_cer = if num_evaluated > 0 {
        macro_cer_sum / num_evaluated as f64
    } else {
        0.0
    };
    let micro_cer = if total_reference_chars > 0 {
        (total_substitutions + total_deletions + total_insertions) as f64
            / total_reference_chars as f64
    } else {
        0.0
    };

    let report = BenchmarkCerReport {
        summary: BenchmarkCerSummary {
            num_records: records.len(),
            num_evaluated,
            num_failed,
            num_skipped_empty_reference,
            macro_cer,
            micro_cer,
            total_substitutions,
            total_deletions,
            total_insertions,
            total_reference_chars,
        },
        items,
        inference_performance: InferencePerformance {
            batch_wall_seconds: batch_wall,
            num_items: records.len(),
            throughput_items_per_second: records.len() as f64 / batch_wall.max(f64::EPSILON),
            total_audio_duration_seconds: total_audio_seconds,
            speedup: total_audio_seconds / batch_wall.max(f64::EPSILON),
            rtf: batch_wall / total_audio_seconds.max(f64::EPSILON),
        },
    };

    if let Some(output) = &args.output {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory {}", parent.display())
            })?;
        }
        let file = fs::File::create(output)
            .with_context(|| format!("failed to create output file {}", output.display()))?;
        serde_json::to_writer_pretty(&file, &report)
            .with_context(|| format!("failed to write JSON to {}", output.display()))?;
        tracing::info!(
            target: "vasr_cli::benchmark",
            "Wrote benchmark report to `{}`.",
            output.display()
        );
    }

    let speedup = total_audio_seconds / batch_wall.max(f64::EPSILON);
    let rtf = batch_wall / total_audio_seconds.max(f64::EPSILON);
    tracing::info!(
        target: "vasr_cli::benchmark",
        "Done: records={} evaluated={} failed={} skipped_empty_ref={} macro_cer={:.4} micro_cer={:.4} audio_seconds={:.3} wall_seconds={:.3} speedup={:.3} rtf={:.4}",
        report.summary.num_records,
        report.summary.num_evaluated,
        report.summary.num_failed,
        report.summary.num_skipped_empty_reference,
        report.summary.macro_cer,
        report.summary.micro_cer,
        total_audio_seconds,
        batch_wall,
        speedup,
        rtf
    );

    if num_failed > 0 {
        bail!("{num_failed} of {} transcription(s) failed", records.len());
    }
    Ok(())
}

fn reference_text(record: &VasrRecord) -> String {
    record.timeline.transcript().text
}

fn audio_source_from_record(record: &VasrRecord) -> Result<(AudioSource, String)> {
    let record_id = record.record_id();
    if let Some(cache) = &record.waveform_cache {
        return Ok((AudioSource::Waveform(cache.waveform.clone()), record_id));
    }

    match &record.media {
        AudioAsset::Uri { uri, .. } => {
            if uri.starts_with("http://") || uri.starts_with("https://") {
                Ok((AudioSource::Url(uri.clone()), record_id))
            } else if let Some(path) = local_path_from_uri(uri) {
                Ok((AudioSource::Path(path), record_id))
            } else {
                Ok((AudioSource::Url(uri.clone()), record_id))
            }
        }
        AudioAsset::Embedded { bytes, .. } => {
            let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
            Ok((AudioSource::Base64(encoded), record_id))
        }
    }
}

fn local_path_from_uri(uri: &str) -> Option<PathBuf> {
    if let Some(rest) = uri.strip_prefix("file://") {
        return Some(PathBuf::from(percent_decode_path(rest)));
    }
    if uri.starts_with("http://") || uri.starts_with("https://") {
        return None;
    }
    Some(PathBuf::from(uri))
}

fn percent_decode_path(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_value(bytes[i + 1]), hex_value(bytes[i + 2])) {
                out.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{audio_source_from_record, local_path_from_uri, reference_text};
    use vasr_data::{
        Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, AudioAsset,
        AudioEncoding, PersistedAudioFormat, TextSegment, TimeRange, Timeline, VasrRecord,
        Waveform,
    };

    #[test]
    fn reference_text_uses_final_timeline_segments() {
        let mut timeline = Timeline::new("audio_1");
        timeline.push(Annotation::new(
            TimeRange::new(vasr_data::DurationMs(0), vasr_data::DurationMs(100)),
            AnnotationPayload::Segment(TextSegment::new("hello")),
            AnnotationSource::Model("asr".to_string()),
            AnnotationStatus::Final,
        ));
        let record = VasrRecord::new(
            AudioAsset::Embedded {
                bytes: vec![1, 2, 3],
                format: PersistedAudioFormat {
                    sample_rate: Some(16_000),
                    channels: Some(1),
                    encoding: AudioEncoding::Wav,
                },
                duration: None,
                sha256: None,
            },
            timeline,
        );
        assert_eq!(reference_text(&record), "hello");
    }

    #[test]
    fn audio_source_prefers_waveform_cache() {
        let timeline = Timeline::new("audio_1");
        let waveform = Waveform::from_i16_pcm(&[0, 1000, -1000], 16_000);
        let record = VasrRecord::new(
            AudioAsset::Uri {
                uri: "file:///tmp/audio.wav".to_string(),
                format: PersistedAudioFormat {
                    sample_rate: Some(16_000),
                    channels: Some(1),
                    encoding: AudioEncoding::Wav,
                },
                duration: None,
                sha256: None,
            },
            timeline,
        )
        .with_waveform_cache(vasr_data::WaveformCache {
            waveform: waveform.clone(),
            source: "cache".to_string(),
        });
        let (source, _) = audio_source_from_record(&record).expect("audio source");
        assert_eq!(source, vasr_data::AudioSource::Waveform(waveform));
    }

    #[test]
    fn local_path_from_uri_supports_file_urls() {
        assert_eq!(
            local_path_from_uri("file:///tmp/audio.wav").as_deref(),
            Some(std::path::Path::new("/tmp/audio.wav"))
        );
    }
}
