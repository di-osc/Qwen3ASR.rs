use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Args;
use vasr_data::AudioSource;
use vasr_protocol::{InferenceData, InferencePerformance, TranscribeResponse};
use vasr_server::{TranscribeInput, TranscribeItemOutcome};

use crate::serve::{TranscribePipelineArgs, build_async_transcribe_pipeline, validate_pipeline};

#[derive(Debug, Clone, Args)]
pub struct RunTranscribeArgs {
    #[command(flatten)]
    pub pipeline: TranscribePipelineArgs,

    /// Audio file or directory containing audio files.
    #[arg(short, long, value_name = "PATH")]
    pub input: PathBuf,

    /// Output file or directory. Directory mode writes `{stem}.transcribe.json` per input.
    #[arg(short, long, value_name = "PATH")]
    pub output: PathBuf,

    /// Recursively scan subdirectories when `--input` is a directory.
    #[arg(long, default_value_t = false)]
    pub recursive: bool,

    /// Optional ASR language hint.
    #[arg(long, env = "VASR_LANGUAGE")]
    pub language: Option<String>,

    /// Process at most this many audio files.
    #[arg(long)]
    pub limit: Option<usize>,
}

pub async fn run_local(args: RunTranscribeArgs) -> Result<()> {
    validate_pipeline(&args.pipeline)?;
    if !args.input.exists() {
        bail!("input path does not exist: {}", args.input.display());
    }

    let files = collect_audio_inputs(&args.input, args.recursive)?;
    if files.is_empty() {
        bail!("no audio files found under {}", args.input.display());
    }
    let mut files = files;
    if let Some(limit) = args.limit {
        files.truncate(limit);
    }

    let multiple_inputs = files.len() > 1 || args.input.is_dir();
    validate_output_target(&args.output, multiple_inputs)?;

    let pipeline = build_async_transcribe_pipeline(&args.pipeline, args.language.clone())?;
    let inputs = files
        .iter()
        .enumerate()
        .map(|(index, path)| TranscribeInput {
            index,
            source: AudioSource::Path(path.clone()),
        })
        .collect::<Vec<_>>();

    tracing::info!(
        target: "vasr_cli::run",
        "Transcribing {} audio file(s) from `{}`.",
        files.len(),
        args.input.display()
    );

    let batch_start = Instant::now();
    let outcomes = pipeline.transcribe_many(inputs).await;
    let batch_wall = batch_start.elapsed().as_secs_f64();

    let mut total_audio_seconds = 0.0;
    let mut bad_count = 0usize;

    for (path, outcome) in files.iter().zip(outcomes.iter()) {
        total_audio_seconds += outcome.audio_seconds;
        let output_path = resolve_output_json_path(&args.output, path, multiple_inputs)?;
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory {}", parent.display())
            })?;
        }

        let response = transcribe_response_for_outcome(path, outcome, batch_wall, files.len());
        if response.data.first().is_some_and(|item| item.is_bad) {
            bad_count += 1;
        }

        let file = fs::File::create(&output_path)
            .with_context(|| format!("failed to create output file {}", output_path.display()))?;
        serde_json::to_writer_pretty(&file, &response)
            .with_context(|| format!("failed to write JSON to {}", output_path.display()))?;

        tracing::debug!(
            target: "vasr_cli::run",
            "Wrote `{}`.",
            output_path.display()
        );
    }

    let speedup = total_audio_seconds / batch_wall.max(f64::EPSILON);
    let rtf = batch_wall / total_audio_seconds.max(f64::EPSILON);
    tracing::info!(
        target: "vasr_cli::run",
        "Done: files={} bad={} audio_seconds={:.3} wall_seconds={:.3} speedup={:.3} rtf={:.4}",
        files.len(),
        bad_count,
        total_audio_seconds,
        batch_wall,
        speedup,
        rtf
    );

    if bad_count > 0 {
        bail!("{bad_count} of {} transcription(s) failed", files.len());
    }
    Ok(())
}

fn transcribe_response_for_outcome(
    path: &Path,
    outcome: &TranscribeItemOutcome,
    batch_wall: f64,
    batch_size: usize,
) -> TranscribeResponse {
    let service_id = path.display().to_string();
    let data = match &outcome.result {
        Ok(timeline) => InferenceData::from_timeline(&service_id, timeline),
        Err(error) => error_inference_data(
            &service_id,
            outcome.bad_component.unwrap_or("recognizer"),
            error.to_string(),
        ),
    };

    TranscribeResponse {
        data: vec![data],
        inference_performance: InferencePerformance {
            batch_wall_seconds: batch_wall,
            num_items: batch_size,
            throughput_items_per_second: batch_size as f64 / batch_wall.max(f64::EPSILON),
            total_audio_duration_seconds: outcome.audio_seconds,
            speedup: outcome.audio_seconds / batch_wall.max(f64::EPSILON),
            rtf: batch_wall / outcome.audio_seconds.max(f64::EPSILON),
        },
    }
}

fn error_inference_data(service_id: &str, component: &str, reason: String) -> InferenceData {
    InferenceData {
        service_id: service_id.to_string(),
        spent_seconds: 0.0,
        spent_details: Default::default(),
        text: Default::default(),
        sentences: Vec::new(),
        is_bad: true,
        bad_reason: Some(reason),
        bad_component: Some(component.to_string()),
    }
}

pub fn collect_audio_inputs(input: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        if is_audio_file(input) {
            return Ok(vec![input.to_path_buf()]);
        }
        bail!("unsupported audio file extension: {}", input.display());
    }
    if !input.is_dir() {
        bail!("input path is not a file or directory: {}", input.display());
    }

    let mut files = Vec::new();
    collect_audio_in_dir(input, recursive, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_audio_in_dir(dir: &Path, recursive: bool, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            if recursive {
                collect_audio_in_dir(&path, true, out)?;
            }
            continue;
        }
        if is_audio_file(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn is_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "wav" | "mp3" | "flac" | "ogg" | "m4a" | "aac" | "opus" | "webm"
            )
        })
        .unwrap_or(false)
}

fn validate_output_target(output: &Path, multiple_inputs: bool) -> Result<()> {
    if multiple_inputs {
        if output.extension().is_some_and(|ext| ext == "json") {
            bail!(
                "when transcribing multiple files, --output must be a directory, not `{}`",
                output.display()
            );
        }
        return Ok(());
    }
    Ok(())
}

pub fn resolve_output_json_path(
    output: &Path,
    audio: &Path,
    multiple_inputs: bool,
) -> Result<PathBuf> {
    if !multiple_inputs && output.extension().is_some_and(|ext| ext == "json") {
        return Ok(output.to_path_buf());
    }

    let stem = audio
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| anyhow::anyhow!("invalid audio file name: {}", audio.display()))?;

    Ok(output.join(format!("{stem}.transcribe.json")))
}

#[cfg(test)]
mod tests {
    use super::{collect_audio_inputs, resolve_output_json_path};
    use std::path::Path;

    #[test]
    fn resolve_output_json_path_uses_stem_for_directory_output() {
        let output = resolve_output_json_path(
            Path::new("/tmp/out"),
            Path::new("/data/audio (1).wav"),
            true,
        )
        .expect("resolve output");
        assert_eq!(output, Path::new("/tmp/out/audio (1).transcribe.json"));
    }

    #[test]
    fn resolve_output_json_path_honors_explicit_json_file() {
        let output = resolve_output_json_path(
            Path::new("/tmp/custom.transcribe.json"),
            Path::new("/data/audio.wav"),
            false,
        )
        .expect("resolve output");
        assert_eq!(output, Path::new("/tmp/custom.transcribe.json"));
    }

    #[test]
    fn collect_audio_inputs_accepts_single_file() {
        let dir = std::env::temp_dir().join(format!("vasr-run-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let wav = dir.join("sample.wav");
        std::fs::write(&wav, b"RIFF").expect("write wav");

        let files = collect_audio_inputs(&wav, false).expect("collect file");
        assert_eq!(files, vec![wav]);

        std::fs::remove_dir_all(&dir).ok();
    }
}
