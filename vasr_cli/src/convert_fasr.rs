use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Args;
use vasr_data::{convert_fasr_audio_list_file, extract_embedded_audio, inspect_fasr_audio_list};

#[derive(Debug, Clone, Args)]
pub struct ConvertFasrArgs {
    /// FASR `FASRAL01` AudioList binary file.
    #[arg(short, long, value_name = "PATH")]
    pub input: PathBuf,

    /// Output `VasrRecordList` MessagePack file.
    #[arg(short, long, value_name = "PATH")]
    pub output: PathBuf,

    /// Optional directory to also write extracted source audio files.
    #[arg(long, value_name = "DIR")]
    pub audio_dir: Option<PathBuf>,

    /// Convert at most this many records.
    #[arg(long)]
    pub limit: Option<usize>,
}

pub fn run_convert_fasr(args: ConvertFasrArgs) -> Result<()> {
    if !args.input.exists() {
        bail!("input path does not exist: {}", args.input.display());
    }

    let summary = inspect_fasr_audio_list(&args.input)
        .with_context(|| format!("failed to inspect FASR AudioList {}", args.input.display()))?;

    tracing::info!(
        target: "vasr_cli::convert_fasr",
        "Converting {} record(s) from `{}` (has_reference_text={}).",
        summary.sample_count,
        args.input.display(),
        summary.has_reference_text
    );

    let list =
        convert_fasr_audio_list_file(&args.input, &args.output, args.limit).with_context(|| {
            format!(
                "failed to convert FASR AudioList {} to {}",
                args.input.display(),
                args.output.display()
            )
        })?;

    if let Some(audio_dir) = &args.audio_dir {
        let summary = extract_embedded_audio(&list, audio_dir)
            .with_context(|| format!("failed to extract audio to {}", audio_dir.display()))?;
        tracing::info!(
            target: "vasr_cli::convert_fasr",
            "Extracted {} audio file(s) to `{}` (skipped={}).",
            summary.extracted,
            audio_dir.display(),
            summary.skipped
        );
    }

    tracing::info!(
        target: "vasr_cli::convert_fasr",
        "Done: records={} output=`{}` duration_ms={}",
        list.len(),
        args.output.display(),
        list.total_duration().0
    );
    Ok(())
}
