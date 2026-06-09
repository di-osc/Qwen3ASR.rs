use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Args;
use vasr_data::{VasrRecordList, extract_embedded_audio};

#[derive(Debug, Clone, Args)]
pub struct ExtractAudioArgs {
    /// `VasrRecordList` MessagePack file.
    #[arg(short, long, value_name = "PATH")]
    pub input: PathBuf,

    /// Directory to write extracted audio files.
    #[arg(short, long, value_name = "DIR")]
    pub output_dir: PathBuf,
}

pub fn run_extract_audio(args: ExtractAudioArgs) -> Result<()> {
    if !args.input.exists() {
        bail!("input path does not exist: {}", args.input.display());
    }

    let list = VasrRecordList::read_msgpack(&args.input).with_context(|| {
        format!(
            "failed to read VasrRecordList from {}",
            args.input.display()
        )
    })?;

    tracing::info!(
        target: "vasr_cli::extract_audio",
        "Extracting embedded audio from `{}` to `{}`.",
        args.input.display(),
        args.output_dir.display()
    );

    let summary = extract_embedded_audio(&list, &args.output_dir)
        .with_context(|| format!("failed to extract audio to {}", args.output_dir.display()))?;

    tracing::info!(
        target: "vasr_cli::extract_audio",
        "Done: extracted={} skipped={} output_dir=`{}`",
        summary.extracted,
        summary.skipped,
        args.output_dir.display()
    );
    Ok(())
}
