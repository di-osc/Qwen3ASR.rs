use anyhow::Result;
use clap::{Parser, Subcommand};
use vasr_transcribe::{
    BenchmarkTranscribeArgs, ExtractAudioArgs, RunTranscribeArgs, ServeTranscribeArgs,
    init_logging, run_benchmark, run_extract_audio, run_local, run_transcribe,
};

#[derive(Debug, Parser)]
#[command(
    name = "vasr-transcribe",
    version,
    about = "vASR offline transcribe CLI"
)]
struct Cli {
    /// Show loader, VAD, ASR, and other component runtime logs.
    #[arg(long, short, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Start the offline transcribe HTTP service.
    Serve(ServeTranscribeArgs),
    /// Transcribe local audio files and write `{stem}.transcribe.json` outputs.
    Run(RunTranscribeArgs),
    /// Benchmark ASR CER against a `VasrRecordList` MessagePack file.
    Benchmark(BenchmarkTranscribeArgs),
    /// Extract embedded audio from a `VasrRecordList` MessagePack file.
    ExtractAudio(ExtractAudioArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_logging(cli.verbose);
    match cli.command {
        Command::Serve(args) => run_transcribe(args).await,
        Command::Run(args) => run_local(args).await,
        Command::Benchmark(args) => run_benchmark(args).await,
        Command::ExtractAudio(args) => run_extract_audio(args),
    }
}
