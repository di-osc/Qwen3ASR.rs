use anyhow::Result;
use clap::Parser;
use vasr_cli::{RealtimeArgs, init_logging, run_realtime};

#[derive(Debug, Parser)]
#[command(
    name = "vasr-realtime",
    version,
    about = "vASR realtime WebSocket service"
)]
struct Cli {
    /// Show VAD, ASR, and other component runtime logs.
    #[arg(long, short, global = true)]
    verbose: bool,

    #[command(flatten)]
    args: RealtimeArgs,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_logging(cli.verbose);
    run_realtime(cli.args).await
}
