mod serve;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(name = "vasr", version, about = "vASR speech inference service")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Start a fasr-compatible transcribe or realtime service.
    Serve(serve::ServeArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    init_logging();
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => serve::run(args).await,
    }
}

fn init_logging() {
    let filter = std::env::var("VASR_LOG")
        .unwrap_or_else(|_| "warn,vasr_cli=info,vasr_runtime=info,vasr_server=info".to_string());

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(filter))
        .with_timer(tracing_subscriber::fmt::time::UtcTime::rfc_3339())
        .with_target(true)
        .compact()
        .init();
}
