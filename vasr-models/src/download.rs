//! Vendored ModelScope downloader.
//!
//! Kept in-tree so we control the download UX instead of relying on the
//! external `modelscope` crate.

use anyhow::{Context, bail};
use futures_util::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const FILES_URL: &str =
    "https://modelscope.cn/api/v1/models/<model_id>/repo/files?Recursive=true";
const DOWNLOAD_URL: &str = "https://modelscope.cn/models/<model_id>/resolve/master/<path>";
const COOKIES_FILE: &str = "cookies";

const UA: (&str, &str) = (
    "User-Agent",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
);

const BAR_STYLE: &str = "{msg:<30} {bar} {decimal_bytes:<10} / {decimal_total_bytes:<10} {decimal_bytes_per_sec:<12} {percent:<3}%  {eta_precise}";

#[derive(Debug, Deserialize)]
struct ModelScopeResponse {
    #[serde(rename = "Code")]
    #[allow(unused)]
    code: i64,
    #[serde(rename = "Success")]
    success: bool,
    #[serde(rename = "Message")]
    message: String,
    #[serde(rename = "Data")]
    data: Option<ModelScopeResponseData>,
}

#[derive(Debug, Deserialize)]
struct ModelScopeResponseData {
    #[serde(rename = "Files")]
    files: Vec<RepoFile>,
}

#[derive(Debug, Deserialize)]
struct RepoFile {
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Path")]
    path: String,
    #[serde(rename = "Size")]
    size: u64,
    #[serde(rename = "Sha256")]
    #[allow(unused)]
    sha256: String,
    #[serde(rename = "Type")]
    r#type: String,
}

fn cookies_dir() -> anyhow::Result<PathBuf> {
    let dir = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
        .join(".modelscope")
        .join("config");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn get_cookies() -> anyhow::Result<Option<String>> {
    let cookies_file = cookies_dir()?.join(COOKIES_FILE);
    if !cookies_file.exists() {
        return Ok(None);
    }
    let cookies = fs::read_to_string(&cookies_file)?;
    let cookies: serde_json::Value = serde_json::from_str(&cookies)?;
    let cookies = cookies
        .as_object()
        .context("Failed to parse cookies")?
        .iter()
        .map(|(k, v)| format!("{}={}", k, v.as_str().unwrap_or_default()))
        .collect::<Vec<_>>()
        .join("; ");
    Ok(Some(cookies))
}

async fn get_client() -> anyhow::Result<reqwest::Client> {
    let client = reqwest::Client::builder().connect_timeout(std::time::Duration::from_secs(10));
    let mut default_headers = reqwest::header::HeaderMap::new();
    if let Some(cookies) = get_cookies()? {
        default_headers.insert("Cookie", cookies.parse()?);
    }
    Ok(client.default_headers(default_headers).build()?)
}

async fn download_file(
    client: Arc<reqwest::Client>,
    model_id: String,
    repo_file: RepoFile,
    save_dir: PathBuf,
    bar: ProgressBar,
) -> anyhow::Result<()> {
    let path = &repo_file.path;
    let file_path = save_dir.join(path);
    if let Some(parent) = file_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Skip if already fully downloaded.
    if file_path.exists() {
        if let Ok(meta) = fs::metadata(&file_path) {
            if meta.len() == repo_file.size {
                bar.set_message(format!("{} (cached)", repo_file.name));
                bar.set_length(1);
                bar.set_position(1);
                bar.finish();
                return Ok(());
            }
        }
    }

    bar.set_message(repo_file.name.clone());
    bar.set_length(repo_file.size);

    let url = DOWNLOAD_URL
        .replace("<model_id>", &model_id)
        .replace("<path>", path);

    let mut file = BufWriter::new(fs::File::create(&file_path)?);
    let response = client.get(&url).header(UA.0, UA.1).send().await?;

    let status = response.status();
    if !status.is_success() && status != reqwest::StatusCode::PARTIAL_CONTENT {
        bar.abandon();
        bail!(
            "Failed to download file {}: HTTP {}",
            repo_file.name,
            status
        );
    }

    let mut stream = response.bytes_stream();
    while let Some(item) = stream.next().await {
        let chunk = item?;
        file.write_all(&chunk)?;
        bar.inc(chunk.len() as u64);
    }
    file.flush()?;
    bar.finish();
    Ok(())
}

/// Download a model from ModelScope into `save_dir`.
///
/// `save_dir` is the *cache root* (e.g. `$HOME/.cache/vasr`).
/// The actual files are placed under `save_dir/<model_id>`.
///
/// Progress bars are displayed only when files actually need downloading.
pub async fn download_model(model_id: &str, save_dir: impl Into<PathBuf>) -> anyhow::Result<()> {
    let save_dir = save_dir.into();
    fs::create_dir_all(&save_dir)?;

    let model_dir = save_dir.join(model_id);
    fs::create_dir_all(&model_dir)?;

    let files_url = FILES_URL.replace("<model_id>", model_id);
    let client = Arc::new(get_client().await?);

    let resp = client.get(files_url).send().await?;
    if !resp.status().is_success() {
        bail!(
            "Failed to list model files for {model_id}: {}\nTip: Maybe the model ID is incorrect or login is required",
            resp.text().await?
        );
    }

    let response = resp.json::<ModelScopeResponse>().await?;
    if !response.success {
        bail!("Failed to list model files: {}", response.message);
    }

    let repo_files: Vec<_> = response
        .data
        .unwrap()
        .files
        .into_iter()
        .filter(|f| f.r#type == "blob")
        .collect();

    let bars = MultiProgress::new();
    let mut tasks = Vec::new();

    for repo_file in repo_files {
        let bar = ProgressBar::new(0);
        let style = ProgressStyle::default_bar().template(BAR_STYLE)?;
        bar.set_style(style);
        bars.add(bar.clone());

        let client = client.clone();
        let model_id = model_id.to_string();
        let save_dir = model_dir.clone();
        let task = tokio::spawn(async move {
            download_file(client, model_id, repo_file, save_dir, bar).await
        });
        tasks.push(task);
    }

    for task in tasks {
        task.await??;
    }

    tracing::info!("Downloaded `{model_id}` to {}", model_dir.display());
    Ok(())
}
