//! Model download functionality.
//!
//! This module is only available when the `model-fetch` feature is enabled.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;

use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};

use outline::OutlineResult;

const APP_DIR_NAME: &str = "outline";
const MODEL_FILENAME: &str = "model.onnx";
const DEFAULT_MODEL_URL: &str =
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx";
const DEFAULT_MODEL_SHA256: &str =
    "75da6c8d2f8096ec743d071951be73b4a8bc7b3e51d9a6625d63644f90ffeedb";

fn download_error(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::other(msg.into())
}

/// Options for fetching the model.
#[derive(Debug, Clone)]
pub struct FetchOptions {
    /// URL to download the model from.
    pub url: String,
    /// Expected SHA-256 checksum (hex string).
    pub expected_sha256: String,
    /// Output path for the downloaded model.
    pub output: PathBuf,
    /// Whether to overwrite existing files.
    pub force: bool,
}

impl Default for FetchOptions {
    fn default() -> Self {
        Self {
            url: DEFAULT_MODEL_URL.to_string(),
            expected_sha256: DEFAULT_MODEL_SHA256.to_string(),
            output: default_model_cache_path(),
            force: false,
        }
    }
}

impl FetchOptions {
    /// Create new fetch options with a custom output path.
    pub fn with_output(mut self, output: PathBuf) -> Self {
        self.output = output;
        self
    }

    /// Set whether to overwrite existing files.
    pub fn with_force(mut self, force: bool) -> Self {
        self.force = force;
        self
    }
}

/// Get the default model cache directory path.
///
/// Returns `~/.cache/outline/` on Linux, `~/Library/Caches/outline/` on macOS,
/// or falls back to current directory if home cannot be determined.
pub fn default_model_cache_dir() -> PathBuf {
    if let Some(from_env) = std::env::var_os("OUTLINE_MODEL_CACHE_DIR") {
        let path = PathBuf::from(from_env);
        if !path.as_os_str().is_empty() {
            return path;
        }
    }

    dirs::cache_dir()
        .map(|p| p.join(APP_DIR_NAME))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Get the default model cache path.
///
/// Returns `~/.cache/outline/model.onnx` on Linux, etc.
pub fn default_model_cache_path() -> PathBuf {
    default_model_cache_dir().join(MODEL_FILENAME)
}

/// Fetch the model from the configured URL.
///
/// Downloads the model file with a progress bar and verifies the checksum.
pub fn fetch_model(options: &FetchOptions) -> OutlineResult<PathBuf> {
    // Check if file already exists
    if options.output.exists() && !options.force {
        eprintln!(
            "Model already exists at: {}\nUse --force to overwrite.",
            options.output.display()
        );
        return Ok(options.output.clone());
    }

    // Create parent directories if needed
    if let Some(parent) = options.output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    eprintln!("Downloading model from: {}", options.url);
    eprintln!("Saving to: {}", options.output.display());

    // Download with progress
    let response = ureq::get(&options.url)
        .call()
        .map_err(|error| match error {
            ureq::Error::StatusCode(status) => {
                download_error(format!("Model download failed: HTTP error {status}"))
            }
            other => download_error(format!("Model download failed: {other}")),
        })?;
    let total_size = response.body().content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})")
            .expect("Invalid progress bar template")
            .progress_chars("#>-"),
    );

    // Download to a temporary file first
    let temp_path = options.output.with_extension("onnx.tmp");
    let mut file = File::create(&temp_path)?;
    let mut hasher = Sha256::new();

    let mut downloaded: u64 = 0;
    let mut buffer = [0u8; 8192];
    let mut reader = response.into_body().into_reader();

    loop {
        let bytes_read = reader
            .read(&mut buffer)
            .map_err(|e| download_error(format!("Model download failed: {e}")))?;

        if bytes_read == 0 {
            break;
        }

        file.write_all(&buffer[..bytes_read])?;
        hasher.update(&buffer[..bytes_read]);

        downloaded += bytes_read as u64;
        pb.set_position(downloaded);
    }

    pb.finish_with_message("Download complete");
    file.flush()?;
    drop(file);

    // Verify checksum
    let actual_hash = format!("{:x}", hasher.finalize());
    if actual_hash != options.expected_sha256 {
        // Clean up the temporary file
        let _ = fs::remove_file(&temp_path);
        return Err(download_error(format!(
            "Checksum verification failed: expected {}, got {}",
            options.expected_sha256, actual_hash
        ))
        .into());
    }
    eprintln!("Checksum verified.");

    // Move temp file to final location
    if options.force && options.output.exists() {
        fs::remove_file(&options.output)?;
    }
    fs::rename(&temp_path, &options.output)?;

    eprintln!("Model saved to: {}", options.output.display());
    if options.output == default_model_cache_path() {
        eprintln!();
        eprintln!(
            "Tip: If you don't pass --model and {} is not set, outline will automatically use this cached model when `./{}` is missing.",
            outline::ENV_MODEL_PATH,
            MODEL_FILENAME
        );
        eprintln!(
            "To pin this model explicitly, set environment variable {} to that path.",
            outline::ENV_MODEL_PATH
        );
    } else {
        eprintln!();
        eprintln!(
            "To use this model by default, set environment variable {} to:",
            outline::ENV_MODEL_PATH
        );
        eprintln!("  {}", options.output.display());
        eprintln!("(Exact syntax depends on your shell/OS; you can also pass `--model <path>`.)");
    }

    Ok(options.output.clone())
}
