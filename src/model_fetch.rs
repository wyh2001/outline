//! Model download functionality.
//!
//! This module is only available when the `fetch-model` feature is enabled.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};

use outline::OutlineResult;

const APP_DIR_NAME: &str = "outline-core";
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
/// Returns `~/.cache/outline-core/` on Linux, `~/Library/Caches/outline-core/` on macOS,
/// or falls back to current directory if home cannot be determined.
pub fn default_model_cache_dir() -> PathBuf {
    let env_override = std::env::var_os("OUTLINE_MODEL_CACHE_DIR").map(PathBuf::from);
    resolve_cache_dir(env_override, dirs::cache_dir())
}

fn resolve_cache_dir(env_override: Option<PathBuf>, system_cache_dir: Option<PathBuf>) -> PathBuf {
    if let Some(path) = env_override
        && !path.as_os_str().is_empty()
    {
        return path;
    }

    system_cache_dir
        .map(|path| path.join(APP_DIR_NAME))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Get the default model cache path.
///
/// Returns `~/.cache/outline-core/model.onnx` on Linux, etc.
pub fn default_model_cache_path() -> PathBuf {
    default_model_cache_dir().join(MODEL_FILENAME)
}

fn download_and_verify<R: Read>(
    reader: &mut R,
    temp_path: &Path,
    expected_sha256: &str,
    pb: &ProgressBar,
) -> OutlineResult<()> {
    let mut file = File::create(temp_path)?;
    let mut hasher = Sha256::new();
    let mut downloaded: u64 = 0;
    let mut buffer = [0u8; 8192];

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

    let actual_hash = format!("{:x}", hasher.finalize());
    if actual_hash != expected_sha256 {
        let _ = fs::remove_file(temp_path);
        return Err(download_error(format!(
            "Checksum verification failed: expected {}, got {}",
            expected_sha256, actual_hash
        ))
        .into());
    }
    eprintln!("Checksum verified.");

    Ok(())
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
    let mut reader = response.into_body().into_reader();
    download_and_verify(&mut reader, &temp_path, &options.expected_sha256, &pb)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn resolve_cache_dir_uses_env_override() {
        let custom_dir = tempfile::tempdir().expect("failed to create temp dir");

        let resolved = resolve_cache_dir(Some(custom_dir.path().to_path_buf()), None);
        assert_eq!(resolved, custom_dir.path().to_path_buf());
    }

    #[test]
    fn resolve_cache_dir_empty_override_falls_back_to_system_cache() {
        let system_cache = tempfile::tempdir().expect("failed to create temp dir");
        let expected = system_cache.path().join(APP_DIR_NAME);

        let resolved = resolve_cache_dir(Some(PathBuf::from("")), Some(system_cache.path().into()));
        assert_eq!(resolved, expected);
    }

    #[test]
    fn resolve_cache_dir_without_any_source_falls_back_to_current_dir() {
        let resolved = resolve_cache_dir(None, None);
        assert_eq!(resolved, PathBuf::from("."));
    }

    #[test]
    fn default_model_cache_path_uses_model_filename() {
        assert_eq!(
            default_model_cache_path()
                .file_name()
                .and_then(|name| name.to_str()),
            Some(MODEL_FILENAME)
        );
    }

    #[test]
    fn fetch_options_default_has_expected_values() {
        let options = FetchOptions::default();

        assert_eq!(options.url, DEFAULT_MODEL_URL);
        assert_eq!(options.expected_sha256, DEFAULT_MODEL_SHA256);
        assert_eq!(options.output, default_model_cache_path());
        assert!(!options.force);
    }

    #[test]
    fn fetch_options_builders_override_fields() {
        let output = PathBuf::from("custom/model.onnx");
        let options = FetchOptions::default()
            .with_output(output.clone())
            .with_force(true);

        assert_eq!(options.output, output);
        assert!(options.force);
    }

    #[test]
    fn fetch_model_existing_file_without_force_skips_download() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let output = temp_dir.path().join("model.onnx");
        fs::write(&output, b"existing-model").expect("failed to write existing model");

        let options = FetchOptions::default().with_output(output.clone());
        let result = fetch_model(&options).expect("expected existing-file shortcut");

        assert_eq!(result, output);
        assert_eq!(
            fs::read(&output).expect("failed to read existing model"),
            b"existing-model"
        );
        assert!(!output.with_extension("onnx.tmp").exists());
    }

    #[test]
    fn download_and_verify_writes_file_when_checksum_matches() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let temp_path = temp_dir.path().join("model.onnx.tmp");
        let bytes = b"outline-model-bytes".to_vec();
        let expected_sha256 = format!("{:x}", Sha256::digest(&bytes));
        let mut reader = Cursor::new(bytes.clone());
        let pb = ProgressBar::hidden();

        download_and_verify(&mut reader, &temp_path, &expected_sha256, &pb)
            .expect("expected checksum success");

        assert_eq!(
            fs::read(&temp_path).expect("failed to read temp model"),
            bytes
        );
    }

    #[test]
    fn download_and_verify_removes_temp_file_when_checksum_fails() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let temp_path = temp_dir.path().join("model.onnx.tmp");
        let mut reader = Cursor::new(b"outline-model-bytes".to_vec());
        let pb = ProgressBar::hidden();

        let error = download_and_verify(&mut reader, &temp_path, "deadbeef", &pb)
            .expect_err("expected checksum failure");

        assert!(error.to_string().contains("Checksum verification failed"));
        assert!(!temp_path.exists());
    }
}
