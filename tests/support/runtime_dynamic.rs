use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::Tensor;
use tempfile::TempDir;

pub const ENV_TEST_ORT_DYLIB: &str = "OUTLINE_TEST_ORT_DYLIB";

// Embed a small model file so the test does not depend on external model files.
const IDENTITY_ORT: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/identity.ort"
));

pub fn shared_library_from_env(var: &str) -> io::Result<PathBuf> {
    let path = env::var_os(var).ok_or_else(|| {
        io::Error::other(format!(
            "{var} is not set; this ignored test requires a real ONNX Runtime shared library"
        ))
    })?;
    validate_runtime_path(PathBuf::from(path))
}

pub fn runtime_dylib_from_env() -> io::Result<PathBuf> {
    shared_library_from_env(ENV_TEST_ORT_DYLIB)
}

pub fn write_identity_model() -> io::Result<(TempDir, PathBuf)> {
    let temp_dir = tempfile::tempdir()?;
    let model_path = temp_dir.path().join("identity.ort");
    fs::write(&model_path, IDENTITY_ORT)?;
    Ok((temp_dir, model_path))
}

pub fn assert_identity_model_runs(model_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut session = Session::builder()?.commit_from_file(model_path)?;
    assert_eq!(session.inputs().len(), 1);
    assert_eq!(session.outputs().len(), 1);

    let input_tensor = Tensor::from_array(([3usize], vec![true, false, true]))?;
    let outputs = session.run(ort::inputs![input_tensor])?;
    let output = outputs[0]
        .try_extract_array::<bool>()?
        .iter()
        .copied()
        .collect::<Vec<_>>();

    assert_eq!(output, vec![true, false, true]);
    Ok(())
}

fn validate_runtime_path(path: PathBuf) -> io::Result<PathBuf> {
    if !path.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "ONNX Runtime shared library not found at {}",
                path.display()
            ),
        ));
    }

    path.canonicalize().or(Ok(path))
}
