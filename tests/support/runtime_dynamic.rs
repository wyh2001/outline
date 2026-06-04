use std::env;
use std::io;
use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::Tensor;

// CI injects a real shared library path; local runs must set it explicitly.
pub const ENV_TEST_ORT_DYLIB: &str = "OUTLINE_TEST_ORT_DYLIB";

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

pub fn assert_tiny_matte_model_runs(model_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut session = Session::builder()?.commit_from_file(model_path)?;
    assert_eq!(session.inputs().len(), 1);
    assert_eq!(session.outputs().len(), 1);

    let input_tensor = Tensor::from_array(([1usize, 3, 2, 2], vec![0.0f32; 12]))?;
    let outputs = session.run(ort::inputs![input_tensor])?;
    let output = outputs[0]
        .try_extract_array::<f32>()?
        .iter()
        .copied()
        .collect::<Vec<_>>();

    assert_eq!(output, vec![0.0, 0.25, 0.5, 1.0]);
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
