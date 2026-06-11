use std::path::PathBuf;

#[cfg(feature = "backend-ort")]
use ort::session::builder::SessionBuilder;
use thiserror::Error;

/// Result type alias for operations that may fail with [`OutlineError`].
pub type OutlineResult<T> = std::result::Result<T, OutlineError>;

/// Error types that can occur during outline processing.
///
/// This enum covers errors from model inference, image I/O, mask processing,
/// and vectorization operations.
///
/// This enum is non-exhaustive; include a wildcard arm when matching it.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OutlineError {
    /// ONNX Runtime inference error.
    #[cfg(feature = "backend-ort")]
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),
    /// RTen model loading failed.
    #[cfg(feature = "backend-rten")]
    #[error("RTen model load failed: {0}")]
    RtenLoad(#[from] rten::LoadError),
    /// RTen model execution failed.
    #[cfg(feature = "backend-rten")]
    #[error("RTen model execution failed: {0}")]
    RtenRun(#[from] rten::RunError),
    /// RTen output conversion failed.
    #[cfg(feature = "backend-rten")]
    #[error("RTen output conversion failed: {0}")]
    RtenValue(#[from] rten::TryFromValueError),
    /// Image loading, decoding, or encoding error.
    #[error("Image processing failed: {0}")]
    Image(#[from] image::ImageError),
    /// File system I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Tensor shape mismatch or invalid dimensions.
    #[error("Invalid tensor shape: {0}")]
    Shape(#[from] ndarray::ShapeError),
    /// Vectorization or tracing operation failed.
    #[error("Tracing failed: {0}")]
    Trace(String),
    /// Alpha matte dimensions do not match the source image.
    #[error("Alpha matte size {found:?} does not match source image size {expected:?}")]
    AlphaMismatch {
        /// Expected dimensions (width, height).
        expected: (u32, u32),
        /// Actual dimensions (width, height).
        found: (u32, u32),
    },
    /// Model file not found at the specified path.
    #[error("Model file not found: {}", path.display())]
    ModelNotFound {
        /// The path that was searched.
        path: PathBuf,
    },
}

// Normalize SessionBuilder-specific ORT errors into OutlineError.
#[cfg(feature = "backend-ort")]
impl From<ort::Error<SessionBuilder>> for OutlineError {
    fn from(err: ort::Error<SessionBuilder>) -> Self {
        Self::Ort(err.into())
    }
}
