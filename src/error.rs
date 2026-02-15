use std::path::PathBuf;

use thiserror::Error;

/// Result type alias for operations that may fail with [`OutlineError`].
pub type OutlineResult<T> = std::result::Result<T, OutlineError>;

/// Error types that can occur during outline processing.
///
/// This enum covers errors from model inference, image I/O, mask processing,
/// and vectorization operations.
#[derive(Debug, Error)]
pub enum OutlineError {
    /// ONNX Runtime inference error.
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),
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
        expected: (u32, u32),
        found: (u32, u32),
    },
    /// Model file not found at the specified path.
    #[error("Model file not found: {}", path.display())]
    ModelNotFound { path: PathBuf },
}
