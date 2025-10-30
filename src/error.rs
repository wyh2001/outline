use thiserror::Error;

pub type OutlineResult<T> = std::result::Result<T, OutlineError>;

#[derive(Debug, Error)]
pub enum OutlineError {
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),
    #[error("Image processing failed: {0}")]
    Image(#[from] image::ImageError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("Invalid tensor shape: {0}")]
    Shape(#[from] ndarray::ShapeError),
    #[error("Tracing failed: {0}")]
    Trace(String),
    #[error("Alpha matte size {found:?} does not match source image size {expected:?}")]
    AlphaMismatch {
        expected: (u32, u32),
        found: (u32, u32),
    },
}
