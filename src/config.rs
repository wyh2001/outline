use std::path::PathBuf;

use image::imageops::FilterType;

/// Environment variable name for specifying the model path.
pub const ENV_MODEL_PATH: &str = "OUTLINE_MODEL_PATH";

/// Default model path used when no explicit path is provided.
///
/// This is the fallback when neither `--model` nor `OUTLINE_MODEL_PATH` is set.
/// By default it points to `model.onnx` in the current working directory.
pub const DEFAULT_MODEL_PATH: &str = "model.onnx";

/// Inference backend used to execute the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InferenceBackend {
    /// Use ONNX Runtime through the `ort` crate.
    #[cfg(feature = "backend-ort")]
    #[default]
    Ort,
    /// Use the pure Rust RTen backend.
    #[cfg(feature = "backend-rten")]
    #[cfg_attr(not(feature = "backend-ort"), default)]
    Rten,
}

/// Configuration for ONNX model inference and image preprocessing.
///
/// Controls the model path, image resize filters for input/output, and threading behavior.
/// Use builder methods like [`with_input_resize_filter`](InferenceSettings::with_input_resize_filter)
/// to customize settings.
///
/// This struct is non-exhaustive; use [`new`](InferenceSettings::new) to construct it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InferenceSettings {
    /// Path to the ONNX model file.
    pub model_path: PathBuf,
    /// Backend used to execute the model.
    pub backend: InferenceBackend,
    /// Filter to use when resizing the input image for the model.
    pub input_resize_filter: FilterType,
    /// Filter to use when resizing the output matte to the original image size.
    pub output_resize_filter: FilterType,
    /// Number of intra-op threads for the inference.
    pub intra_threads: Option<usize>,
}

impl InferenceSettings {
    /// Create new inference settings with default values.
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            backend: InferenceBackend::default(),
            input_resize_filter: FilterType::Triangle,
            output_resize_filter: FilterType::Lanczos3,
            intra_threads: None,
        }
    }

    /// Set the inference backend.
    pub fn with_backend(mut self, backend: InferenceBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set the model resize filter.
    pub fn with_input_resize_filter(mut self, filter: FilterType) -> Self {
        self.input_resize_filter = filter;
        self
    }

    /// Set the matte resize filter.
    pub fn with_output_resize_filter(mut self, filter: FilterType) -> Self {
        self.output_resize_filter = filter;
        self
    }

    /// Set the number of intra-op threads for the inference.
    pub fn with_intra_threads(mut self, intra_threads: Option<usize>) -> Self {
        self.intra_threads = intra_threads;
        self
    }
}

/// How erosion treats pixels outside the image bounds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ErosionBorderMode {
    /// Treat pixels outside the image as background, allowing edge-touching foreground to shrink.
    #[default]
    OutsideIsBackground,
    /// Treat pixels outside the image as unknown, so erosion only uses visible in-image background.
    OutsideIsUnknown,
}

/// Default parameters used by no-argument mask processing methods.
///
/// This does not define which operations run. Use [`MaskPipeline`](crate::MaskPipeline) or the
/// chained methods on [`MatteHandle`](crate::MatteHandle) and [`MaskHandle`](crate::MaskHandle)
/// to choose an explicit operation order.
///
/// This struct is non-exhaustive; start with [`Default`] and then adjust fields as needed.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MaskProcessingDefaults {
    /// Standard deviation (sigma) for Gaussian blur.
    pub blur_sigma: f32,
    /// Threshold value (0–255) used for binary conversion and hole-filling.
    pub mask_threshold: u8,
    /// Radius in pixels for the dilation operation.
    pub dilation_radius: f32,
    /// Radius in pixels for the erosion operation.
    pub erosion_radius: f32,
    /// How erosion treats pixels outside the image bounds.
    pub erosion_border_mode: ErosionBorderMode,
}

impl Default for MaskProcessingDefaults {
    fn default() -> Self {
        Self {
            blur_sigma: 6.0,
            mask_threshold: 120,
            dilation_radius: 5.0,
            erosion_radius: 5.0,
            erosion_border_mode: ErosionBorderMode::default(),
        }
    }
}
