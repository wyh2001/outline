use std::path::PathBuf;

use image::imageops::FilterType;

/// Environment variable name for specifying the model path.
pub const ENV_MODEL_PATH: &str = "OUTLINE_MODEL_PATH";

/// Default model path used when no explicit path is provided.
pub const DEFAULT_MODEL_PATH: &str = "model.onnx";

/// Configuration for ONNX model inference and image preprocessing.
///
/// Controls the model path, image resize filters for input/output, and threading behavior.
/// Use builder methods like [`with_input_resize_filter`](InferenceSettings::with_input_resize_filter)
/// to customize settings.
#[derive(Debug, Clone)]
pub struct InferenceSettings {
    /// Path to the ONNX model file.
    pub model_path: PathBuf,
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
            input_resize_filter: FilterType::Triangle,
            output_resize_filter: FilterType::Lanczos3,
            intra_threads: None,
        }
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

/// Configuration for mask post-processing operations.
///
/// Defines the pipeline of blur, threshold, dilation, and hole-filling operations applied
/// to raw mattes. Used as defaults in [`Outline`](crate::Outline) and can be overridden
/// per operation via [`MatteHandle`](crate::MatteHandle) and [`MaskHandle`](crate::MaskHandle).
///
/// # Explicit Configuration
///
/// This struct does **not** apply automatic logic. For example, setting `dilate = true` or
/// `fill_holes = true` will **not** automatically enable `binary`. If you need a binary mask
/// for dilation or hole-filling to work meaningfully, you must explicitly set `binary = true`
/// or call [`threshold`](crate::MatteHandle::threshold) in your processing chain.
///
/// **Note**: The CLI's `--binary auto` mode *does* automatically enable thresholding when
/// `--dilate` or `--fill-holes` is specified. The library leaves this decision to you for
/// maximum control and predictability.
#[derive(Debug, Clone)]
pub struct MaskProcessingOptions {
    pub binary: bool,
    pub blur: bool,
    pub blur_sigma: f32,
    pub mask_threshold: u8,
    pub dilate: bool,
    pub dilation_radius: f32,
    pub fill_holes: bool,
}

impl Default for MaskProcessingOptions {
    fn default() -> Self {
        Self {
            binary: false,
            blur: false,
            blur_sigma: 6.0,
            mask_threshold: 120,
            dilate: false,
            dilation_radius: 5.0,
            fill_holes: false,
        }
    }
}
