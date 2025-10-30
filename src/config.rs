use std::path::PathBuf;

use image::imageops::FilterType;

/// Options for the model and input image.
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
    pub fn with_model_filter(mut self, filter: FilterType) -> Self {
        self.input_resize_filter = filter;
        self
    }

    /// Set the matte resize filter.
    pub fn with_matte_filter(mut self, filter: FilterType) -> Self {
        self.output_resize_filter = filter;
        self
    }

    /// Set the number of intra-op threads for the inference.
    pub fn with_intra_threads(mut self, intra_threads: Option<usize>) -> Self {
        self.intra_threads = intra_threads;
        self
    }
}

/// Options describing how a mask should be post-processed.
#[derive(Debug, Clone)]
pub struct MaskProcessingOptions {
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
            blur: false,
            blur_sigma: 6.0,
            mask_threshold: 120,
            dilate: false,
            dilation_radius: 5.0,
            fill_holes: false,
        }
    }
}
