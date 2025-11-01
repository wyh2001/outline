pub mod config;
pub mod error;
pub mod foreground;
pub mod inference;
pub mod mask;
pub mod vectorizer;

pub use config::{InferenceSettings, MaskProcessingOptions};
pub use error::{OutlineError, OutlineResult};
pub use vectorizer::MaskVectorizer;
#[cfg(feature = "vectorizer-vtracer")]
pub use vectorizer::vtracer::{TraceOptions, VtracerSvgVectorizer, trace_to_svg_string};

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use image::imageops::FilterType;
use image::{GrayImage, RgbImage, RgbaImage};

use crate::foreground::compose_foreground;
use crate::inference::run_matte_pipeline;
use crate::mask::{MaskOperation, apply_operations, operations_from_options};

/// Entry point for configuring and running matte extraction.
#[derive(Debug, Clone)]
pub struct Outline {
    /// Inference settings for model and image handling.
    settings: InferenceSettings,
    /// If nothing is specified and processing is requested, these options will be used.
    default_mask_processing: MaskProcessingOptions,
}

impl Outline {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            settings: InferenceSettings::new(model_path),
            default_mask_processing: MaskProcessingOptions::default(),
        }
    }

    /// Set the filter used to resize the input image for the model.
    pub fn with_input_resize_filter(mut self, filter: FilterType) -> Self {
        self.settings.input_resize_filter = filter;
        self
    }

    /// Set the filter used to resize the output matte to the original image size.
    pub fn with_output_resize_filter(mut self, filter: FilterType) -> Self {
        self.settings.output_resize_filter = filter;
        self
    }

    /// Set the number of intra-op threads for the inference.
    pub fn with_intra_threads(mut self, intra_threads: Option<usize>) -> Self {
        self.settings.intra_threads = intra_threads;
        self
    }

    /// Set the default mask processing options to use when none are specified.
    pub fn with_default_mask_processing(mut self, options: MaskProcessingOptions) -> Self {
        self.default_mask_processing = options;
        self
    }

    /// Get a reference to the default mask processing options.
    pub fn default_mask_processing(&self) -> &MaskProcessingOptions {
        &self.default_mask_processing
    }

    /// Run the inference pipeline for a single image, returning the orginal image, raw matte, and processing options,
    /// wrapped in an `InferencedMatte`.
    pub fn for_image(&self, image_path: impl AsRef<Path>) -> OutlineResult<InferencedMatte> {
        let (rgb, matte) = run_matte_pipeline(&self.settings, image_path.as_ref())?;
        Ok(InferencedMatte::new(
            rgb,
            matte,
            self.default_mask_processing.clone(),
        ))
    }
}

/// Result of running the model for a single image, from which all artefacts can be derived.
#[derive(Debug, Clone)]
pub struct InferencedMatte {
    rgb_image: Arc<RgbImage>,
    raw_matte: Arc<GrayImage>,
    default_mask_processing: MaskProcessingOptions,
}

impl InferencedMatte {
    fn new(
        rgb_image: RgbImage,
        raw_matte: GrayImage,
        default_mask_processing: MaskProcessingOptions,
    ) -> Self {
        Self {
            rgb_image: Arc::new(rgb_image),
            raw_matte: Arc::new(raw_matte),
            default_mask_processing,
        }
    }

    /// Get a reference to the original RGB image.
    pub fn rgb_image(&self) -> &RgbImage {
        self.rgb_image.as_ref()
    }

    /// Get a reference to the raw grayscale matte.
    pub fn raw_matte(&self) -> &GrayImage {
        self.raw_matte.as_ref()
    }

    pub fn matte(&self) -> MatteHandle {
        MatteHandle {
            rgb_image: Arc::clone(&self.rgb_image),
            raw_matte: Arc::clone(&self.raw_matte),
            default_mask_processing: self.default_mask_processing.clone(),
            operations: Vec::new(),
        }
    }
}

/// Builder-style handle for operating on the raw matte produced by the model.
#[derive(Debug, Clone)]
pub struct MatteHandle {
    rgb_image: Arc<RgbImage>,
    raw_matte: Arc<GrayImage>,
    default_mask_processing: MaskProcessingOptions,
    operations: Vec<MaskOperation>,
}

impl MatteHandle {
    /// Get the raw grayscale matte.
    pub fn raw(&self) -> GrayImage {
        (*self.raw_matte).clone()
    }

    /// Consume the handle and return the raw grayscale matte.
    pub fn into_image(self) -> GrayImage {
        (*self.raw_matte).clone()
    }

    /// Save the raw grayscale matte to the specified path.
    pub fn save(&self, path: impl AsRef<Path>) -> OutlineResult<()> {
        self.raw_matte.as_ref().save(path)?;
        Ok(())
    }

    /// Add a blur operation to the processing pipeline.
    pub fn blur(mut self, sigma: f32) -> Self {
        self.operations.push(MaskOperation::Blur { sigma });
        self
    }

    /// Add a threshold operation to the processing pipeline.
    pub fn threshold(mut self, value: u8) -> Self {
        self.operations.push(MaskOperation::Threshold { value });
        self
    }

    /// Add a dilation operation to the processing pipeline.
    pub fn dilate(mut self, radius: f32) -> Self {
        self.operations.push(MaskOperation::Dilate { radius });
        self
    }

    /// Add a hole-filling operation to the processing pipeline.
    pub fn fill_holes(mut self) -> Self {
        let threshold = self.default_mask_processing.mask_threshold;
        self.operations.push(MaskOperation::FillHoles { threshold });
        self
    }

    /// Process the raw matte with the accumulated operations and optional custom options.
    pub fn processed(self, options: Option<&MaskProcessingOptions>) -> OutlineResult<MaskHandle> {
        let mut ops = self.operations;
        match options {
            Some(custom) => ops.extend(operations_from_options(custom)),
            None if ops.is_empty() => {
                ops.extend(operations_from_options(&self.default_mask_processing))
            }
            None => {}
        }

        let mask = apply_operations(self.raw_matte.as_ref(), &ops);
        Ok(MaskHandle::new(
            Arc::clone(&self.rgb_image),
            mask,
            self.default_mask_processing,
        ))
    }

    /// Compose the RGBA foreground image from the RGB image and the raw matte.
    pub fn foreground(&self) -> OutlineResult<ForegroundHandle> {
        let rgba = compose_foreground(self.rgb_image.as_ref(), self.raw_matte.as_ref())?;
        Ok(ForegroundHandle { image: rgba })
    }

    /// Trace the raw matte using the specified vectorizer and options.
    pub fn trace<V>(&self, vectorizer: &V, options: &V::Options) -> OutlineResult<V::Output>
    where
        V: MaskVectorizer,
    {
        vectorizer.vectorize(self.raw_matte.as_ref(), options)
    }
}

/// Represents a concrete mask image along with pending processing instructions.
#[derive(Debug, Clone)]
pub struct MaskHandle {
    rgb_image: Arc<RgbImage>,
    mask: GrayImage,
    default_mask_processing: MaskProcessingOptions,
    operations: Vec<MaskOperation>,
}

impl MaskHandle {
    fn new(
        rgb_image: Arc<RgbImage>,
        mask: GrayImage,
        default_mask_processing: MaskProcessingOptions,
    ) -> Self {
        Self {
            rgb_image,
            mask,
            default_mask_processing,
            operations: Vec::new(),
        }
    }

    /// Get the raw  mask.
    pub fn raw(&self) -> GrayImage {
        self.mask.clone()
    }

    /// Get a reference to the mask.
    pub fn image(&self) -> &GrayImage {
        &self.mask
    }

    /// Consume the handle and return the mask.
    pub fn into_image(self) -> GrayImage {
        self.mask
    }

    /// Save the mask to the specified path.
    pub fn save(&self, path: impl AsRef<Path>) -> OutlineResult<()> {
        self.mask.save(path)?;
        Ok(())
    }

    /// Add a blur operation to the processing pipeline.
    pub fn blur(mut self, sigma: f32) -> Self {
        self.operations.push(MaskOperation::Blur { sigma });
        self
    }

    /// Add a threshold operation to the processing pipeline.
    pub fn threshold(mut self, value: u8) -> Self {
        self.operations.push(MaskOperation::Threshold { value });
        self
    }

    /// Add a dilation operation to the processing pipeline.
    pub fn dilate(mut self, radius: f32) -> Self {
        self.operations.push(MaskOperation::Dilate { radius });
        self
    }

    /// Add a hole-filling operation to the processing pipeline.
    pub fn fill_holes(mut self) -> Self {
        let threshold = self.default_mask_processing.mask_threshold;
        self.operations.push(MaskOperation::FillHoles { threshold });
        self
    }

    /// Process the mask with the accumulated operations and optional custom options.
    pub fn processed(self, options: Option<&MaskProcessingOptions>) -> OutlineResult<MaskHandle> {
        let mut ops = self.operations;
        match options {
            Some(custom) => ops.extend(operations_from_options(custom)),
            None if ops.is_empty() => {
                ops.extend(operations_from_options(&self.default_mask_processing))
            }
            None => {}
        }

        let mask = apply_operations(&self.mask, &ops);
        Ok(MaskHandle::new(
            self.rgb_image,
            mask,
            self.default_mask_processing,
        ))
    }

    /// Compose the RGBA foreground image from the RGB image and the current mask.
    pub fn foreground(&self) -> OutlineResult<ForegroundHandle> {
        let rgba = compose_foreground(self.rgb_image.as_ref(), &self.mask)?;
        Ok(ForegroundHandle { image: rgba })
    }

    /// Trace the current mask using the specified vectorizer and options.
    pub fn trace<V>(&self, vectorizer: &V, options: &V::Options) -> OutlineResult<V::Output>
    where
        V: MaskVectorizer,
    {
        vectorizer.vectorize(&self.mask, options)
    }
}

pub struct ForegroundHandle {
    image: RgbaImage,
}

impl ForegroundHandle {
    /// Get a reference to the RGBA foreground image.
    pub fn image(&self) -> &RgbaImage {
        &self.image
    }

    /// Consume the handle and return the RGBA foreground image.
    pub fn into_image(self) -> RgbaImage {
        self.image
    }

    /// Save the RGBA foreground image to the specified path.
    pub fn save(&self, path: impl AsRef<Path>) -> OutlineResult<()> {
        self.image.save(path)?;
        Ok(())
    }
}
