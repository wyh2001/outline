use std::path::Path;
use std::sync::Arc;

use image::{GrayImage, RgbImage};

use crate::config::MaskProcessingOptions;
use crate::foreground::{ForegroundHandle, compose_foreground};
use crate::mask::{MaskHandle, MaskOperation, apply_operations, operations_from_options};
use crate::{MaskVectorizer, OutlineResult};

/// Inference result containing the original RGB image and raw matte prediction.
///
/// Returned by [`crate::Outline::for_image`] after running model inference.
///
/// # Example
/// ```no_run
/// use outline::Outline;
///
/// let outline = Outline::new("model.onnx");
/// let session = outline.for_image("input.png")?;
/// let matte = session.matte();
///
/// // Access the original image and raw matte directly
/// let rgb = session.rgb_image();
/// let raw_matte = session.raw_matte();
///
/// // Compose the foreground from the raw matte
/// let foreground = matte.foreground()?;
/// foreground.save("foreground.png")?;
/// # Ok::<_, outline::OutlineError>(())
/// ```
#[derive(Debug, Clone)]
pub struct InferencedMatte {
    rgb_image: Arc<RgbImage>,
    raw_matte: Arc<GrayImage>,
    default_mask_processing: MaskProcessingOptions,
}

impl InferencedMatte {
    pub(crate) fn new(
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

    /// Begin building a mask processing pipeline from the raw matte.
    pub fn matte(&self) -> MatteHandle {
        MatteHandle {
            rgb_image: Arc::clone(&self.rgb_image),
            raw_matte: Arc::clone(&self.raw_matte),
            default_mask_processing: self.default_mask_processing.clone(),
            operations: Vec::new(),
        }
    }
}

/// Builder for chaining mask processing operations on the raw matte.
///
/// The raw matte is the soft, grayscale alpha prediction from the model.
///
/// # Example
/// ```no_run
/// use outline::Outline;
///
/// let outline = Outline::new("model.onnx");
/// let session = outline.for_image("input.jpg")?;
///
/// // Chain operations and execute them
/// let mask = session.matte()
///     .blur_with(6.0)          // Smooth edges
///     .threshold_with(120)     // Convert to binary
///     .dilate_with(5.0)        // Expand slightly
///     .processed()?;           // Execute operations
///
/// mask.save("mask.png")?;
/// # Ok::<_, outline::OutlineError>(())
/// ```
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

    /// Add a blur operation using the default sigma.
    pub fn blur(mut self) -> Self {
        let sigma = self.default_mask_processing.blur_sigma;
        self.operations.push(MaskOperation::Blur { sigma });
        self
    }

    /// Add a blur operation with a custom sigma.
    pub fn blur_with(mut self, sigma: f32) -> Self {
        self.operations.push(MaskOperation::Blur { sigma });
        self
    }

    /// Add a threshold operation using the default mask threshold.
    pub fn threshold(mut self) -> Self {
        let value = self.default_mask_processing.mask_threshold;
        self.operations.push(MaskOperation::Threshold { value });
        self
    }

    /// Add a threshold operation with a custom value.
    pub fn threshold_with(mut self, value: u8) -> Self {
        self.operations.push(MaskOperation::Threshold { value });
        self
    }

    /// Add a dilation operation using the default radius.
    ///
    /// **Note**: Dilation typically works best on binary masks. Consider calling
    /// [`threshold`](MatteHandle::threshold) before `dilate` if working with a soft matte.
    pub fn dilate(mut self) -> Self {
        let radius = self.default_mask_processing.dilation_radius;
        self.operations.push(MaskOperation::Dilate { radius });
        self
    }

    /// Add a dilation operation with a custom radius.
    ///
    /// **Note**: Dilation typically works best on binary masks. Consider calling
    /// [`threshold`](MatteHandle::threshold) before `dilate` if working with a soft matte.
    pub fn dilate_with(mut self, radius: f32) -> Self {
        self.operations.push(MaskOperation::Dilate { radius });
        self
    }

    /// Add a hole-filling operation to the processing pipeline.
    ///
    /// **Note**: Hole-filling typically works best on binary masks. Consider calling
    /// [`threshold`](MatteHandle::threshold) before `fill_holes` if working with a soft matte.
    pub fn fill_holes(mut self) -> Self {
        let threshold = self.default_mask_processing.mask_threshold;
        self.operations.push(MaskOperation::FillHoles { threshold });
        self
    }

    /// Process the raw matte with the accumulated operations and default options.
    pub fn processed(self) -> OutlineResult<MaskHandle> {
        self.process_with_options(None)
    }

    /// Process the raw matte with the accumulated operations and custom options.
    pub fn processed_with(self, options: &MaskProcessingOptions) -> OutlineResult<MaskHandle> {
        self.process_with_options(Some(options))
    }

    /// Helper function to process with options.
    fn process_with_options(
        mut self,
        options: Option<&MaskProcessingOptions>,
    ) -> OutlineResult<MaskHandle> {
        let mut ops = std::mem::take(&mut self.operations);
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
        Ok(ForegroundHandle::new(rgba))
    }

    /// Trace the raw matte using the specified vectorizer and options.
    pub fn trace<V>(&self, vectorizer: &V, options: &V::Options) -> OutlineResult<V::Output>
    where
        V: MaskVectorizer,
    {
        vectorizer.vectorize(self.raw_matte.as_ref(), options)
    }
}
