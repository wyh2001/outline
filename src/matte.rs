use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;

use image::{GrayImage, RgbImage, RgbaImage};

use crate::config::{ErosionBorderMode, MaskProcessingOptions};
use crate::foreground::{ForegroundHandle, compose_foreground};
use crate::geometry::{
    BoundingBox, Padding, crop_gray_image, crop_rgb_image, mask_bounding_box, pad_gray_image,
    pad_rgb_image,
};
use crate::mask::{
    MaskColor, MaskHandle, MaskOperation, apply_operations, colorize_mask, operations_from_options,
};
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
    fn resolved_matte(&self) -> Cow<'_, GrayImage> {
        if self.operations.is_empty() {
            Cow::Borrowed(self.raw_matte.as_ref())
        } else {
            Cow::Owned(apply_operations(self.raw_matte.as_ref(), &self.operations))
        }
    }

    fn resolve_pending_operations(mut self) -> Self {
        if self.operations.is_empty() {
            return self;
        }

        let operations = std::mem::take(&mut self.operations);
        let matte = apply_operations(self.raw_matte.as_ref(), &operations);
        Self {
            rgb_image: self.rgb_image,
            raw_matte: Arc::new(matte),
            default_mask_processing: self.default_mask_processing,
            operations: Vec::new(),
        }
    }

    /// Get the raw grayscale matte.
    pub fn raw(&self) -> GrayImage {
        (*self.raw_matte).clone()
    }

    /// Consume the handle and return the current matte as a grayscale image.
    pub fn into_image(self) -> GrayImage {
        if self.operations.is_empty() {
            Arc::unwrap_or_clone(self.raw_matte)
        } else {
            apply_operations(self.raw_matte.as_ref(), &self.operations)
        }
    }

    /// Save the current matte to the specified path.
    pub fn save(&self, path: impl AsRef<Path>) -> OutlineResult<()> {
        self.resolved_matte().save(path)?;
        Ok(())
    }

    /// Compute the bounding box of the current matte using a non-zero threshold.
    pub fn bounding_box(&self) -> Option<BoundingBox> {
        self.bounding_box_with(1)
    }

    /// Compute the bounding box of the current matte at or above `threshold`.
    pub fn bounding_box_with(&self, threshold: u8) -> Option<BoundingBox> {
        let mask = self.resolved_matte();
        mask_bounding_box(&mask, threshold)
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

    /// Add an erosion operation using the default radius.
    ///
    /// **Note**: Erosion typically works best on binary masks. Consider calling
    /// [`threshold`](MatteHandle::threshold) before `erode` if working with a soft matte.
    pub fn erode(mut self) -> Self {
        let radius = self.default_mask_processing.erosion_radius;
        let border_mode = self.default_mask_processing.erosion_border_mode;
        self.operations.push(MaskOperation::Erode {
            radius,
            border_mode,
        });
        self
    }

    /// Add an erosion operation with a custom radius.
    ///
    /// **Note**: Erosion typically works best on binary masks. Consider calling
    /// [`threshold`](MatteHandle::threshold) before `erode` if working with a soft matte.
    pub fn erode_with(mut self, radius: f32) -> Self {
        let border_mode = self.default_mask_processing.erosion_border_mode;
        self.operations.push(MaskOperation::Erode {
            radius,
            border_mode,
        });
        self
    }

    /// Add an erosion operation with a custom radius and boundary behavior.
    ///
    /// **Note**: Erosion typically works best on binary masks. Consider calling
    /// [`threshold`](MatteHandle::threshold) before `erode` if working with a soft matte.
    pub fn erode_with_border_mode(mut self, radius: f32, border_mode: ErosionBorderMode) -> Self {
        self.operations.push(MaskOperation::Erode {
            radius,
            border_mode,
        });
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

    /// Compose the RGBA foreground image from the RGB image and the current matte.
    pub fn foreground(&self) -> OutlineResult<ForegroundHandle> {
        let mask = self.resolved_matte();
        let rgba = compose_foreground(self.rgb_image.as_ref(), &mask)?;
        Ok(ForegroundHandle::new(rgba))
    }

    /// Colorize the current matte into a flat-color RGBA image.
    pub fn colorize(&self, color: impl Into<MaskColor>) -> RgbaImage {
        let mask = self.resolved_matte();
        colorize_mask(&mask, color)
    }

    /// Trace the current matte using the specified vectorizer and options.
    pub fn trace<V>(&self, vectorizer: &V, options: &V::Options) -> OutlineResult<V::Output>
    where
        V: MaskVectorizer,
    {
        let mask = self.resolved_matte();
        vectorizer.vectorize(&mask, options)
    }

    /// Expand the matte canvas by the given padding while keeping content at the same offset.
    ///
    /// Any pending matte operations are applied before padding so method call order is preserved.
    /// The underlying RGB canvas is padded with black to keep [`foreground`](MatteHandle::foreground)
    /// aligned with the matte.
    pub fn pad(self, padding: impl Into<Padding>) -> Self {
        let this = self.resolve_pending_operations();
        let padding = padding.into();
        let matte = Arc::new(pad_gray_image(this.raw_matte.as_ref(), padding, 0));
        let rgb = Arc::new(pad_rgb_image(this.rgb_image.as_ref(), padding, [0, 0, 0]));
        Self {
            rgb_image: rgb,
            raw_matte: matte,
            default_mask_processing: this.default_mask_processing,
            operations: Vec::new(),
        }
    }

    /// Crop the matte and source RGB image to the smallest non-zero content box plus `margin`.
    ///
    /// Returns `None` when the current matte has no non-zero pixels.
    pub fn crop_to_content(self, margin: impl Into<Padding>) -> Option<Self> {
        self.crop_to_content_with(1, margin)
    }

    /// Crop the matte and source RGB image to the smallest content box at or above `threshold`,
    /// plus `margin`.
    ///
    /// Returns `None` when the current matte has no pixels at or above `threshold`.
    pub fn crop_to_content_with(self, threshold: u8, margin: impl Into<Padding>) -> Option<Self> {
        let this = self.resolve_pending_operations();
        let margin = margin.into();
        let bounds = mask_bounding_box(this.raw_matte.as_ref(), threshold)?.expanded_to_fit(
            margin,
            this.raw_matte.width(),
            this.raw_matte.height(),
        );
        let matte = Arc::new(crop_gray_image(this.raw_matte.as_ref(), bounds));
        let rgb = Arc::new(crop_rgb_image(this.rgb_image.as_ref(), bounds));
        Some(Self {
            rgb_image: rgb,
            raw_matte: matte,
            default_mask_processing: this.default_mask_processing,
            operations: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgb};

    fn matte_handle() -> MatteHandle {
        MatteHandle {
            rgb_image: Arc::new(RgbImage::from_pixel(1, 1, Rgb([255, 255, 255]))),
            raw_matte: Arc::new(GrayImage::from_pixel(1, 1, Luma([255]))),
            default_mask_processing: MaskProcessingOptions::default(),
            operations: Vec::new(),
        }
    }

    fn single_pixel_matte_handle() -> MatteHandle {
        MatteHandle {
            rgb_image: Arc::new(RgbImage::from_pixel(5, 5, Rgb([10, 20, 30]))),
            raw_matte: Arc::new(GrayImage::from_fn(5, 5, |x, y| {
                if x == 2 && y == 2 {
                    Luma([255])
                } else {
                    Luma([0])
                }
            })),
            default_mask_processing: MaskProcessingOptions::default(),
            operations: Vec::new(),
        }
    }

    fn matte_handle_with_images(rgb_image: RgbImage, raw_matte: GrayImage) -> MatteHandle {
        MatteHandle {
            rgb_image: Arc::new(rgb_image),
            raw_matte: Arc::new(raw_matte),
            default_mask_processing: MaskProcessingOptions::default(),
            operations: Vec::new(),
        }
    }

    struct BoundingBoxVectorizer;

    impl MaskVectorizer for BoundingBoxVectorizer {
        type Options = ();
        type Output = Option<BoundingBox>;

        fn vectorize(
            &self,
            mask: &GrayImage,
            _options: &Self::Options,
        ) -> OutlineResult<Self::Output> {
            Ok(mask_bounding_box(mask, 1))
        }
    }

    #[test]
    fn matte_handle_erode_uses_default_radius() {
        let handle = matte_handle().erode();
        assert!(matches!(
            handle.operations.as_slice(),
            [MaskOperation::Erode { radius, border_mode }]
                if (*radius - MaskProcessingOptions::default().erosion_radius).abs() < f32::EPSILON
                    && *border_mode == MaskProcessingOptions::default().erosion_border_mode
        ));
    }

    #[test]
    fn matte_handle_erode_with_uses_custom_radius() {
        let handle = matte_handle().erode_with(2.5);
        assert!(matches!(
            handle.operations.as_slice(),
            [MaskOperation::Erode { radius, border_mode }]
                if (*radius - 2.5).abs() < f32::EPSILON
                    && *border_mode == MaskProcessingOptions::default().erosion_border_mode
        ));
    }

    #[test]
    fn matte_handle_erode_with_border_mode_uses_custom_mode() {
        let handle =
            matte_handle().erode_with_border_mode(2.5, ErosionBorderMode::OutsideIsUnknown);
        assert!(matches!(
            handle.operations.as_slice(),
            [MaskOperation::Erode { radius, border_mode }]
                if (*radius - 2.5).abs() < f32::EPSILON
                    && *border_mode == ErosionBorderMode::OutsideIsUnknown
        ));
    }

    #[test]
    fn matte_handle_bounding_box_applies_pending_operations() {
        let bounds = single_pixel_matte_handle()
            .dilate_with(1.0)
            .bounding_box_with(1)
            .expect("expected bounding box");

        assert_eq!(bounds, BoundingBox::new(1, 1, 3, 3));
    }

    #[test]
    fn matte_handle_bounding_box_uses_non_zero_threshold() {
        let bounds = single_pixel_matte_handle()
            .bounding_box()
            .expect("expected bounding box");

        assert_eq!(bounds, BoundingBox::new(2, 2, 1, 1));
    }

    #[test]
    fn matte_handle_pad_updates_matte_and_foreground_canvas() {
        let rgb = RgbImage::from_pixel(2, 2, Rgb([10, 20, 30]));
        let mut matte = GrayImage::from_pixel(2, 2, Luma([0]));
        matte.put_pixel(1, 1, Luma([255]));

        let padded = matte_handle_with_images(rgb, matte).pad(Padding::new(1, 2, 3, 4));

        assert_eq!(padded.raw().dimensions(), (6, 8));
        let foreground = padded.foreground().expect("foreground should compose");
        assert_eq!(foreground.image().dimensions(), (6, 8));
        assert_eq!(foreground.image().get_pixel(2, 3)[3], 255);
    }

    #[test]
    fn matte_handle_pad_applies_pending_operations_first() {
        let padded = single_pixel_matte_handle()
            .dilate_with(1.0)
            .pad(Padding::new(1, 2, 0, 0));

        assert_eq!(
            mask_bounding_box(&padded.raw(), 1),
            Some(BoundingBox::new(2, 3, 3, 3))
        );
    }

    #[test]
    fn matte_handle_crop_to_content_crops_matte_and_rgb_together() {
        let rgb = RgbImage::from_fn(4, 4, |x, y| Rgb([x as u8, y as u8, 0]));
        let mut matte = GrayImage::from_pixel(4, 4, Luma([0]));
        matte.put_pixel(2, 1, Luma([255]));
        matte.put_pixel(2, 2, Luma([255]));

        let cropped = matte_handle_with_images(rgb, matte)
            .crop_to_content(1)
            .expect("matte has content");

        assert_eq!(cropped.raw().dimensions(), (3, 4));
        let foreground = cropped.foreground().expect("foreground should compose");
        assert_eq!(foreground.image().dimensions(), (3, 4));
        assert_eq!(foreground.image().get_pixel(1, 1)[0], 2);
        assert_eq!(foreground.image().get_pixel(1, 1)[1], 1);
        assert_eq!(foreground.image().get_pixel(1, 1)[3], 255);
    }

    #[test]
    fn matte_handle_into_image_applies_pending_operations() {
        let mask = single_pixel_matte_handle().dilate_with(1.0).into_image();

        assert_eq!(
            mask_bounding_box(&mask, 1),
            Some(BoundingBox::new(1, 1, 3, 3))
        );
    }

    #[test]
    fn matte_handle_save_applies_pending_operations() {
        let temp_dir = tempfile::tempdir().expect("temp dir should be created");
        let path = temp_dir.path().join("matte.png");

        single_pixel_matte_handle()
            .dilate_with(1.0)
            .save(&path)
            .expect("matte should save");

        let saved = image::open(&path)
            .expect("saved matte should load")
            .to_luma8();

        assert_eq!(
            mask_bounding_box(&saved, 1),
            Some(BoundingBox::new(1, 1, 3, 3))
        );
    }

    #[test]
    fn matte_handle_foreground_applies_pending_operations() {
        let foreground = single_pixel_matte_handle()
            .dilate_with(1.0)
            .foreground()
            .expect("foreground should compose");

        assert_eq!(foreground.image().get_pixel(1, 2)[3], 255);
        assert_eq!(foreground.image().get_pixel(0, 0)[3], 0);
    }

    #[test]
    fn matte_handle_colorize_applies_pending_operations() {
        let colorized = single_pixel_matte_handle()
            .dilate_with(1.0)
            .colorize([0, 180, 255, 255]);

        assert_eq!(colorized.get_pixel(1, 2).0, [0, 180, 255, 255]);
        assert_eq!(colorized.get_pixel(0, 0).0, [0, 180, 255, 0]);
    }

    #[test]
    fn matte_handle_trace_applies_pending_operations() {
        let bounds = single_pixel_matte_handle()
            .dilate_with(1.0)
            .trace(&BoundingBoxVectorizer, &())
            .expect("trace should run");

        assert_eq!(bounds, Some(BoundingBox::new(1, 1, 3, 3)));
    }
}
