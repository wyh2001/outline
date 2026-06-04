use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

use image::{GrayImage, Luma, RgbImage, Rgba, RgbaImage};
use imageproc::contrast::{ThresholdType, threshold as ip_threshold};
use imageproc::distance_transform::euclidean_squared_distance_transform;
use imageproc::filter::gaussian_blur_f32;
use ndarray::Array2;

use crate::MaskVectorizer;
use crate::OutlineResult;
use crate::config::{ErosionBorderMode, MaskProcessingOptions};
use crate::foreground::{ForegroundHandle, compose_foreground};
use crate::geometry::{
    BoundingBox, Padding, crop_gray_image, crop_rgb_image, mask_bounding_box, pad_gray_image,
    pad_rgb_image,
};

#[cfg(feature = "vectorizer-vtracer")]
use vtracer::ColorImage;

/// A single transformation step applied to a grayscale mask image.
#[derive(Debug, Clone)]
pub enum MaskOperation {
    Blur {
        sigma: f32,
    },
    Threshold {
        value: u8,
    },
    Dilate {
        radius: f32,
    },
    Erode {
        radius: f32,
        border_mode: ErosionBorderMode,
    },
    FillHoles {
        threshold: u8,
    },
}

impl MaskOperation {
    pub fn apply(&self, input: &GrayImage) -> GrayImage {
        match self {
            MaskOperation::Blur { sigma } => gaussian_blur_f32(input, *sigma),
            MaskOperation::Threshold { value } => threshold_mask(input, *value),
            MaskOperation::Dilate { radius } => dilate_euclidean(input, *radius),
            MaskOperation::Erode {
                radius,
                border_mode,
            } => erode_euclidean_with_border_mode(input, *radius, *border_mode),
            MaskOperation::FillHoles { threshold } => fill_mask_holes(input, *threshold),
        }
    }
}

/// Run a list of operations against the provided source image, returning the transformed mask.
pub fn apply_operations(source: &GrayImage, operations: &[MaskOperation]) -> GrayImage {
    let mut current = source.clone();
    for op in operations {
        current = op.apply(&current);
    }
    current
}

/// Produce a standard operation sequence based on simple mask processing options.
pub fn operations_from_options(options: &MaskProcessingOptions) -> Vec<MaskOperation> {
    let mut operations = Vec::new();
    if options.blur {
        operations.push(MaskOperation::Blur {
            sigma: options.blur_sigma,
        });
    }
    if options.binary {
        operations.push(MaskOperation::Threshold {
            value: options.mask_threshold,
        });
    }
    if options.dilate {
        operations.push(MaskOperation::Dilate {
            radius: options.dilation_radius,
        });
    }
    if options.erode {
        operations.push(MaskOperation::Erode {
            radius: options.erosion_radius,
            border_mode: options.erosion_border_mode,
        });
    }
    if options.fill_holes {
        operations.push(MaskOperation::FillHoles {
            threshold: options.mask_threshold,
        });
    }
    operations
}

/// Convert a 2D array of f32 values in [0.0, 1.0] to a grayscale image.
pub fn array_to_gray_image(array: &Array2<f32>) -> GrayImage {
    let (h, w) = array.dim();
    GrayImage::from_fn(w as u32, h as u32, |x, y| {
        let value = array[[y as usize, x as usize]].clamp(0.0, 1.0);
        let byte = (value * 255.0 + 0.5) as u8;
        Luma([byte])
    })
}

/// Convert a grayscale image to an RGBA color image.
#[cfg(feature = "vectorizer-vtracer")]
pub fn gray_to_color_image_rgba(
    gray: &GrayImage,
    threshold: Option<u8>,
    invert: bool,
) -> ColorImage {
    let (w, h) = gray.dimensions();
    let (w_usize, h_usize) = (w as usize, h as usize);
    let mut rgba = vec![0u8; 4 * w_usize * h_usize];

    for (i, gray_pixel) in gray.pixels().enumerate() {
        let Luma([g]) = gray_pixel;
        let base = if let Some(t) = threshold {
            if *g >= t { 255 } else { 0 }
        } else {
            *g
        };
        let v = if invert {
            255u8.saturating_sub(base)
        } else {
            base
        };
        let idx = i * 4;
        rgba[idx] = v;
        rgba[idx + 1] = v;
        rgba[idx + 2] = v;
        rgba[idx + 3] = 255;
    }

    ColorImage {
        pixels: rgba,
        width: w_usize,
        height: h_usize,
    }
}

/// Threshold the grayscale image to produce a binary mask.
pub fn threshold_mask(gray: &GrayImage, thr: u8) -> GrayImage {
    ip_threshold(gray, thr, ThresholdType::Binary)
}

pub fn dilate_euclidean(mask_bin: &GrayImage, r: f32) -> GrayImage {
    if r <= 0.0 {
        return mask_bin.clone();
    }

    let d2 = euclidean_squared_distance_transform(mask_bin);
    let r2: f64 = (r as f64) * (r as f64);
    let (w, h) = mask_bin.dimensions();
    let mut out = GrayImage::new(w, h);
    for (o_pixel, d2pixel) in out.pixels_mut().zip(d2.pixels()) {
        let d2xy: f64 = d2pixel[0];
        let v: u8 = if d2xy <= r2 { 255 } else { 0 };
        *o_pixel = Luma([v]);
    }
    out
}

/// Erode a binary mask by the provided radius using the requested boundary behavior.
pub fn erode_euclidean_with_border_mode(
    mask_bin: &GrayImage,
    r: f32,
    border_mode: ErosionBorderMode,
) -> GrayImage {
    if r <= 0.0 {
        return mask_bin.clone();
    }

    let inverted = invert_mask(mask_bin);
    match border_mode {
        ErosionBorderMode::OutsideIsUnknown => invert_mask(&dilate_euclidean(&inverted, r)),
        ErosionBorderMode::OutsideIsBackground => {
            let padding = Padding::uniform(1);
            let padded = pad_gray_image(&inverted, padding, 255);
            let dilated = dilate_euclidean(&padded, r);
            let cropped = crop_gray_image(
                &dilated,
                BoundingBox::new(
                    padding.left,
                    padding.top,
                    mask_bin.width(),
                    mask_bin.height(),
                ),
            );
            invert_mask(&cropped)
        }
    }
}

/// Invert a grayscale mask so each pixel becomes `255 - value`.
pub fn invert_mask(mask: &GrayImage) -> GrayImage {
    let (w, h) = mask.dimensions();
    let mut out = GrayImage::new(w, h);
    for (src, dst) in mask.pixels().zip(out.pixels_mut()) {
        *dst = Luma([255u8.saturating_sub(src[0])]);
    }
    out
}

/// Fill holes in a binary mask using a flood-fill algorithm from the borders.
pub fn fill_mask_holes(mask: &GrayImage, threshold: u8) -> GrayImage {
    let (w, h) = mask.dimensions();
    let (w_usize, h_usize) = (w as usize, h as usize);
    let mut visited = vec![false; w_usize * h_usize];
    let mut queue = VecDeque::new();

    let idx = |x: u32, y: u32| -> usize { (y as usize) * w_usize + x as usize };
    let mask_raw = mask.as_raw();

    // Start flood-fill from all dark pixels at the image borders
    for x in 0..w {
        if mask_raw[idx(x, 0)] < threshold {
            queue.push_back((x, 0));
        }
        if mask_raw[idx(x, h - 1)] < threshold {
            queue.push_back((x, h - 1));
        }
    }

    for y in 0..h {
        if mask_raw[idx(0, y)] < threshold {
            queue.push_back((0, y));
        }
        if mask_raw[idx(w - 1, y)] < threshold {
            queue.push_back((w - 1, y));
        }
    }

    // Use BFS to find all dark pixels connected to the borders
    while let Some((x, y)) = queue.pop_front() {
        let id = idx(x, y);
        if visited[id] {
            continue;
        }
        visited[id] = true;

        // Check neighbors (left, right, up, down) and enqueue if dark and unvisited
        if x > 0 {
            let nx = x - 1;
            let nid = idx(nx, y);
            if !visited[nid] && mask_raw[nid] < threshold {
                queue.push_back((nx, y));
            }
        }
        if x + 1 < w {
            let nx = x + 1;
            let nid = idx(nx, y);
            if !visited[nid] && mask_raw[nid] < threshold {
                queue.push_back((nx, y));
            }
        }
        if y > 0 {
            let ny = y - 1;
            let nid = idx(x, ny);
            if !visited[nid] && mask_raw[nid] < threshold {
                queue.push_back((x, ny));
            }
        }
        if y + 1 < h {
            let ny = y + 1;
            let nid = idx(x, ny);
            if !visited[nid] && mask_raw[nid] < threshold {
                queue.push_back((x, ny));
            }
        }
    }

    let mut out = GrayImage::new(w, h);
    for ((x, y, out_pixel), mask_pixel) in out.enumerate_pixels_mut().zip(mask.pixels()) {
        let id = idx(x, y);
        let value = mask_pixel[0];
        // A pixel is part of a hole if it's dark but was not visited
        let filled = if value >= threshold || !visited[id] {
            255
        } else {
            0
        };
        *out_pixel = Luma([filled]);
    }

    out
}

/// How mask values are converted into output alpha.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum MaskAlphaMode {
    /// Use the mask value directly as alpha.
    #[default]
    UseMask,
    /// Scale the mask alpha by a factor.
    Scale(f32),
    /// Treat any non-zero mask value as a solid alpha.
    Solid(u8),
}

/// Options for colorizing a mask into an RGBA image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskColor {
    /// Flat RGBA color applied to mask-covered pixels; A acts as a global opacity multiplier.
    pub color: [u8; 4],
    /// How mask values influence the output alpha.
    pub alpha_mode: MaskAlphaMode,
}

impl MaskColor {
    /// Create a flat-color mask colorization that uses the mask as-is for alpha.
    pub fn new(color: [u8; 4]) -> Self {
        Self {
            color,
            alpha_mode: MaskAlphaMode::UseMask,
        }
    }

    /// Override the alpha mode while keeping the color.
    pub fn with_alpha_mode(mut self, alpha_mode: MaskAlphaMode) -> Self {
        self.alpha_mode = alpha_mode;
        self
    }
}

impl Default for MaskColor {
    fn default() -> Self {
        Self::new([255, 255, 255, 255])
    }
}

impl From<[u8; 4]> for MaskColor {
    fn from(color: [u8; 4]) -> Self {
        Self::new(color)
    }
}

fn resolve_mask_alpha(mask_value: u8, mode: MaskAlphaMode) -> u8 {
    match mode {
        MaskAlphaMode::UseMask => mask_value,
        MaskAlphaMode::Scale(scale) => {
            let scaled = (mask_value as f32) * scale.max(0.0);
            scaled.round().clamp(0.0, 255.0) as u8
        }
        MaskAlphaMode::Solid(alpha) => {
            if mask_value > 0 {
                alpha
            } else {
                0
            }
        }
    }
}

/// Convert a grayscale mask into a flat-color RGBA image.
///
/// The final alpha is the mask-derived alpha, as controlled by [`MaskAlphaMode`],
/// multiplied by `color[3]`.
pub fn colorize_mask(mask: &GrayImage, color: impl Into<MaskColor>) -> RgbaImage {
    let color = color.into();
    let (w, h) = mask.dimensions();
    let [r, g, b, base_alpha] = color.color;

    let mut out = RgbaImage::new(w, h);
    for (mask_px, out_px) in mask.pixels().zip(out.pixels_mut()) {
        let mask_alpha = resolve_mask_alpha(mask_px[0], color.alpha_mode);
        let alpha = ((mask_alpha as u16 * base_alpha as u16) / 255) as u8;
        *out_px = Rgba([r, g, b, alpha]);
    }

    out
}

/// Processed mask image with optional further refinement and output generation.
///
/// Represents a concrete mask image (typically binary after thresholding) produced by executing
/// operations from a [`crate::MatteHandle`].
///
/// # Example
/// ```no_run
/// use outline::Outline;
///
/// let outline = Outline::new("model.onnx");
/// let session = outline.for_image("input.jpg")?;
/// let mask = session.matte().blur().threshold().processed()?;
///
/// // Generate multiple outputs from the mask
/// mask.save("mask.png")?;
/// mask.foreground()?.save("subject.png")?;
/// # Ok::<_, outline::OutlineError>(())
/// ```
///
/// To trace the mask into SVG, pass a [`MaskVectorizer`] implementation to [`MaskHandle::trace`].
/// The crate re-exports `VtracerSvgVectorizer` when the `vectorizer-vtracer` feature is enabled.
#[derive(Debug, Clone)]
pub struct MaskHandle {
    rgb_image: Arc<RgbImage>,
    mask: GrayImage,
    default_mask_processing: MaskProcessingOptions,
    operations: Vec<MaskOperation>,
}

impl MaskHandle {
    pub(crate) fn new(
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

    fn resolve_pending_operations(mut self) -> Self {
        if self.operations.is_empty() {
            return self;
        }

        let operations = std::mem::take(&mut self.operations);
        let mask = apply_operations(&self.mask, &operations);
        Self::new(self.rgb_image, mask, self.default_mask_processing)
    }

    fn resolved_mask(&self) -> GrayImage {
        if self.operations.is_empty() {
            self.mask.clone()
        } else {
            apply_operations(&self.mask, &self.operations)
        }
    }

    /// Get the raw mask.
    pub fn raw(&self) -> GrayImage {
        self.mask.clone()
    }

    /// Get a reference to the raw mask.
    pub fn image(&self) -> &GrayImage {
        &self.mask
    }

    /// Consume the handle and return the current mask.
    pub fn into_image(self) -> GrayImage {
        self.resolve_pending_operations().mask
    }

    /// Save the current mask to the specified path.
    pub fn save(&self, path: impl AsRef<Path>) -> OutlineResult<()> {
        self.resolved_mask().save(path)?;
        Ok(())
    }

    /// Compute the bounding box of the current mask using a non-zero threshold.
    pub fn bounding_box(&self) -> Option<BoundingBox> {
        self.bounding_box_with(1)
    }

    /// Compute the bounding box of the current mask at or above `threshold`.
    pub fn bounding_box_with(&self, threshold: u8) -> Option<BoundingBox> {
        let mask = self.resolved_mask();
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
    /// **Note**: Dilation typically works best on binary masks. If this mask is still grayscale,
    /// consider calling [`threshold`](MaskHandle::threshold) first.
    pub fn dilate(mut self) -> Self {
        let radius = self.default_mask_processing.dilation_radius;
        self.operations.push(MaskOperation::Dilate { radius });
        self
    }

    /// Add a dilation operation with a custom radius.
    ///
    /// **Note**: Dilation typically works best on binary masks. If this mask is still grayscale,
    /// consider calling [`threshold`](MaskHandle::threshold) first.
    pub fn dilate_with(mut self, radius: f32) -> Self {
        self.operations.push(MaskOperation::Dilate { radius });
        self
    }

    /// Add an erosion operation using the default radius.
    ///
    /// **Note**: Erosion typically works best on binary masks. If this mask is still grayscale,
    /// consider calling [`threshold`](MaskHandle::threshold) first.
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
    /// **Note**: Erosion typically works best on binary masks. If this mask is still grayscale,
    /// consider calling [`threshold`](MaskHandle::threshold) first.
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
    /// **Note**: Erosion typically works best on binary masks. If this mask is still grayscale,
    /// consider calling [`threshold`](MaskHandle::threshold) first.
    pub fn erode_with_border_mode(mut self, radius: f32, border_mode: ErosionBorderMode) -> Self {
        self.operations.push(MaskOperation::Erode {
            radius,
            border_mode,
        });
        self
    }

    /// Add a hole-filling operation to the processing pipeline.
    ///
    /// **Note**: Hole-filling typically works best on binary masks. If this mask is still grayscale,
    /// consider calling [`threshold`](MaskHandle::threshold) first.
    pub fn fill_holes(mut self) -> Self {
        let threshold = self.default_mask_processing.mask_threshold;
        self.operations.push(MaskOperation::FillHoles { threshold });
        self
    }

    /// Process the mask with the accumulated operations and default options.
    pub fn processed(self) -> OutlineResult<MaskHandle> {
        self.process_with_options(None)
    }

    /// Process the mask with the accumulated operations and custom options.
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

        let mask = apply_operations(&self.mask, &ops);
        Ok(MaskHandle::new(
            self.rgb_image,
            mask,
            self.default_mask_processing,
        ))
    }

    /// Compose the RGBA foreground image from the RGB image and the current mask.
    pub fn foreground(&self) -> OutlineResult<ForegroundHandle> {
        let mask = self.resolved_mask();
        let rgba = compose_foreground(self.rgb_image.as_ref(), &mask)?;
        Ok(ForegroundHandle::new(rgba))
    }

    /// Colorize the current mask into a flat-color RGBA image.
    pub fn colorize(&self, color: impl Into<MaskColor>) -> RgbaImage {
        let mask = self.resolved_mask();
        colorize_mask(&mask, color)
    }

    /// Trace the current mask using the specified vectorizer and options.
    pub fn trace<V>(&self, vectorizer: &V, options: &V::Options) -> OutlineResult<V::Output>
    where
        V: MaskVectorizer,
    {
        let mask = self.resolved_mask();
        vectorizer.vectorize(&mask, options)
    }

    /// Expand the mask canvas by the given padding while keeping content at the same offset.
    ///
    /// Any pending mask operations are applied before padding so method call order is preserved.
    /// The underlying RGB canvas is padded with black to keep [`foreground`](MaskHandle::foreground)
    /// aligned with the mask.
    pub fn pad(self, padding: impl Into<Padding>) -> Self {
        let this = self.resolve_pending_operations();
        let padding = padding.into();
        let mask = pad_gray_image(&this.mask, padding, 0);
        let rgb = Arc::new(pad_rgb_image(this.rgb_image.as_ref(), padding, [0, 0, 0]));
        Self::new(rgb, mask, this.default_mask_processing)
    }

    /// Crop the mask and source RGB image to the smallest non-zero content box plus `margin`.
    ///
    /// Returns `None` when the current mask has no non-zero pixels.
    pub fn crop_to_content(self, margin: impl Into<Padding>) -> Option<Self> {
        self.crop_to_content_with(1, margin)
    }

    /// Crop the mask and source RGB image to the smallest content box at or above `threshold`,
    /// plus `margin`.
    ///
    /// Returns `None` when the current mask has no pixels at or above `threshold`.
    pub fn crop_to_content_with(self, threshold: u8, margin: impl Into<Padding>) -> Option<Self> {
        let this = self.resolve_pending_operations();
        let margin = margin.into();
        let bounds = mask_bounding_box(&this.mask, threshold)?.expanded_to_fit(
            margin,
            this.mask.width(),
            this.mask.height(),
        );
        let mask = crop_gray_image(&this.mask, bounds);
        let rgb = Arc::new(crop_rgb_image(this.rgb_image.as_ref(), bounds));
        Some(Self::new(rgb, mask, this.default_mask_processing))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn gray_image(w: u32, h: u32, value: u8) -> GrayImage {
        GrayImage::from_pixel(w, h, Luma([value]))
    }

    mod threshold_mask {
        use super::*;

        mod unit {
            use super::*;

            #[test]
            fn all_below_threshold_become_black() {
                let input = gray_image(2, 2, 100);
                let result = threshold_mask(&input, 128);
                for px in result.pixels() {
                    assert_eq!(px.0[0], 0);
                }
            }

            #[test]
            fn all_above_threshold_become_white() {
                let input = gray_image(2, 2, 200);
                let result = threshold_mask(&input, 128);
                for px in result.pixels() {
                    assert_eq!(px.0[0], 255);
                }
            }

            #[test]
            fn exact_threshold_becomes_black() {
                // imageproc threshold: > threshold -> white, <= threshold -> black
                let input = gray_image(2, 2, 128);
                let result = threshold_mask(&input, 128);
                for px in result.pixels() {
                    assert_eq!(px.0[0], 0);
                }
            }

            #[test]
            fn one_below_threshold_becomes_black() {
                let input = gray_image(2, 2, 127);
                let result = threshold_mask(&input, 128);
                for px in result.pixels() {
                    assert_eq!(px.0[0], 0);
                }
            }

            #[test]
            fn dimensions_preserved() {
                let input = gray_image(5, 3, 100);
                let result = threshold_mask(&input, 128);
                assert_eq!(result.dimensions(), (5, 3));
            }

            #[test]
            fn mixed_values_per_pixel() {
                // 2x2 image with values: 127, 128, 129, 255
                let mut input = GrayImage::new(2, 2);
                input.put_pixel(0, 0, Luma([127])); // thr-1
                input.put_pixel(1, 0, Luma([128])); // thr
                input.put_pixel(0, 1, Luma([129])); // thr+1
                input.put_pixel(1, 1, Luma([255])); // max

                let result = threshold_mask(&input, 128);

                assert_eq!(result.get_pixel(0, 0).0[0], 0); // 127 <= 128
                assert_eq!(result.get_pixel(1, 0).0[0], 0); // 128 <= 128
                assert_eq!(result.get_pixel(0, 1).0[0], 255); // 129 > 128
                assert_eq!(result.get_pixel(1, 1).0[0], 255); // 255 > 128
            }

            #[test]
            fn threshold_zero() {
                // thr=0: only value=0 becomes black, everything else white
                let mut input = GrayImage::new(2, 1);
                input.put_pixel(0, 0, Luma([0]));
                input.put_pixel(1, 0, Luma([1]));

                let result = threshold_mask(&input, 0);

                assert_eq!(result.get_pixel(0, 0).0[0], 0); // 0 <= 0
                assert_eq!(result.get_pixel(1, 0).0[0], 255); // 1 > 0
            }

            #[test]
            fn threshold_255() {
                // thr=255: everything becomes black (nothing > 255)
                let mut input = GrayImage::new(2, 1);
                input.put_pixel(0, 0, Luma([254]));
                input.put_pixel(1, 0, Luma([255]));

                let result = threshold_mask(&input, 255);

                assert_eq!(result.get_pixel(0, 0).0[0], 0); // 254 <= 255
                assert_eq!(result.get_pixel(1, 0).0[0], 0); // 255 <= 255
            }
        }

        mod prop {
            use super::*;
            use proptest::prelude::*;

            proptest! {
                /// threshold_mask: output is always binary (0 or 255)
                #[test]
                fn output_is_binary(
                    w in 1u32..20,
                    h in 1u32..20,
                    fill_value in proptest::num::u8::ANY,
                    threshold in proptest::num::u8::ANY
                ) {
                    let input = GrayImage::from_pixel(w, h, Luma([fill_value]));
                    let result = threshold_mask(&input, threshold);

                    prop_assert_eq!(result.dimensions(), (w, h));
                    for px in result.pixels() {
                        prop_assert!(px.0[0] == 0 || px.0[0] == 255);
                    }
                }

                /// threshold_mask: values > threshold become 255, values <= threshold become 0
                #[test]
                fn respects_threshold(
                    value in proptest::num::u8::ANY,
                    threshold in proptest::num::u8::ANY
                ) {
                    let input = GrayImage::from_pixel(1, 1, Luma([value]));
                    let result = threshold_mask(&input, threshold);
                    let out = result.get_pixel(0, 0).0[0];

                    if value > threshold {
                        prop_assert_eq!(out, 255);
                    } else {
                        prop_assert_eq!(out, 0);
                    }
                }
            }
        }
    }

    mod array_to_gray_image {
        use super::*;

        mod unit {
            use super::*;

            #[test]
            fn all_zeros_black() {
                let arr = arr2(&[[0.0, 0.0], [0.0, 0.0]]);
                let result = array_to_gray_image(&arr);
                for px in result.pixels() {
                    assert_eq!(px.0[0], 0);
                }
            }

            #[test]
            fn all_ones_white() {
                let arr = arr2(&[[1.0, 1.0], [1.0, 1.0]]);
                let result = array_to_gray_image(&arr);
                for px in result.pixels() {
                    assert_eq!(px.0[0], 255);
                }
            }

            #[test]
            fn half_value_gray() {
                let arr = arr2(&[[0.5]]);
                let result = array_to_gray_image(&arr);
                // 0.5 * 255 + 0.5 = 128
                assert_eq!(result.get_pixel(0, 0).0[0], 128);
            }

            #[test]
            fn clamps_above_one() {
                let arr = arr2(&[[2.0]]);
                let result = array_to_gray_image(&arr);
                assert_eq!(result.get_pixel(0, 0).0[0], 255);
            }

            #[test]
            fn clamps_below_zero() {
                let arr = arr2(&[[-1.0]]);
                let result = array_to_gray_image(&arr);
                assert_eq!(result.get_pixel(0, 0).0[0], 0);
            }

            #[test]
            fn dimensions_preserved() {
                let arr = arr2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
                let result = array_to_gray_image(&arr);
                // ndarray is (rows, cols) = (h, w), image is (w, h)
                assert_eq!(result.dimensions(), (3, 2));
            }

            #[test]
            fn rounding_behavior() {
                let arr = arr2(&[[0.4]]);
                let result = array_to_gray_image(&arr);
                assert_eq!(result.get_pixel(0, 0).0[0], 102); // 0.4 * 255 + 0.5 = 102.5, truncated
            }
        }

        mod prop {
            use super::*;
            use proptest::prelude::*;

            proptest! {
                /// array_to_gray_image: any f32 input clamps to [0, 255] range
                #[test]
                fn clamps_out_of_range_values(
                    values in proptest::collection::vec(-10.0f32..10.0f32, 1..100)
                ) {
                    // Create a 1D array that we'll reshape - use length as width, height=1
                    let len = values.len();
                    let arr = ndarray::Array2::from_shape_vec((1, len), values.clone()).unwrap();
                    let result = array_to_gray_image(&arr);

                    // Verify clamping: negative values -> 0, values > 1 -> 255
                    for (i, px) in result.pixels().enumerate() {
                        let input = values[i];
                        if input <= 0.0 {
                            prop_assert_eq!(px.0[0], 0);
                        } else if input >= 1.0 {
                            prop_assert_eq!(px.0[0], 255);
                        }
                    }
                    // Dimensions preserved
                    prop_assert_eq!(result.dimensions(), (len as u32, 1));
                }

                /// array_to_gray_image: values in [0, 1] map to proportional bytes
                #[test]
                fn valid_range_maps_proportionally(value in 0.0f32..=1.0f32) {
                    let arr = ndarray::arr2(&[[value]]);
                    let result = array_to_gray_image(&arr);
                    let byte = result.get_pixel(0, 0).0[0];

                    // Expected: (value * 255.0 + 0.5) as u8
                    let expected = (value * 255.0 + 0.5) as u8;
                    prop_assert_eq!(byte, expected);
                }
            }
        }
    }

    mod fill_mask_holes {
        use super::*;

        mod unit {
            use super::*;

            #[test]
            fn solid_white_unchanged() {
                let input = gray_image(4, 4, 255);
                let result = fill_mask_holes(&input, 128);
                for px in result.pixels() {
                    assert_eq!(px.0[0], 255);
                }
            }

            #[test]
            fn solid_black_unchanged() {
                let input = gray_image(4, 4, 0); // all black, connected to border
                let result = fill_mask_holes(&input, 128);
                for px in result.pixels() {
                    assert_eq!(px.0[0], 0); // stays black
                }
            }

            #[test]
            fn interior_hole_filled() {
                // Create a 5x5 image: white border, black interior
                // W W W W W
                // W B B B W
                // W B B B W
                // W B B B W
                // W W W W W
                let mut input = gray_image(5, 5, 255);
                for y in 1..4 {
                    for x in 1..4 {
                        input.put_pixel(x, y, Luma([0]));
                    }
                }

                let result = fill_mask_holes(&input, 128);

                // The interior black region is NOT connected to border, so it gets filled
                for px in result.pixels() {
                    assert_eq!(px.0[0], 255);
                }
            }

            #[test]
            fn border_connected_black_not_filled() {
                // Create a 4x4 image with black on left edge
                // B W W W
                // B W W W
                // B W W W
                // B W W W
                let mut input = gray_image(4, 4, 255);
                for y in 0..4 {
                    input.put_pixel(0, y, Luma([0]));
                }

                let result = fill_mask_holes(&input, 128);

                // Left column is connected to border, stays black (=0 in output)
                for y in 0..4 {
                    assert_eq!(result.get_pixel(0, y).0[0], 0);
                }
                // Rest stays white
                for y in 0..4 {
                    for x in 1..4 {
                        assert_eq!(result.get_pixel(x, y).0[0], 255);
                    }
                }
            }

            #[test]
            fn dimensions_preserved() {
                let input = gray_image(7, 5, 128);
                let result = fill_mask_holes(&input, 128);
                assert_eq!(result.dimensions(), (7, 5));
            }

            #[test]
            fn diagonal_only_connection_not_traversed() {
                // 4-connectivity: diagonal doesn't count as connected
                // W W W
                // W B W
                // B W W  (bottom-left black only diagonally connected to center)
                let mut input = GrayImage::new(3, 3);
                for y in 0..3 {
                    for x in 0..3 {
                        input.put_pixel(x, y, Luma([255]));
                    }
                }
                input.put_pixel(1, 1, Luma([0])); // center black
                input.put_pixel(0, 2, Luma([0])); // corner black (touches border)

                let result = fill_mask_holes(&input, 128);

                // corner (0,2) is on border, stays black
                assert_eq!(result.get_pixel(0, 2).0[0], 0);
                // center (1,1) is NOT 4-connected to border, gets filled
                assert_eq!(result.get_pixel(1, 1).0[0], 255);
            }

            #[test]
            fn threshold_changes_border_connectivity() {
                // pixel value 110 on border:
                // thr=128: 110 < 128, dark, BFS marks visited, output 0
                // thr=100: 110 >= 100, light, not visited by BFS, output 255
                let mut input = gray_image(3, 3, 255);
                input.put_pixel(0, 1, Luma([110])); // left border

                let r128 = fill_mask_holes(&input, 128);
                assert_eq!(r128.get_pixel(0, 1).0[0], 0);

                let r100 = fill_mask_holes(&input, 100);
                assert_eq!(r100.get_pixel(0, 1).0[0], 255);
            }

            #[test]
            fn output_is_always_binary() {
                // fill_mask_holes should always produce binary output (0 or 255)
                let mut input = gray_image(5, 5, 200); // non-binary input
                input.put_pixel(2, 2, Luma([50])); // interior dark pixel

                let result = fill_mask_holes(&input, 128);

                let is_binary = result.pixels().all(|p| p.0[0] == 0 || p.0[0] == 255);
                assert!(is_binary);
            }
        }

        mod prop {
            use super::*;
            use proptest::prelude::*;

            proptest! {
                /// fill_mask_holes: output is always binary and dimensions preserved
                #[test]
                fn output_is_binary(
                    w in 1u32..15,
                    h in 1u32..15,
                    fill_value in proptest::num::u8::ANY,
                    threshold in proptest::num::u8::ANY
                ) {
                    let input = GrayImage::from_pixel(w, h, Luma([fill_value]));
                    let result = fill_mask_holes(&input, threshold);

                    prop_assert_eq!(result.dimensions(), (w, h));
                    for px in result.pixels() {
                        prop_assert!(px.0[0] == 0 || px.0[0] == 255);
                    }
                }
            }
        }
    }

    mod dilate_euclidean {
        use super::*;

        mod unit {
            use super::*;

            #[test]
            fn solid_white_stays_white() {
                let input = gray_image(4, 4, 255);
                let result = dilate_euclidean(&input, 2.0);
                for px in result.pixels() {
                    assert_eq!(px.0[0], 255);
                }
            }

            #[test]
            fn solid_black_stays_black() {
                let input = gray_image(4, 4, 0);
                let result = dilate_euclidean(&input, 2.0);
                for px in result.pixels() {
                    assert_eq!(px.0[0], 0);
                }
            }

            #[test]
            fn single_white_pixel_dilates() {
                // 5x5 black image with center pixel white
                let mut input = gray_image(5, 5, 0);
                input.put_pixel(2, 2, Luma([255]));

                let result = dilate_euclidean(&input, 1.5);

                // Center should be white
                assert_eq!(result.get_pixel(2, 2).0[0], 255);
                // Neighbors within radius should also be white
                assert_eq!(result.get_pixel(2, 1).0[0], 255);
                assert_eq!(result.get_pixel(2, 3).0[0], 255);
                assert_eq!(result.get_pixel(1, 2).0[0], 255);
                assert_eq!(result.get_pixel(3, 2).0[0], 255);
            }

            #[test]
            fn dimensions_preserved() {
                let input = gray_image(6, 4, 128);
                let result = dilate_euclidean(&input, 1.0);
                assert_eq!(result.dimensions(), (6, 4));
            }

            #[test]
            fn diagonal_within_radius() {
                // r=1.5, diagonal distance = sqrt(2) ~ 1.414 < 1.5
                let mut input = gray_image(5, 5, 0);
                input.put_pixel(2, 2, Luma([255]));

                let result = dilate_euclidean(&input, 1.5);

                // diagonals should be white (euclidean distance ~1.414)
                assert_eq!(result.get_pixel(1, 1).0[0], 255);
                assert_eq!(result.get_pixel(3, 1).0[0], 255);
                assert_eq!(result.get_pixel(1, 3).0[0], 255);
                assert_eq!(result.get_pixel(3, 3).0[0], 255);
            }

            #[test]
            fn radius_zero_only_original_pixels() {
                // r=0: only original white pixels stay white
                let mut input = gray_image(3, 3, 0);
                input.put_pixel(1, 1, Luma([255]));

                let result = dilate_euclidean(&input, 0.0);

                assert_eq!(result.get_pixel(1, 1).0[0], 255);
                // neighbors should stay black
                assert_eq!(result.get_pixel(0, 1).0[0], 0);
                assert_eq!(result.get_pixel(2, 1).0[0], 0);
                assert_eq!(result.get_pixel(1, 0).0[0], 0);
                assert_eq!(result.get_pixel(1, 2).0[0], 0);
            }

            #[test]
            fn negative_radius_preserves_mask() {
                let mut input = gray_image(3, 3, 0);
                input.put_pixel(0, 0, Luma([255]));
                input.put_pixel(1, 1, Luma([255]));
                input.put_pixel(2, 2, Luma([255]));

                let result = dilate_euclidean(&input, -1.0);

                assert_eq!(result.as_raw(), input.as_raw());
            }
        }

        mod prop {
            use super::*;
            use proptest::prelude::*;

            proptest! {
                /// dilate_euclidean: output is always binary and dimensions preserved
                #[test]
                fn output_is_binary(
                    w in 1u32..15,
                    h in 1u32..15,
                    fill_value in proptest::num::u8::ANY,
                    radius in 0.0f32..5.0f32
                ) {
                    let input = GrayImage::from_pixel(w, h, Luma([fill_value]));
                    let result = dilate_euclidean(&input, radius);

                    prop_assert_eq!(result.dimensions(), (w, h));
                    for px in result.pixels() {
                        prop_assert!(px.0[0] == 0 || px.0[0] == 255);
                    }
                }
            }
        }
    }

    mod erode_euclidean_tests {
        use super::*;
        use crate::config::ErosionBorderMode;

        #[test]
        fn erosion_shrinks_binary_mask() {
            let mut input = gray_image(5, 5, 0);
            for y in 1..4 {
                for x in 1..4 {
                    input.put_pixel(x, y, Luma([255]));
                }
            }

            let result =
                erode_euclidean_with_border_mode(&input, 1.0, ErosionBorderMode::default());

            assert_eq!(result.get_pixel(2, 2).0[0], 255);
            assert_eq!(result.get_pixel(1, 1).0[0], 0);
            assert_eq!(result.get_pixel(1, 2).0[0], 0);
            assert_eq!(result.get_pixel(2, 1).0[0], 0);
            assert_eq!(result.get_pixel(3, 3).0[0], 0);
        }

        #[test]
        fn erosion_treats_image_exterior_as_background() {
            let input = gray_image(5, 5, 255);

            let result =
                erode_euclidean_with_border_mode(&input, 1.0, ErosionBorderMode::default());

            for y in 0..5 {
                for x in 0..5 {
                    let expected = if x == 0 || y == 0 || x == 4 || y == 4 {
                        0
                    } else {
                        255
                    };
                    assert_eq!(result.get_pixel(x, y).0[0], expected, "pixel ({x}, {y})");
                }
            }
        }

        #[test]
        fn erosion_can_treat_image_exterior_as_unknown() {
            let input = gray_image(5, 5, 255);

            let result =
                erode_euclidean_with_border_mode(&input, 1.0, ErosionBorderMode::OutsideIsUnknown);

            assert_eq!(result.as_raw(), input.as_raw());
        }

        #[test]
        fn erosion_radius_zero_preserves_mask() {
            let mut input = gray_image(3, 3, 0);
            input.put_pixel(0, 0, Luma([255]));
            input.put_pixel(1, 1, Luma([255]));
            input.put_pixel(2, 2, Luma([255]));

            let result =
                erode_euclidean_with_border_mode(&input, 0.0, ErosionBorderMode::default());

            assert_eq!(result.as_raw(), input.as_raw());
        }

        #[test]
        fn erosion_negative_radius_preserves_mask() {
            let mut input = gray_image(3, 3, 0);
            input.put_pixel(0, 0, Luma([255]));
            input.put_pixel(1, 1, Luma([255]));
            input.put_pixel(2, 2, Luma([255]));

            let result =
                erode_euclidean_with_border_mode(&input, -1.0, ErosionBorderMode::default());

            assert_eq!(result.as_raw(), input.as_raw());
        }

        #[test]
        fn erosion_shrinks_foreground_touching_one_edge() {
            let mut input = gray_image(5, 5, 0);
            for y in 1..4 {
                for x in 0..3 {
                    input.put_pixel(x, y, Luma([255]));
                }
            }

            let result =
                erode_euclidean_with_border_mode(&input, 1.0, ErosionBorderMode::default());

            for y in 0..5 {
                for x in 0..5 {
                    let expected = if x == 1 && y == 2 { 255 } else { 0 };
                    assert_eq!(result.get_pixel(x, y).0[0], expected, "pixel ({x}, {y})");
                }
            }
        }
    }

    mod apply_operations {
        use super::*;

        mod unit {
            use super::*;

            #[test]
            fn empty_operations_returns_clone() {
                let input = gray_image(3, 3, 100);
                let result = apply_operations(&input, &[]);
                assert_eq!(result.as_raw(), input.as_raw());
            }

            #[test]
            fn single_threshold_operation() {
                let input = gray_image(2, 2, 200);
                let ops = vec![MaskOperation::Threshold { value: 128 }];
                let result = apply_operations(&input, &ops);
                for px in result.pixels() {
                    assert_eq!(px.0[0], 255); // 200 > 128
                }
            }

            #[test]
            fn threshold_then_dilate() {
                let mut input = gray_image(5, 5, 0);
                input.put_pixel(2, 2, Luma([200]));

                // threshold (200 > 128 = white), then dilate expands it
                let ops = vec![
                    MaskOperation::Threshold { value: 128 },
                    MaskOperation::Dilate { radius: 1.0 },
                ];
                let result = apply_operations(&input, &ops);

                // center and neighbors should be white
                assert_eq!(result.get_pixel(2, 2).0[0], 255);
                assert_eq!(result.get_pixel(2, 1).0[0], 255);
            }

            #[test]
            fn threshold_then_erode() {
                let mut input = gray_image(5, 5, 0);
                for y in 1..4 {
                    for x in 1..4 {
                        input.put_pixel(x, y, Luma([200]));
                    }
                }

                let ops = vec![
                    MaskOperation::Threshold { value: 128 },
                    MaskOperation::Erode {
                        radius: 1.0,
                        border_mode: ErosionBorderMode::default(),
                    },
                ];
                let result = apply_operations(&input, &ops);

                assert_eq!(result.get_pixel(2, 2).0[0], 255);
                assert_eq!(result.get_pixel(1, 1).0[0], 0);
            }

            #[test]
            fn order_matters_blur_vs_threshold() {
                // blur then threshold vs threshold then blur produce different results
                let mut input = gray_image(5, 5, 0);
                input.put_pixel(2, 2, Luma([255])); // single white pixel

                // blur first spreads the white, then threshold produces binary output
                let ops_blur_first = vec![
                    MaskOperation::Blur { sigma: 1.0 },
                    MaskOperation::Threshold { value: 50 },
                ];
                let result_blur_first = apply_operations(&input, &ops_blur_first);

                // threshold first (255 > 50 = white), then blur produces soft edges
                let ops_threshold_first = vec![
                    MaskOperation::Threshold { value: 50 },
                    MaskOperation::Blur { sigma: 1.0 },
                ];
                let result_threshold_first = apply_operations(&input, &ops_threshold_first);

                // blur->threshold is binary (all pixels are 0 or 255)
                let is_binary = result_blur_first
                    .pixels()
                    .all(|p| p.0[0] == 0 || p.0[0] == 255);
                assert!(is_binary);

                // threshold->blur has intermediate grays (some 0 < value < 255)
                let has_intermediate = result_threshold_first
                    .pixels()
                    .any(|p| p.0[0] > 0 && p.0[0] < 255);
                assert!(has_intermediate);

                // fallback: overall results must differ
                assert_ne!(result_blur_first.as_raw(), result_threshold_first.as_raw());
            }
        }

        mod prop {
            use super::*;
            use proptest::prelude::*;

            proptest! {
                /// apply_operations: dimensions always preserved regardless of operations
                #[test]
                fn preserves_dimensions(
                    w in 1u32..10,
                    h in 1u32..10,
                    fill_value in proptest::num::u8::ANY
                ) {
                    let input = GrayImage::from_pixel(w, h, Luma([fill_value]));

                    // Test with various operation combinations
                    let ops_threshold = vec![MaskOperation::Threshold { value: 128 }];
                    let result = apply_operations(&input, &ops_threshold);
                    prop_assert_eq!(result.dimensions(), (w, h));

                    let ops_dilate = vec![MaskOperation::Dilate { radius: 1.0 }];
                    let result = apply_operations(&input, &ops_dilate);
                    prop_assert_eq!(result.dimensions(), (w, h));

                    let ops_erode = vec![MaskOperation::Erode {
                        radius: 1.0,
                        border_mode: ErosionBorderMode::default(),
                    }];
                    let result = apply_operations(&input, &ops_erode);
                    prop_assert_eq!(result.dimensions(), (w, h));

                    let ops_fill = vec![MaskOperation::FillHoles { threshold: 128 }];
                    let result = apply_operations(&input, &ops_fill);
                    prop_assert_eq!(result.dimensions(), (w, h));

                    let ops_blur = vec![MaskOperation::Blur { sigma: 1.0 }];
                    let result = apply_operations(&input, &ops_blur);
                    prop_assert_eq!(result.dimensions(), (w, h));
                }
            }
        }
    }

    mod operations_from_options {
        use super::*;

        mod unit {
            use super::*;

            #[test]
            fn all_disabled_returns_empty() {
                let opts = MaskProcessingOptions {
                    blur: false,
                    binary: false,
                    dilate: false,
                    erode: false,
                    fill_holes: false,
                    ..Default::default()
                };
                let ops = operations_from_options(&opts);
                assert!(ops.is_empty());
            }

            #[test]
            fn blur_only() {
                let opts = MaskProcessingOptions {
                    blur: true,
                    blur_sigma: 3.0,
                    ..Default::default()
                };
                let ops = operations_from_options(&opts);
                assert_eq!(ops.len(), 1);
                assert!(
                    matches!(ops[0], MaskOperation::Blur { sigma } if (sigma - 3.0).abs() < 0.001)
                );
            }

            #[test]
            fn full_pipeline_order_and_values() {
                // order: blur, threshold, dilate, erode, fill_holes
                let opts = MaskProcessingOptions {
                    blur: true,
                    blur_sigma: 2.0,
                    binary: true,
                    mask_threshold: 128,
                    dilate: true,
                    dilation_radius: 5.0,
                    erode: true,
                    erosion_radius: 3.0,
                    erosion_border_mode: ErosionBorderMode::OutsideIsUnknown,
                    fill_holes: true,
                };
                let ops = operations_from_options(&opts);
                assert_eq!(ops.len(), 5);
                assert!(
                    matches!(ops[0], MaskOperation::Blur { sigma } if (sigma - 2.0).abs() < 1e-6)
                );
                assert!(matches!(ops[1], MaskOperation::Threshold { value: 128 }));
                assert!(
                    matches!(ops[2], MaskOperation::Dilate { radius } if (radius - 5.0).abs() < 1e-6)
                );
                assert!(matches!(
                    ops[3],
                    MaskOperation::Erode { radius, border_mode }
                        if (radius - 3.0).abs() < 1e-6
                            && border_mode == ErosionBorderMode::OutsideIsUnknown
                ));
                assert!(matches!(
                    ops[4],
                    MaskOperation::FillHoles { threshold: 128 }
                ));
            }

            #[test]
            fn partial_pipeline_skips_disabled() {
                let opts = MaskProcessingOptions {
                    blur: false,
                    binary: true,
                    mask_threshold: 100,
                    dilate: false,
                    erode: false,
                    fill_holes: true,
                    ..Default::default()
                };
                let ops = operations_from_options(&opts);
                assert_eq!(ops.len(), 2);
                assert!(matches!(ops[0], MaskOperation::Threshold { value: 100 }));
                assert!(matches!(
                    ops[1],
                    MaskOperation::FillHoles { threshold: 100 }
                ));
            }
        }
    }

    mod mask_handle_api {
        use super::*;
        use image::Rgb;

        fn mask_handle() -> MaskHandle {
            MaskHandle {
                rgb_image: Arc::new(RgbImage::from_pixel(1, 1, Rgb([255, 255, 255]))),
                mask: GrayImage::from_pixel(1, 1, Luma([255])),
                default_mask_processing: MaskProcessingOptions::default(),
                operations: Vec::new(),
            }
        }

        fn mask_handle_with_images(rgb: RgbImage, mask: GrayImage) -> MaskHandle {
            MaskHandle {
                rgb_image: Arc::new(rgb),
                mask,
                default_mask_processing: MaskProcessingOptions::default(),
                operations: Vec::new(),
            }
        }

        fn single_pixel_mask_handle() -> MaskHandle {
            mask_handle_with_images(
                RgbImage::from_pixel(5, 5, Rgb([10, 20, 30])),
                GrayImage::from_fn(5, 5, |x, y| {
                    if x == 2 && y == 2 {
                        Luma([255])
                    } else {
                        Luma([0])
                    }
                }),
            )
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

        mod colorize {
            use super::*;

            #[test]
            fn default_color_is_white_with_mask_alpha() {
                let color = MaskColor::default();
                assert_eq!(color.color, [255, 255, 255, 255]);
                assert!(matches!(color.alpha_mode, MaskAlphaMode::UseMask));
            }

            #[test]
            fn use_mask_mode_sets_alpha_from_mask() {
                let mask = gray_image(2, 2, 128);
                let result = colorize_mask(&mask, MaskColor::new([0, 180, 255, 255]));

                for px in result.pixels() {
                    assert_eq!(px.0, [0, 180, 255, 128]);
                }
            }

            #[test]
            fn base_alpha_is_multiplied_with_mask_alpha() {
                let mask = gray_image(1, 1, 128);
                let result = colorize_mask(&mask, MaskColor::new([255, 255, 255, 128]));

                assert_eq!(result.get_pixel(0, 0).0, [255, 255, 255, 64]);
            }

            #[test]
            fn scale_mode_scales_mask_alpha() {
                let mask = gray_image(1, 1, 200);
                let color =
                    MaskColor::new([255, 0, 0, 255]).with_alpha_mode(MaskAlphaMode::Scale(0.5));
                let result = colorize_mask(&mask, color);

                assert_eq!(result.get_pixel(0, 0).0, [255, 0, 0, 100]);
            }

            #[test]
            fn solid_mode_uses_alpha_for_nonzero_mask() {
                let mask =
                    GrayImage::from_fn(2, 1, |x, _| if x == 0 { Luma([0]) } else { Luma([25]) });
                let color =
                    MaskColor::new([0, 255, 0, 255]).with_alpha_mode(MaskAlphaMode::Solid(200));
                let result = colorize_mask(&mask, color);

                assert_eq!(result.get_pixel(0, 0).0, [0, 255, 0, 0]);
                assert_eq!(result.get_pixel(1, 0).0, [0, 255, 0, 200]);
            }
        }

        mod erode_builder {
            use super::*;

            #[test]
            fn mask_handle_erode_uses_default_radius() {
                let handle = mask_handle().erode();
                assert!(matches!(
                    handle.operations.as_slice(),
                    [MaskOperation::Erode { radius, border_mode }]
                        if (*radius - MaskProcessingOptions::default().erosion_radius).abs() < f32::EPSILON
                            && *border_mode == MaskProcessingOptions::default().erosion_border_mode
                ));
            }

            #[test]
            fn mask_handle_erode_with_uses_custom_radius() {
                let handle = mask_handle().erode_with(3.0);
                assert!(matches!(
                    handle.operations.as_slice(),
                    [MaskOperation::Erode { radius, border_mode }]
                        if (*radius - 3.0).abs() < f32::EPSILON
                            && *border_mode == MaskProcessingOptions::default().erosion_border_mode
                ));
            }

            #[test]
            fn mask_handle_erode_with_border_mode_uses_custom_mode() {
                let handle =
                    mask_handle().erode_with_border_mode(3.0, ErosionBorderMode::OutsideIsUnknown);
                assert!(matches!(
                    handle.operations.as_slice(),
                    [MaskOperation::Erode { radius, border_mode }]
                        if (*radius - 3.0).abs() < f32::EPSILON
                            && *border_mode == ErosionBorderMode::OutsideIsUnknown
                ));
            }
        }

        mod geometry {
            use super::*;

            #[test]
            fn mask_handle_pad_updates_mask_and_foreground_canvas() {
                let rgb = RgbImage::from_pixel(2, 2, Rgb([10, 20, 30]));
                let mut mask = GrayImage::from_pixel(2, 2, Luma([0]));
                mask.put_pixel(1, 1, Luma([255]));

                let padded = mask_handle_with_images(rgb, mask).pad(Padding::new(1, 2, 3, 4));

                assert_eq!(padded.image().dimensions(), (6, 8));
                let foreground = padded.foreground().expect("foreground should compose");
                assert_eq!(foreground.image().dimensions(), (6, 8));
                assert_eq!(foreground.image().get_pixel(2, 3)[3], 255);
            }

            #[test]
            fn mask_handle_crop_to_content_crops_mask_and_rgb_together() {
                let rgb = RgbImage::from_fn(4, 4, |x, y| Rgb([x as u8, y as u8, 0]));
                let mut mask = GrayImage::from_pixel(4, 4, Luma([0]));
                mask.put_pixel(2, 1, Luma([255]));
                mask.put_pixel(2, 2, Luma([255]));

                let cropped = mask_handle_with_images(rgb, mask)
                    .crop_to_content(1)
                    .expect("mask has content");

                assert_eq!(cropped.image().dimensions(), (3, 4));
                let foreground = cropped.foreground().expect("foreground should compose");
                assert_eq!(foreground.image().dimensions(), (3, 4));
                assert_eq!(foreground.image().get_pixel(1, 1)[0], 2);
                assert_eq!(foreground.image().get_pixel(1, 1)[1], 1);
                assert_eq!(foreground.image().get_pixel(1, 1)[3], 255);
            }
        }

        mod pending_operations {
            use super::*;

            #[test]
            fn mask_handle_pad_applies_pending_operations_first() {
                let padded = single_pixel_mask_handle()
                    .dilate_with(1.0)
                    .pad(Padding::new(1, 2, 0, 0));

                assert_eq!(
                    mask_bounding_box(padded.image(), 1),
                    Some(BoundingBox::new(2, 3, 3, 3))
                );
            }

            #[test]
            fn mask_handle_into_image_applies_pending_operations() {
                let mask = single_pixel_mask_handle().dilate_with(1.0).into_image();

                assert_eq!(
                    mask_bounding_box(&mask, 1),
                    Some(BoundingBox::new(1, 1, 3, 3))
                );
            }

            #[test]
            fn mask_handle_save_applies_pending_operations() {
                let temp_dir = tempfile::tempdir().expect("temp dir should be created");
                let path = temp_dir.path().join("mask.png");

                single_pixel_mask_handle()
                    .dilate_with(1.0)
                    .save(&path)
                    .expect("mask should save");

                let saved = image::open(&path)
                    .expect("saved mask should load")
                    .to_luma8();

                assert_eq!(
                    mask_bounding_box(&saved, 1),
                    Some(BoundingBox::new(1, 1, 3, 3))
                );
            }

            #[test]
            fn mask_handle_foreground_applies_pending_operations() {
                let foreground = single_pixel_mask_handle()
                    .dilate_with(1.0)
                    .foreground()
                    .expect("foreground should compose");

                assert_eq!(foreground.image().get_pixel(1, 2)[3], 255);
                assert_eq!(foreground.image().get_pixel(0, 0)[3], 0);
            }

            #[test]
            fn mask_handle_colorize_applies_pending_operations() {
                let colorized = single_pixel_mask_handle()
                    .dilate_with(1.0)
                    .colorize([0, 180, 255, 255]);

                assert_eq!(colorized.get_pixel(1, 2).0, [0, 180, 255, 255]);
                assert_eq!(colorized.get_pixel(0, 0).0, [0, 180, 255, 0]);
            }

            #[test]
            fn mask_handle_trace_applies_pending_operations() {
                let bounds = single_pixel_mask_handle()
                    .dilate_with(1.0)
                    .trace(&BoundingBoxVectorizer, &())
                    .expect("trace should run");

                assert_eq!(bounds, Some(BoundingBox::new(1, 1, 3, 3)));
            }
        }
    }

    #[cfg(feature = "vectorizer-vtracer")]
    mod gray_to_color_image_rgba {
        use super::*;

        mod unit {
            use super::*;

            #[test]
            fn passthrough_no_threshold_no_invert() {
                let mut input = GrayImage::new(2, 1);
                input.put_pixel(0, 0, Luma([0]));
                input.put_pixel(1, 0, Luma([128]));

                let result = gray_to_color_image_rgba(&input, None, false);

                assert_eq!(result.width, 2);
                assert_eq!(result.height, 1);
                // pixel 0: gray=0
                assert_eq!(result.pixels[0..4], [0, 0, 0, 255]);
                // pixel 1: gray=128
                assert_eq!(result.pixels[4..8], [128, 128, 128, 255]);
            }

            #[test]
            fn threshold_boundary() {
                // threshold=128: >= 128 becomes 255, < 128 becomes 0
                let mut input = GrayImage::new(3, 1);
                input.put_pixel(0, 0, Luma([127])); // below
                input.put_pixel(1, 0, Luma([128])); // exact
                input.put_pixel(2, 0, Luma([129])); // above

                let result = gray_to_color_image_rgba(&input, Some(128), false);

                assert_eq!(result.pixels[0..4], [0, 0, 0, 255]); // 127 < 128
                assert_eq!(result.pixels[4..8], [255, 255, 255, 255]); // 128 >= 128
                assert_eq!(result.pixels[8..12], [255, 255, 255, 255]); // 129 >= 128
            }

            #[test]
            fn invert_without_threshold() {
                let mut input = GrayImage::new(2, 1);
                input.put_pixel(0, 0, Luma([0]));
                input.put_pixel(1, 0, Luma([255]));

                let result = gray_to_color_image_rgba(&input, None, true);

                assert_eq!(result.pixels[0..4], [255, 255, 255, 255]); // 255 - 0
                assert_eq!(result.pixels[4..8], [0, 0, 0, 255]); // 255 - 255
            }

            #[test]
            fn invert_with_threshold() {
                let mut input = GrayImage::new(2, 1);
                input.put_pixel(0, 0, Luma([100])); // < 128, becomes 0, then inverted to 255
                input.put_pixel(1, 0, Luma([200])); // >= 128, becomes 255, then inverted to 0

                let result = gray_to_color_image_rgba(&input, Some(128), true);

                assert_eq!(result.pixels[0..4], [255, 255, 255, 255]);
                assert_eq!(result.pixels[4..8], [0, 0, 0, 255]);
            }

            #[test]
            fn alpha_always_255() {
                let input = gray_image(3, 3, 100);
                let result = gray_to_color_image_rgba(&input, None, false);

                // check every 4th byte (alpha channel)
                for i in 0..9 {
                    assert_eq!(result.pixels[i * 4 + 3], 255);
                }
            }

            #[test]
            fn dimensions_correct() {
                let input = gray_image(7, 5, 128);
                let result = gray_to_color_image_rgba(&input, None, false);

                assert_eq!(result.width, 7);
                assert_eq!(result.height, 5);
                assert_eq!(result.pixels.len(), 7 * 5 * 4);
            }
        }
    }
}
