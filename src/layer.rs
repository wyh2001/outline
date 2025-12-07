use image::{GrayImage, RgbImage, Rgba, RgbaImage};

use crate::foreground::compose_foreground;
use crate::{OutlineError, OutlineResult};

/// How the grayscale mask values drive the alpha channel of a filled layer.
/// - `UseMask`: use the mask value directly as alpha.
/// - `Scale(f32)`: scale the mask alpha by a factor (0.0-1.0 typical; values >1.0 get clamped).
/// - `Solid(u8)`: treat any non-zero mask value as a solid alpha (before multiplying by the fill color's alpha).
#[derive(Debug, Clone, Copy)]
pub enum MaskAlphaMode {
    /// Use the mask value directly as alpha.
    UseMask,
    /// Scale the mask alpha by a factor (0.0-1.0 typical; values >1.0 get clamped).
    Scale(f32),
    /// Treat any non-zero mask value as a solid alpha.
    /// (Before multiplying by the fill color's alpha.)
    Solid(u8),
}

/// Options for turning a mask into a colored RGBA layer.
#[derive(Debug, Clone, Copy)]
pub struct MaskFill {
    /// RGBA color applied to mask-covered pixels; A acts as a global opacity multiplier.
    pub color: [u8; 4],
    /// How mask values influence the output alpha.
    pub alpha_mode: MaskAlphaMode,
}

impl MaskFill {
    /// Create a fill that uses the mask as-is for alpha.
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

impl Default for MaskFill {
    fn default() -> Self {
        Self::new([255, 255, 255, 255])
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

/// Turn a grayscale mask into a colored RGBA layer.
/// Note: final alpha = mask-derived alpha (per [`MaskAlphaMode`]) × `color[3]` / 255.
pub fn create_rgba_layer_from_mask(mask: &GrayImage, fill: MaskFill) -> RgbaImage {
    let (w, h) = mask.dimensions();
    let [r, g, b, base_alpha] = fill.color;

    let mut out = RgbaImage::new(w, h);
    for (mask_px, out_px) in mask.pixels().zip(out.pixels_mut()) {
        let mask_alpha = resolve_mask_alpha(mask_px[0], fill.alpha_mode);
        let alpha = ((mask_alpha as u16 * base_alpha as u16) / 255) as u8;
        *out_px = Rgba([r, g, b, alpha]);
    }

    out
}

/// Alpha composite `top` over `bottom` (RGBA over operator).
pub fn alpha_composite(bottom: &RgbaImage, top: &RgbaImage) -> RgbaImage {
    let (w, h) = bottom.dimensions();
    let mut out = RgbaImage::new(w, h);

    for ((bg_px, fg_px), out_px) in bottom.pixels().zip(top.pixels()).zip(out.pixels_mut()) {
        let fg_a = fg_px[3] as f32 / 255.0;
        let bg_a = bg_px[3] as f32 / 255.0;
        let out_a = fg_a + bg_a * (1.0 - fg_a);

        let mut rgba = [0u8; 4];
        if out_a > 0.0 {
            let fg_weight = fg_a / out_a;
            let bg_weight = (bg_a * (1.0 - fg_a)) / out_a;
            for c in 0..3 {
                let fg_c = fg_px[c] as f32;
                let bg_c = bg_px[c] as f32;
                let blended = fg_c * fg_weight + bg_c * bg_weight;
                rgba[c] = blended.round().clamp(0.0, 255.0) as u8;
            }
        }
        rgba[3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
        *out_px = Rgba(rgba);
    }

    out
}

/// Compose a colored mask layer and a cut-out foreground layer.
///
/// - `layer_mask`: drives the colored underlay.
/// - `foreground_mask`: drives the foreground alpha cut-out.
pub fn overlay_foreground_on_layer(
    rgb: &RgbImage,
    foreground_mask: &GrayImage,
    layer_mask: &GrayImage,
    fill: MaskFill,
) -> OutlineResult<RgbaImage> {
    let expected = rgb.dimensions();

    if layer_mask.dimensions() != expected {
        return Err(OutlineError::AlphaMismatch {
            expected,
            found: layer_mask.dimensions(),
        });
    }

    let base_layer = create_rgba_layer_from_mask(layer_mask, fill);
    let foreground = compose_foreground(rgb, foreground_mask)?;

    if foreground.dimensions() != base_layer.dimensions() {
        return Err(OutlineError::AlphaMismatch {
            expected: base_layer.dimensions(),
            found: foreground.dimensions(),
        });
    }

    Ok(alpha_composite(&base_layer, &foreground))
}

/// Compose an arbitrary RGBA image on top of a mask-derived filled layer.
///
/// Useful when the overlay is not tied to the original RGB source.
pub fn overlay_image_on_mask(
    layer_mask: &GrayImage,
    fill: MaskFill,
    overlay: &RgbaImage,
) -> OutlineResult<RgbaImage> {
    let expected = overlay.dimensions();

    if layer_mask.dimensions() != expected {
        return Err(OutlineError::AlphaMismatch {
            expected,
            found: layer_mask.dimensions(),
        });
    }

    let base_layer = create_rgba_layer_from_mask(layer_mask, fill);
    Ok(alpha_composite(&base_layer, overlay))
}
