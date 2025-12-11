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

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a solid gray image filled with a single value.
    fn gray_image(w: u32, h: u32, value: u8) -> GrayImage {
        GrayImage::from_pixel(w, h, image::Luma([value]))
    }

    /// Create a solid RGBA image filled with a single color.
    fn rgba_image(w: u32, h: u32, color: [u8; 4]) -> RgbaImage {
        RgbaImage::from_pixel(w, h, Rgba(color))
    }

    mod resolve_mask_alpha {
        use super::*;

        #[test]
        fn use_mask_passes_through() {
            assert_eq!(resolve_mask_alpha(0, MaskAlphaMode::UseMask), 0);
            assert_eq!(resolve_mask_alpha(128, MaskAlphaMode::UseMask), 128);
            assert_eq!(resolve_mask_alpha(255, MaskAlphaMode::UseMask), 255);
        }

        #[test]
        fn scale_by_half() {
            assert_eq!(resolve_mask_alpha(0, MaskAlphaMode::Scale(0.5)), 0);
            assert_eq!(resolve_mask_alpha(200, MaskAlphaMode::Scale(0.5)), 100);
            assert_eq!(resolve_mask_alpha(255, MaskAlphaMode::Scale(0.5)), 128);
        }

        #[test]
        fn scale_clamps_overflow() {
            // 200 * 2.0 = 400, clamped to 255
            assert_eq!(resolve_mask_alpha(200, MaskAlphaMode::Scale(2.0)), 255);
        }

        #[test]
        fn scale_negative_treated_as_zero() {
            assert_eq!(resolve_mask_alpha(255, MaskAlphaMode::Scale(-1.0)), 0);
        }

        #[test]
        fn solid_returns_alpha_for_nonzero() {
            assert_eq!(resolve_mask_alpha(0, MaskAlphaMode::Solid(200)), 0);
            assert_eq!(resolve_mask_alpha(1, MaskAlphaMode::Solid(200)), 200);
            assert_eq!(resolve_mask_alpha(255, MaskAlphaMode::Solid(200)), 200);
        }
    }

    mod create_rgba_layer_from_mask {
        use super::*;

        #[test]
        fn dimensions_preserved() {
            let mask = gray_image(8, 6, 128);
            let result = create_rgba_layer_from_mask(&mask, MaskFill::default());
            assert_eq!(result.dimensions(), (8, 6));
        }

        #[test]
        fn use_mask_mode_white_fill() {
            let mask = gray_image(2, 2, 128);
            let fill = MaskFill::new([255, 255, 255, 255]);
            let result = create_rgba_layer_from_mask(&mask, fill);

            // All pixels should be white with alpha=128
            for px in result.pixels() {
                assert_eq!(px.0, [255, 255, 255, 128]);
            }
        }

        #[test]
        fn use_mask_mode_colored_fill() {
            let mask = gray_image(2, 2, 255);
            let fill = MaskFill::new([255, 0, 0, 255]); // Red, fully opaque
            let result = create_rgba_layer_from_mask(&mask, fill);

            for px in result.pixels() {
                assert_eq!(px.0, [255, 0, 0, 255]);
            }
        }

        #[test]
        fn base_alpha_multiplied() {
            let mask = gray_image(2, 2, 255); // mask_alpha=255
            let fill = MaskFill::new([255, 255, 255, 128]); // base_alpha=128
            let result = create_rgba_layer_from_mask(&mask, fill);

            for px in result.pixels() {
                assert_eq!(px.0[3], 128); // (255*128)/255 = 128
            }
        }

        #[test]
        fn solid_mode() {
            let mask = gray_image(2, 2, 50); // non-zero mask
            let fill = MaskFill::new([0, 255, 0, 255]) // base_alpha=255
                .with_alpha_mode(MaskAlphaMode::Solid(200)); // solid_alpha=200
            let result = create_rgba_layer_from_mask(&mask, fill);

            for px in result.pixels() {
                assert_eq!(px.0, [0, 255, 0, 200]); // (200*255)/255 = 200
            }
        }

        #[test]
        fn zero_mask_produces_transparent() {
            let mask = gray_image(2, 2, 0);
            let fill = MaskFill::new([255, 0, 0, 255]);
            let result = create_rgba_layer_from_mask(&mask, fill);

            for px in result.pixels() {
                assert_eq!(px.0[3], 0);
            }
        }
    }

    mod alpha_composite {
        use super::*;

        #[test]
        fn opaque_top_replaces_bottom() {
            let bottom = rgba_image(2, 2, [255, 0, 0, 255]); // Red, opaque
            let top = rgba_image(2, 2, [0, 255, 0, 255]); // Green, opaque
            let result = alpha_composite(&bottom, &top);

            for px in result.pixels() {
                assert_eq!(px.0, [0, 255, 0, 255]);
            }
        }

        #[test]
        fn transparent_top_shows_bottom() {
            let bottom = rgba_image(2, 2, [255, 0, 0, 255]); // Red, opaque
            let top = rgba_image(2, 2, [0, 255, 0, 0]); // Green, transparent
            let result = alpha_composite(&bottom, &top);

            for px in result.pixels() {
                assert_eq!(px.0, [255, 0, 0, 255]);
            }
        }

        #[test]
        fn half_transparent_blends() {
            let bottom = rgba_image(2, 2, [0, 0, 0, 255]); // Black, opaque
            let top = rgba_image(2, 2, [255, 255, 255, 128]); // White, ~50% alpha
            let result = alpha_composite(&bottom, &top);

            // Expected: blended gray, fully opaque
            // out_a = 0.502 + 1.0 * (1 - 0.502) = 1.0
            // fg_weight = 0.502, bg_weight = 0.498
            // color = 255 * 0.502 + 0 * 0.498 = 128
            for px in result.pixels() {
                assert_eq!(px.0[3], 255); // Fully opaque result
                // Color should be around 128 (gray)
                assert!((px.0[0] as i32 - 128).abs() <= 2);
            }
        }

        #[test]
        fn both_transparent_stays_transparent() {
            let bottom = rgba_image(2, 2, [255, 0, 0, 0]);
            let top = rgba_image(2, 2, [0, 255, 0, 0]);
            let result = alpha_composite(&bottom, &top);

            for px in result.pixels() {
                assert_eq!(px.0[3], 0);
            }
        }
    }

    mod mask_fill {
        use super::*;

        #[test]
        fn default_is_white_opaque() {
            let fill = MaskFill::default();
            assert_eq!(fill.color, [255, 255, 255, 255]);
            assert!(matches!(fill.alpha_mode, MaskAlphaMode::UseMask));
        }

        #[test]
        fn with_alpha_mode_builder() {
            let fill =
                MaskFill::new([128, 128, 128, 255]).with_alpha_mode(MaskAlphaMode::Scale(0.5));
            assert_eq!(fill.color, [128, 128, 128, 255]);
            assert!(matches!(fill.alpha_mode, MaskAlphaMode::Scale(s) if (s - 0.5).abs() < 0.001));
        }

        #[test]
        fn scale_then_base_alpha_combined() {
            let mask = gray_image(1, 1, 128); // mask=128
            let fill = MaskFill::new([255, 255, 255, 128]) // base_alpha=128
                .with_alpha_mode(MaskAlphaMode::Scale(0.5)); // scale=0.5, scaled=64
            let result = create_rgba_layer_from_mask(&mask, fill);

            let px = result.get_pixel(0, 0);
            assert_eq!(px.0[3], 32); // (64 * 128) / 255 = 32
        }
    }

    mod overlay_image_on_mask {
        use super::*;

        #[test]
        fn dimension_mismatch_returns_error() {
            let layer_mask = gray_image(2, 2, 255);
            let overlay = rgba_image(3, 3, [255, 0, 0, 255]); // Different size

            let err =
                overlay_image_on_mask(&layer_mask, MaskFill::default(), &overlay).unwrap_err();

            match err {
                OutlineError::AlphaMismatch { expected, found } => {
                    assert_eq!(expected, (3, 3)); // overlay's dimensions
                    assert_eq!(found, (2, 2)); // mask's dimensions
                }
                other => panic!("unexpected error: {other:?}"),
            }
        }

        #[test]
        fn matching_dimensions_succeeds() {
            let layer_mask = gray_image(2, 2, 128);
            let overlay = rgba_image(2, 2, [255, 0, 0, 128]);

            let result = overlay_image_on_mask(&layer_mask, MaskFill::default(), &overlay);
            assert!(result.is_ok());
            assert_eq!(result.unwrap().dimensions(), (2, 2));
        }
    }
}
