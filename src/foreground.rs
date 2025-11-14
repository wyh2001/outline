use image::{GrayImage, RgbImage, Rgba, RgbaImage};

use crate::{OutlineError, OutlineResult};

/// Compose an RGBA foreground image from an RGB image and a grayscale alpha matte.
pub fn compose_foreground(rgb: &RgbImage, alpha: &GrayImage) -> OutlineResult<RgbaImage> {
    let expected = rgb.dimensions();
    let found = alpha.dimensions();
    if expected != found {
        return Err(OutlineError::AlphaMismatch { expected, found });
    }

    let (w, h) = rgb.dimensions();
    let mut rgba = RgbaImage::new(w, h);
    // Use iterators instead of nested loops for better performance
    for ((rgb_px, alpha_px), out_px) in rgb.pixels().zip(alpha.pixels()).zip(rgba.pixels_mut()) {
        *out_px = Rgba([rgb_px[0], rgb_px[1], rgb_px[2], alpha_px[0]]);
    }

    Ok(rgba)
}
