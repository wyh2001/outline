use std::io;
use std::path::Path;

use image::{GrayImage, RgbImage, Rgba, RgbaImage};

pub fn export_foreground(
    rgb: &RgbImage,
    alpha: &GrayImage,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if rgb.dimensions() != alpha.dimensions() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Alpha matte size does not match source image",
        )
        .into());
    }

    let (w, h) = rgb.dimensions();
    let mut rgba = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let rgb_px = rgb.get_pixel(x, y);
            let alpha_px = alpha.get_pixel(x, y);
            rgba.put_pixel(x, y, Rgba([rgb_px[0], rgb_px[1], rgb_px[2], alpha_px[0]]));
        }
    }

    rgba.save(output)?;
    Ok(())
}
