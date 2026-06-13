#![cfg(any(feature = "backend-ort", feature = "backend-rten"))]

mod support;

use std::io::Cursor;

use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, ImageFormat, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use outline::Outline;
use tempfile::NamedTempFile;

fn tiny_outline() -> (NamedTempFile, Outline) {
    let model = support::tiny_matte_model_file();
    let outline = Outline::new(model.path())
        .with_input_resize_filter(FilterType::Nearest)
        .with_output_resize_filter(FilterType::Nearest);
    (model, outline)
}

fn rgb_input() -> RgbImage {
    RgbImage::from_fn(2, 2, |x, y| match (x, y) {
        (0, 0) => Rgb([10, 20, 30]),
        (1, 0) => Rgb([40, 50, 60]),
        (0, 1) => Rgb([70, 80, 90]),
        _ => Rgb([100, 110, 120]),
    })
}

fn assert_tiny_matte(matte: &GrayImage) {
    assert_eq!(matte.dimensions(), (2, 2));
    assert_eq!(matte.get_pixel(0, 0).0, [0]);
    assert_eq!(matte.get_pixel(1, 0).0, [64]);
    assert_eq!(matte.get_pixel(0, 1).0, [128]);
    assert_eq!(matte.get_pixel(1, 1).0, [255]);
}

#[test]
fn for_rgb_image_runs_pipeline() {
    let (_model, outline) = tiny_outline();
    let rgb = rgb_input();

    let result = outline
        .for_rgb_image(rgb.clone())
        .expect("RGB image inference should succeed");

    assert_eq!(result.rgb_image(), &rgb);
    assert_tiny_matte(result.raw_matte());
}

#[test]
fn for_rgba_image_discards_alpha_and_runs_pipeline() {
    let (_model, outline) = tiny_outline();
    let rgb = rgb_input();
    let rgba = RgbaImage::from_fn(2, 2, |x, y| {
        let pixel = rgb.get_pixel(x, y).0;
        Rgba([pixel[0], pixel[1], pixel[2], (x + y) as u8])
    });

    let result = outline
        .for_rgba_image(rgba)
        .expect("RGBA image inference should succeed");

    assert_eq!(result.rgb_image(), &rgb);
    assert_tiny_matte(result.raw_matte());
}

#[test]
fn for_dynamic_image_converts_to_rgb_and_runs_pipeline() {
    let (_model, outline) = tiny_outline();
    let gray = GrayImage::from_fn(2, 2, |x, y| Luma([((x + y * 2) * 40) as u8]));
    let expected_rgb = DynamicImage::ImageLuma8(gray.clone()).into_rgb8();

    let result = outline
        .for_dynamic_image(DynamicImage::ImageLuma8(gray))
        .expect("dynamic image inference should succeed");

    assert_eq!(result.rgb_image(), &expected_rgb);
    assert_tiny_matte(result.raw_matte());
}

#[test]
fn for_image_bytes_decodes_and_runs_pipeline() {
    let (_model, outline) = tiny_outline();
    let rgb = rgb_input();
    let mut encoded = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(rgb.clone())
        .write_to(&mut encoded, ImageFormat::Png)
        .expect("PNG encoding should succeed");

    let result = outline
        .for_image_bytes(encoded.get_ref())
        .expect("image bytes inference should succeed");

    assert_eq!(result.rgb_image(), &rgb);
    assert_tiny_matte(result.raw_matte());
}
