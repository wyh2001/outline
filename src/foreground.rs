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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgb};

    fn rgb_image(w: u32, h: u32, color: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(w, h, Rgb(color))
    }

    fn gray_image(w: u32, h: u32, value: u8) -> GrayImage {
        GrayImage::from_pixel(w, h, Luma([value]))
    }

    mod compose_foreground {
        use super::*;

        mod unit {
            use super::*;

            #[test]
            fn dimensions_match_success() {
                let rgb = rgb_image(2, 2, [255, 0, 0]);
                let alpha = gray_image(2, 2, 128);

                let result = compose_foreground(&rgb, &alpha).unwrap();
                assert_eq!(result.dimensions(), (2, 2));
            }

            #[test]
            fn alpha_too_small_returns_error() {
                let rgb = rgb_image(4, 4, [255, 0, 0]);
                let alpha = gray_image(2, 2, 128);

                let err = compose_foreground(&rgb, &alpha).unwrap_err();
                match err {
                    OutlineError::AlphaMismatch { expected, found } => {
                        assert_eq!(expected, (4, 4));
                        assert_eq!(found, (2, 2));
                    }
                    other => panic!("unexpected error: {other:?}"),
                }
            }

            #[test]
            fn alpha_too_large_returns_error() {
                let rgb = rgb_image(2, 2, [255, 0, 0]);
                let alpha = gray_image(4, 4, 128);

                let err = compose_foreground(&rgb, &alpha).unwrap_err();
                match err {
                    OutlineError::AlphaMismatch { expected, found } => {
                        assert_eq!(expected, (2, 2));
                        assert_eq!(found, (4, 4));
                    }
                    other => panic!("unexpected error: {other:?}"),
                }
            }

            #[test]
            fn full_transparency() {
                let rgb = rgb_image(2, 2, [255, 128, 64]);
                let alpha = gray_image(2, 2, 0);

                let result = compose_foreground(&rgb, &alpha).unwrap();
                for px in result.pixels() {
                    assert_eq!(px.0, [255, 128, 64, 0]);
                }
            }

            #[test]
            fn full_opacity() {
                let rgb = rgb_image(2, 2, [100, 150, 200]);
                let alpha = gray_image(2, 2, 255);

                let result = compose_foreground(&rgb, &alpha).unwrap();
                for px in result.pixels() {
                    assert_eq!(px.0, [100, 150, 200, 255]);
                }
            }

            #[test]
            fn rgb_channels_preserved() {
                let rgb = rgb_image(1, 1, [10, 20, 30]);
                let alpha = gray_image(1, 1, 128);

                let result = compose_foreground(&rgb, &alpha).unwrap();
                let px = result.get_pixel(0, 0);
                assert_eq!(px.0[0], 10); // R
                assert_eq!(px.0[1], 20); // G
                assert_eq!(px.0[2], 30); // B
                assert_eq!(px.0[3], 128); // A
            }

            #[test]
            fn per_pixel_values_correct() {
                // 2x2 image with different RGB and alpha per pixel
                let mut rgb = RgbImage::new(2, 2);
                rgb.put_pixel(0, 0, Rgb([10, 20, 30]));
                rgb.put_pixel(1, 0, Rgb([40, 50, 60]));
                rgb.put_pixel(0, 1, Rgb([70, 80, 90]));
                rgb.put_pixel(1, 1, Rgb([100, 110, 120]));

                let mut alpha = GrayImage::new(2, 2);
                alpha.put_pixel(0, 0, Luma([0]));
                alpha.put_pixel(1, 0, Luma([85]));
                alpha.put_pixel(0, 1, Luma([170]));
                alpha.put_pixel(1, 1, Luma([255]));

                let result = compose_foreground(&rgb, &alpha).unwrap();

                assert_eq!(result.get_pixel(0, 0).0, [10, 20, 30, 0]);
                assert_eq!(result.get_pixel(1, 0).0, [40, 50, 60, 85]);
                assert_eq!(result.get_pixel(0, 1).0, [70, 80, 90, 170]);
                assert_eq!(result.get_pixel(1, 1).0, [100, 110, 120, 255]);
            }
        }

        mod prop {
            use super::*;
            use proptest::prelude::*;

            proptest! {
                /// compose_foreground: dimensions preserved when inputs match
                #[test]
                fn dimensions_preserved(
                    w in 1u32..20,
                    h in 1u32..20,
                    rgb_val in proptest::num::u8::ANY,
                    alpha_val in proptest::num::u8::ANY
                ) {
                    let rgb = RgbImage::from_pixel(w, h, Rgb([rgb_val, rgb_val, rgb_val]));
                    let alpha = GrayImage::from_pixel(w, h, Luma([alpha_val]));
                    let result = compose_foreground(&rgb, &alpha).unwrap();

                    prop_assert_eq!(result.dimensions(), (w, h));
                }

                /// compose_foreground: RGB channels are always preserved from input
                #[test]
                fn rgb_channels_preserved(
                    r in proptest::num::u8::ANY,
                    g in proptest::num::u8::ANY,
                    b in proptest::num::u8::ANY,
                    alpha_val in proptest::num::u8::ANY
                ) {
                    let rgb = RgbImage::from_pixel(1, 1, Rgb([r, g, b]));
                    let alpha = GrayImage::from_pixel(1, 1, Luma([alpha_val]));
                    let result = compose_foreground(&rgb, &alpha).unwrap();
                    let px = result.get_pixel(0, 0);

                    prop_assert_eq!(px.0[0], r);
                    prop_assert_eq!(px.0[1], g);
                    prop_assert_eq!(px.0[2], b);
                }

                /// compose_foreground: alpha channel equals the input grayscale value
                #[test]
                fn alpha_from_grayscale(
                    alpha_val in proptest::num::u8::ANY
                ) {
                    let rgb = RgbImage::from_pixel(1, 1, Rgb([128, 128, 128]));
                    let alpha = GrayImage::from_pixel(1, 1, Luma([alpha_val]));
                    let result = compose_foreground(&rgb, &alpha).unwrap();
                    let px = result.get_pixel(0, 0);

                    prop_assert_eq!(px.0[3], alpha_val);
                }

                /// compose_foreground: dimension mismatch always returns error
                #[test]
                fn dimension_mismatch_errors(
                    rgb_w in 1u32..10,
                    rgb_h in 1u32..10,
                    alpha_w in 1u32..10,
                    alpha_h in 1u32..10
                ) {
                    // Skip cases where dimensions match
                    prop_assume!(rgb_w != alpha_w || rgb_h != alpha_h);

                    let rgb = RgbImage::from_pixel(rgb_w, rgb_h, Rgb([128, 128, 128]));
                    let alpha = GrayImage::from_pixel(alpha_w, alpha_h, Luma([128]));
                    let result = compose_foreground(&rgb, &alpha);

                    prop_assert!(result.is_err());
                    if let Err(crate::OutlineError::AlphaMismatch { expected, found }) = result {
                        prop_assert_eq!(expected, (rgb_w, rgb_h));
                        prop_assert_eq!(found, (alpha_w, alpha_h));
                    }
                }

                /// compose_foreground: output RGBA matches input RGB + alpha exactly
                #[test]
                fn output_matches_inputs_exactly(
                    r in proptest::num::u8::ANY,
                    g in proptest::num::u8::ANY,
                    b in proptest::num::u8::ANY,
                    a in proptest::num::u8::ANY
                ) {
                    let rgb = RgbImage::from_pixel(1, 1, Rgb([r, g, b]));
                    let alpha = GrayImage::from_pixel(1, 1, Luma([a]));
                    let result = compose_foreground(&rgb, &alpha).unwrap();
                    let px = result.get_pixel(0, 0);

                    // Output should be exactly [r, g, b, a]
                    prop_assert_eq!(px.0, [r, g, b, a]);
                }
            }
        }
    }
}
