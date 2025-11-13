use std::collections::VecDeque;

use image::{GrayImage, Luma};
use imageproc::contrast::{ThresholdType, threshold as ip_threshold};
use imageproc::distance_transform::Norm;
use imageproc::filter::gaussian_blur_f32;
use imageproc::morphology::dilate as ip_dilate;
use ndarray::Array2;

use crate::config::MaskProcessingOptions;

#[cfg(feature = "vectorizer-vtracer")]
use vtracer::ColorImage;

/// A single transformation step applied to a grayscale mask image.
#[derive(Debug, Clone)]
pub enum MaskOperation {
    Blur { sigma: f32 },
    Threshold { value: u8 },
    Dilate { radius: f32 },
    FillHoles { threshold: u8 },
}

impl MaskOperation {
    pub fn apply(&self, input: &GrayImage) -> GrayImage {
        match self {
            MaskOperation::Blur { sigma } => gaussian_blur_f32(input, *sigma),
            MaskOperation::Threshold { value } => threshold_mask(input, *value),
            MaskOperation::Dilate { radius } => dilate_euclidean(input, *radius),
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
    ip_dilate(mask_bin, Norm::L2, r.round() as u8)
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
