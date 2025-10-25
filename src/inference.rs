use std::convert::TryFrom;
use std::io;

use image::imageops::FilterType;
use image::{ImageBuffer, Luma, RgbImage};
use ndarray::{Array2, Array4, ArrayViewD, Axis, Ix2};
use ort::session::Session;
use ort::value::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelLayout {
    Nchw,
    Nhwc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelInputSpec {
    pub width: usize,
    pub height: usize,
    pub layout: ChannelLayout,
}

pub const DEFAULT_MODEL_INPUT_SPEC: ModelInputSpec = ModelInputSpec {
    width: 320,
    height: 320,
    layout: ChannelLayout::Nchw,
};

/// Tries to figure out the model input spec from the session and falls back to the default.
pub fn determine_model_input_spec(session: &Session) -> ModelInputSpec {
    infer_model_input_spec(session).unwrap_or(DEFAULT_MODEL_INPUT_SPEC)
}

fn infer_model_input_spec(session: &Session) -> Option<ModelInputSpec> {
    let input = session.inputs.get(0)?;
    let shape = input.input_type.tensor_shape()?;
    let dims: &[i64] = &*shape;

    if dims.len() >= 4 {
        if let Some(spec) = infer_nchw_spec(dims) {
            return Some(spec);
        }
        if let Some(spec) = infer_nhwc_spec(dims) {
            return Some(spec);
        }
    }

    None
}

/// Checks for an NCHW layout and returns a matching spec when dimensions line up.
fn infer_nchw_spec(dims: &[i64]) -> Option<ModelInputSpec> {
    let channels = *dims.get(1)?;
    if channels != 3 && channels != -1 {
        return None;
    }
    let height = *dims.get(2)?;
    let width = *dims.get(3)?;
    let height = positive_dim_to_usize(height)?;
    let width = positive_dim_to_usize(width)?;
    Some(ModelInputSpec {
        width,
        height,
        layout: ChannelLayout::Nchw,
    })
}

/// Checks for an NHWC layout and returns a matching spec when dimensions line up.
fn infer_nhwc_spec(dims: &[i64]) -> Option<ModelInputSpec> {
    let channels = *dims.get(3)?;
    if channels != 3 && channels != -1 {
        return None;
    }
    let height = *dims.get(1)?;
    let width = *dims.get(2)?;
    let height = positive_dim_to_usize(height)?;
    let width = positive_dim_to_usize(width)?;
    Some(ModelInputSpec {
        width,
        height,
        layout: ChannelLayout::Nhwc,
    })
}

/// Converts a positive i64 dimension to usize, returning None for non-positive or overflow.
fn positive_dim_to_usize(dim: i64) -> Option<usize> {
    if dim > 0 {
        usize::try_from(dim).ok()
    } else {
        None
    }
}

/// Resizes and normalizes the RGB image into a tensor that matches the model spec.
pub fn preprocess_image_to_tensor(
    rgb: &RgbImage,
    filter: FilterType,
    spec: ModelInputSpec,
) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
    let target_w = u32::try_from(spec.width).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("model width {} exceeds u32", spec.width),
        )
    })?;
    let target_h = u32::try_from(spec.height).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("model height {} exceeds u32", spec.height),
        )
    })?;

    let resized = image::imageops::resize(rgb, target_w, target_h, filter);
    let w = resized.width() as usize;
    let h = resized.height() as usize;
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let inv255 = 1.0 / 255.0;

    let (shape, data) = match spec.layout {
        ChannelLayout::Nchw => {
            let mut buffer = vec![0f32; 3 * h * w];
            for y in 0..h {
                for x in 0..w {
                    let pixel = resized.get_pixel(x as u32, y as u32);
                    let r = f32::from(pixel[0]) * inv255;
                    let g = f32::from(pixel[1]) * inv255;
                    let b = f32::from(pixel[2]) * inv255;
                    let idx = y * w + x;
                    buffer[idx] = (r - mean[0]) / std[0];
                    buffer[h * w + idx] = (g - mean[1]) / std[1];
                    buffer[2 * h * w + idx] = (b - mean[2]) / std[2];
                }
            }
            ((1usize, 3usize, h, w), buffer)
        }
        ChannelLayout::Nhwc => {
            let mut buffer = vec![0f32; h * w * 3];
            for y in 0..h {
                for x in 0..w {
                    let pixel = resized.get_pixel(x as u32, y as u32);
                    let r = f32::from(pixel[0]) * inv255;
                    let g = f32::from(pixel[1]) * inv255;
                    let b = f32::from(pixel[2]) * inv255;
                    let idx = (y * w + x) * 3;
                    buffer[idx] = (r - mean[0]) / std[0];
                    buffer[idx + 1] = (g - mean[1]) / std[1];
                    buffer[idx + 2] = (b - mean[2]) / std[2];
                }
            }
            ((1usize, h, w, 3usize), buffer)
        }
    };

    let array = Array4::from_shape_vec(shape, data)?;
    Ok(Tensor::from_array(array)?)
}

/// Remove singleton axes to get the raw H×W matte from the model output.
pub fn extract_matte_hw(matte: ArrayViewD<f32>) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let original_shape: Vec<usize> = matte.shape().to_vec();
    let mut view = matte;

    while view.ndim() > 2 {
        let axis = view
            .shape()
            .iter()
            .position(|&len| len == 1)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Cannot infer H×W from output shape {:?}", original_shape),
                )
            })?;
        view = view.index_axis_move(Axis(axis), 0);
    }
    Ok(view.into_dimensionality::<Ix2>()?.to_owned())
}

/// Resamples the matte to the requested width and height with the chosen filter.
pub fn resize_matte(
    matte: &Array2<f32>,
    target_w: u32,
    target_h: u32,
    filter: FilterType,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let src_w = matte.shape()[1] as u32;
    let src_h = matte.shape()[0] as u32;
    let mut buffer = ImageBuffer::<Luma<f32>, Vec<f32>>::new(src_w, src_h);
    for (y, row) in matte.axis_iter(Axis(0)).enumerate() {
        for (x, &value) in row.iter().enumerate() {
            buffer.put_pixel(x as u32, y as u32, Luma([value]));
        }
    }
    let resized = image::imageops::resize(&buffer, target_w, target_h, filter);
    let mut out = Array2::<f32>::zeros((target_h as usize, target_w as usize));
    for (x, y, pixel) in resized.enumerate_pixels() {
        out[[y as usize, x as usize]] = pixel[0];
    }
    Ok(out)
}
