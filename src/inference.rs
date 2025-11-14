use std::convert::TryFrom;
use std::io;
use std::path::Path;

use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, ImageBuffer, ImageDecoder, ImageReader, Luma, RgbImage};
use ndarray::{Array2, Array4, ArrayViewD, Axis, Ix2};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use crate::config::InferenceSettings;
use crate::error::OutlineResult;
use crate::mask::array_to_gray_image;

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

/// Try to figure out the model input spec from the session and falls back to the default.
pub fn determine_model_input_spec(session: &Session) -> ModelInputSpec {
    infer_model_input_spec(session).unwrap_or(DEFAULT_MODEL_INPUT_SPEC)
}

/// Infer the model input spec from the ONNX session input tensor shape.
fn infer_model_input_spec(session: &Session) -> Option<ModelInputSpec> {
    let input = session.inputs.first()?;
    let shape = input.input_type.tensor_shape()?;
    let dims: &[i64] = shape;

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

/// Check for an NCHW layout and returns a matching spec when dimensions line up.
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

/// Check for an NHWC layout and returns a matching spec when dimensions line up.
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

/// Convert a positive i64 dimension to usize, returning None for non-positive or overflow.
fn positive_dim_to_usize(dim: i64) -> Option<usize> {
    if dim > 0 {
        usize::try_from(dim).ok()
    } else {
        None
    }
}

/// Load an RGB image from the given path, applying orientation from EXIF data.
fn load_rgb_with_orientation(path: &Path) -> OutlineResult<RgbImage> {
    let mut decoder = ImageReader::open(path)?.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(image.into_rgb8())
}

/// Resize and normalizes the RGB image into a tensor that matches the model spec.
pub fn preprocess_image_to_tensor(
    rgb: &RgbImage,
    filter: FilterType,
    spec: ModelInputSpec,
) -> OutlineResult<Tensor<f32>> {
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
            let (r_plane, rest) = buffer.split_at_mut(h * w);
            let (g_plane, b_plane) = rest.split_at_mut(h * w);

            for (idx, pixel) in resized.pixels().enumerate() {
                let r = f32::from(pixel[0]) * inv255;
                let g = f32::from(pixel[1]) * inv255;
                let b = f32::from(pixel[2]) * inv255;
                r_plane[idx] = (r - mean[0]) / std[0];
                g_plane[idx] = (g - mean[1]) / std[1];
                b_plane[idx] = (b - mean[2]) / std[2];
            }
            ((1usize, 3usize, h, w), buffer)
        }
        ChannelLayout::Nhwc => {
            let mut buffer = Vec::with_capacity(h * w * 3);
            for pixel in resized.pixels() {
                let r = f32::from(pixel[0]) * inv255;
                let g = f32::from(pixel[1]) * inv255;
                let b = f32::from(pixel[2]) * inv255;
                buffer.push((r - mean[0]) / std[0]);
                buffer.push((g - mean[1]) / std[1]);
                buffer.push((b - mean[2]) / std[2]);
            }
            ((1usize, h, w, 3usize), buffer)
        }
    };

    let array = Array4::from_shape_vec(shape, data)?;
    Ok(Tensor::from_array(array)?)
}

/// Remove singleton axes to get the raw H×W matte from the model output.
pub fn extract_matte_hw(matte: ArrayViewD<f32>) -> OutlineResult<Array2<f32>> {
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

/// Resample the matte to the requested width and height with the chosen filter.
pub fn resize_matte(
    matte: &Array2<f32>,
    target_w: u32,
    target_h: u32,
    filter: FilterType,
) -> OutlineResult<Array2<f32>> {
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

/// Run the full matte inference pipeline and return the RGB image and raw matte.
pub fn run_matte_pipeline(
    settings: &InferenceSettings,
    image_path: &Path,
) -> OutlineResult<(RgbImage, GrayImage)> {
    let mut builder =
        Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;
    if let Some(n) = settings.intra_threads {
        builder = builder.with_intra_threads(n)?;
    }
    let mut session = builder.commit_from_file(&settings.model_path)?;

    let rgb_input = load_rgb_with_orientation(image_path)?;
    let orig_w = rgb_input.width();
    let orig_h = rgb_input.height();

    let input_spec = determine_model_input_spec(&session);
    let input_tensor =
        preprocess_image_to_tensor(&rgb_input, settings.input_resize_filter, input_spec)?;
    let outputs = session.run(ort::inputs![input_tensor])?;
    let matte = outputs[0].try_extract_array::<f32>()?;
    let matte_hw = extract_matte_hw(matte)?;
    let matte_orig = resize_matte(&matte_hw, orig_w, orig_h, settings.output_resize_filter)?;
    let raw_matte = array_to_gray_image(&matte_orig);

    Ok((rgb_input, raw_matte))
}
