use std::convert::TryFrom;
use std::io;
use std::io::Cursor;
use std::path::Path;
#[cfg(feature = "backend-ort")]
use std::sync::Mutex;

use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, ImageBuffer, ImageDecoder, ImageReader, Luma, RgbImage};
use ndarray::{Array2, Array4, ArrayViewD, Axis, Ix2};
#[cfg(feature = "backend-rten")]
use ndarray::{ArrayD, IxDyn};
#[cfg(feature = "backend-ort")]
use ort::session::Session;
#[cfg(feature = "backend-ort")]
use ort::session::builder::GraphOptimizationLevel;
#[cfg(feature = "backend-ort")]
use ort::value::Tensor;

#[cfg(any(feature = "backend-ort", feature = "backend-rten"))]
use crate::config::InferenceBackend;
use crate::config::InferenceSettings;
use crate::error::{OutlineError, OutlineResult};
use crate::mask::array_to_gray_image;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelLayout {
    Nchw,
    Nhwc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelInputSpec {
    pub height: usize,
    pub width: usize,
    pub layout: ChannelLayout,
}

pub const DEFAULT_MODEL_INPUT_SPEC: ModelInputSpec = ModelInputSpec {
    height: 320,
    width: 320,
    layout: ChannelLayout::Nchw,
};

/// Cached inference entry point for the full matte pipeline.
#[derive(Debug)]
pub struct CachedInferenceSession {
    backend: BackendSession,
}

#[derive(Debug)]
enum BackendSession {
    #[cfg(feature = "backend-ort")]
    Ort(OrtInferenceSession),
    #[cfg(feature = "backend-rten")]
    Rten(Box<RtenInferenceSession>),
}

impl BackendSession {
    fn new(settings: &InferenceSettings) -> OutlineResult<Self> {
        if !settings.model_path().is_file() {
            return Err(OutlineError::ModelNotFound {
                path: settings.model_path().to_path_buf(),
            });
        }

        match settings.backend() {
            #[cfg(feature = "backend-ort")]
            InferenceBackend::Ort => Ok(Self::Ort(OrtInferenceSession::new(settings)?)),
            #[cfg(feature = "backend-rten")]
            InferenceBackend::Rten => {
                Ok(Self::Rten(Box::new(RtenInferenceSession::new(settings)?)))
            }
        }
    }

    fn input_spec(&self) -> ModelInputSpec {
        match self {
            #[cfg(feature = "backend-ort")]
            Self::Ort(session) => session.input_spec(),
            #[cfg(feature = "backend-rten")]
            Self::Rten(session) => session.input_spec(),
            #[cfg(not(any(feature = "backend-ort", feature = "backend-rten")))]
            _ => unreachable!("at least one inference backend feature must be enabled"),
        }
    }

    fn run_model(&self, input_array: Array4<f32>) -> OutlineResult<Array2<f32>> {
        #[cfg(not(any(feature = "backend-ort", feature = "backend-rten")))]
        let _ = &input_array;

        match self {
            #[cfg(feature = "backend-ort")]
            Self::Ort(session) => session.run_model(input_array),
            #[cfg(feature = "backend-rten")]
            Self::Rten(session) => session.run_model(input_array),
            #[cfg(not(any(feature = "backend-ort", feature = "backend-rten")))]
            _ => unreachable!("at least one inference backend feature must be enabled"),
        }
    }
}

impl CachedInferenceSession {
    /// Create a cached inference session.
    pub fn new(settings: &InferenceSettings) -> OutlineResult<Self> {
        Ok(Self {
            backend: BackendSession::new(settings)?,
        })
    }

    /// Run the full matte inference pipeline using this cached session.
    pub fn run_matte_pipeline(
        &self,
        settings: &InferenceSettings,
        image_path: &Path,
    ) -> OutlineResult<(RgbImage, GrayImage)> {
        let rgb_input = load_rgb_with_orientation(image_path)?;
        self.run_matte_pipeline_on_rgb(settings, rgb_input)
    }

    /// Run the full matte inference pipeline using an in-memory RGB image.
    pub fn run_matte_pipeline_on_rgb(
        &self,
        settings: &InferenceSettings,
        rgb_input: RgbImage,
    ) -> OutlineResult<(RgbImage, GrayImage)> {
        let orig_w = rgb_input.width();
        let orig_h = rgb_input.height();
        let mut input_spec = self.backend.input_spec();
        if let Some(size) = settings.model_input_size() {
            input_spec.width = size.width();
            input_spec.height = size.height();
        }

        let input_array =
            preprocess_image_to_array(&rgb_input, settings.input_resize_filter(), input_spec)?;
        let matte_hw = self.backend.run_model(input_array)?;
        let matte_orig = resize_matte(&matte_hw, orig_w, orig_h, settings.output_resize_filter())?;
        let raw_matte = array_to_gray_image(&matte_orig);

        Ok((rgb_input, raw_matte))
    }
}

/// ONNX Runtime-backed model session.
#[cfg(feature = "backend-ort")]
#[derive(Debug)]
struct OrtInferenceSession {
    session: Mutex<Session>,
    input_spec: ModelInputSpec,
}

#[cfg(feature = "backend-ort")]
impl OrtInferenceSession {
    /// Create an ONNX Runtime-backed session.
    fn new(settings: &InferenceSettings) -> OutlineResult<Self> {
        let mut builder =
            Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;
        if let Some(n) = settings.intra_threads() {
            builder = builder.with_intra_threads(n)?;
        }
        let session = builder.commit_from_file(settings.model_path())?;
        let input_spec = determine_model_input_spec(&session);

        Ok(Self {
            session: Mutex::new(session),
            input_spec,
        })
    }

    fn input_spec(&self) -> ModelInputSpec {
        self.input_spec
    }

    /// Execute the model for one preprocessed input array while holding the session lock.
    fn run_model(&self, input_array: Array4<f32>) -> OutlineResult<Array2<f32>> {
        let mut session = self
            .session
            .lock()
            .map_err(|_| io::Error::other("cached inference session mutex poisoned"))?;
        let input_tensor = Tensor::from_array(input_array)?;
        let outputs = session.run(ort::inputs![input_tensor])?;
        let matte = outputs[0].try_extract_array::<f32>()?;
        extract_matte_hw(matte)
    }
}

/// RTen-backed model session.
#[cfg(feature = "backend-rten")]
#[derive(Debug)]
struct RtenInferenceSession {
    model: rten::Model,
    input_spec: ModelInputSpec,
}

#[cfg(feature = "backend-rten")]
impl RtenInferenceSession {
    fn new(settings: &InferenceSettings) -> OutlineResult<Self> {
        let model = rten::Model::load_file(settings.model_path())?;
        let input_spec = determine_rten_model_input_spec(&model);

        Ok(Self { model, input_spec })
    }

    fn input_spec(&self) -> ModelInputSpec {
        self.input_spec
    }

    /// Execute the model for one preprocessed input array.
    fn run_model(&self, input_array: Array4<f32>) -> OutlineResult<Array2<f32>> {
        let shape = input_array.shape().to_vec();
        let (data, offset) = input_array.into_raw_vec_and_offset();
        if offset != Some(0) {
            return Err(io::Error::other("preprocessed input array is not contiguous").into());
        }

        let input = rten::Value::from_shape(shape, data).map_err(io::Error::other)?;
        let output = self.model.run_one(input.into(), None)?;
        let matte = rten_value_to_array(output)?;
        extract_matte_hw(matte.view())
    }
}

/// Try to figure out the model input spec from the session and falls back to the default.
#[cfg(feature = "backend-ort")]
pub fn determine_model_input_spec(session: &Session) -> ModelInputSpec {
    infer_model_input_spec(session).unwrap_or(DEFAULT_MODEL_INPUT_SPEC)
}

/// Infer the model input spec from the ONNX session input tensor shape.
#[cfg(feature = "backend-ort")]
fn infer_model_input_spec(session: &Session) -> Option<ModelInputSpec> {
    let input = session.inputs().first()?;
    let shape = input.dtype().tensor_shape()?;
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

/// Try to figure out the model input spec from the RTen model and falls back to the default.
#[cfg(feature = "backend-rten")]
pub fn determine_rten_model_input_spec(model: &rten::Model) -> ModelInputSpec {
    infer_rten_model_input_spec(model).unwrap_or(DEFAULT_MODEL_INPUT_SPEC)
}

/// Infer the model input spec from the RTen model input tensor shape.
#[cfg(feature = "backend-rten")]
fn infer_rten_model_input_spec(model: &rten::Model) -> Option<ModelInputSpec> {
    let dims: Vec<i64> = model
        .input_shape(0)?
        .into_iter()
        .map(|dim| match dim {
            rten::Dimension::Fixed(value) => i64::try_from(value).unwrap_or(-1),
            rten::Dimension::Symbolic(_) => -1,
        })
        .collect();

    if dims.len() >= 4 {
        if let Some(spec) = infer_nchw_spec(&dims) {
            return Some(spec);
        }
        if let Some(spec) = infer_nhwc_spec(&dims) {
            return Some(spec);
        }
    }

    None
}

#[cfg(feature = "backend-rten")]
fn rten_value_to_array(value: rten::Value) -> OutlineResult<ArrayD<f32>> {
    if let Ok((shape, data)) = value.clone().into_shape_vec::<f32, 2>() {
        return Ok(ArrayD::from_shape_vec(IxDyn(&shape), data)?);
    }
    if let Ok((shape, data)) = value.clone().into_shape_vec::<f32, 3>() {
        return Ok(ArrayD::from_shape_vec(IxDyn(&shape), data)?);
    }
    let (shape, data) = value.into_shape_vec::<f32, 4>()?;
    Ok(ArrayD::from_shape_vec(IxDyn(&shape), data)?)
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
        height,
        width,
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
        height,
        width,
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

/// Decode an RGB image from encoded bytes, applying orientation from EXIF data when present.
pub(crate) fn load_rgb_from_memory_with_orientation(bytes: &[u8]) -> OutlineResult<RgbImage> {
    let cursor = Cursor::new(bytes);
    let mut decoder = ImageReader::new(cursor)
        .with_guessed_format()?
        .into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(image.into_rgb8())
}

/// Resize and normalizes the RGB image into an array that matches the model spec.
pub fn preprocess_image_to_array(
    rgb: &RgbImage,
    filter: FilterType,
    spec: ModelInputSpec,
) -> OutlineResult<Array4<f32>> {
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
    if target_w == 0 || target_h == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "model input size must be non-zero",
        )
        .into());
    }

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

    Ok(Array4::from_shape_vec(shape, data)?)
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageFormat, Rgb, RgbImage, Rgba, RgbaImage};

    #[test]
    fn load_rgb_from_memory_decodes_png() {
        let rgb = RgbImage::from_pixel(3, 2, Rgb([12, 34, 56]));
        let image = DynamicImage::ImageRgb8(rgb);
        let mut encoded = Cursor::new(Vec::new());
        image
            .write_to(&mut encoded, ImageFormat::Png)
            .expect("png encoding should succeed");

        let decoded = load_rgb_from_memory_with_orientation(encoded.get_ref())
            .expect("memory decode should succeed");
        assert_eq!(decoded.dimensions(), (3, 2));
        assert_eq!(decoded.get_pixel(0, 0).0, [12, 34, 56]);
    }

    #[test]
    fn load_rgb_from_memory_discards_alpha() {
        let rgba = RgbaImage::from_pixel(4, 1, Rgba([10, 20, 30, 40]));
        let image = DynamicImage::ImageRgba8(rgba);
        let mut encoded = Cursor::new(Vec::new());
        image
            .write_to(&mut encoded, ImageFormat::Png)
            .expect("png encoding should succeed");

        let decoded = load_rgb_from_memory_with_orientation(encoded.get_ref())
            .expect("memory decode should succeed");
        assert_eq!(decoded.dimensions(), (4, 1));
        assert_eq!(decoded.get_pixel(0, 0).0, [10, 20, 30]);
    }
}
