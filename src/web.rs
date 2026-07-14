//! Browser-specific asynchronous ONNX Runtime Web backend.

use std::{fmt, io};

use image::imageops::FilterType;
use image::{DynamicImage, RgbImage, RgbaImage};
use js_sys::{Array, Float32Array, Object, Reflect, Uint8Array};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

use crate::config::{MaskProcessingDefaults, ModelInputSize};
use crate::inference::{
    DEFAULT_MODEL_INPUT_SPEC, ModelInputSpec, array_from_tensor_parts, extract_matte_hw,
    finish_model_matte, infer_model_input_spec_from_dims, load_rgb_from_memory_with_orientation,
    prepare_model_input,
};
use crate::{InferencedMatte, OutlineError, OutlineResult};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ort)]
    type InferenceSession;

    #[wasm_bindgen(catch, js_namespace = ort, static_method_of = InferenceSession, js_name = create)]
    async fn create_from_url(model_url: &str) -> Result<InferenceSession, JsValue>;

    #[wasm_bindgen(catch, js_namespace = ort, static_method_of = InferenceSession, js_name = create)]
    async fn create_from_bytes(model_bytes: &Uint8Array) -> Result<InferenceSession, JsValue>;

    #[wasm_bindgen(catch, structural, method, js_name = run)]
    async fn run(this: &InferenceSession, feeds: JsValue) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(js_namespace = ort)]
    type Tensor;

    #[wasm_bindgen(constructor, catch, js_namespace = ort, js_class = Tensor)]
    fn new(data_type: &str, data: JsValue, dimensions: JsValue) -> Result<Tensor, JsValue>;
}

/// Asynchronous browser entry point for background matting with ONNX Runtime Web.
///
/// `WebOutline` loads and caches one model session. Construct it from a model URL or model bytes,
/// then reuse the same value for multiple images. Image preprocessing, matte resizing, and the
/// returned [`InferencedMatte`] mask API are shared with the native `Outline` pipeline.
///
/// This type is only available for `wasm32` with the `backend-ort-web` feature.
pub struct WebOutline {
    session: InferenceSession,
    input_name: String,
    output_name: String,
    model_input_spec: ModelInputSpec,
    model_input_size: Option<ModelInputSize>,
    input_resize_filter: FilterType,
    output_resize_filter: FilterType,
    mask_processing_defaults: MaskProcessingDefaults,
}

impl fmt::Debug for WebOutline {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WebOutline")
            .field("input_name", &self.input_name)
            .field("output_name", &self.output_name)
            .field("model_input_spec", &self.model_input_spec)
            .field("model_input_size", &self.model_input_size)
            .field("input_resize_filter", &self.input_resize_filter)
            .field("output_resize_filter", &self.output_resize_filter)
            .field("mask_processing_defaults", &self.mask_processing_defaults)
            .finish_non_exhaustive()
    }
}

impl WebOutline {
    /// Load a model from a URL after initializing the browser runtime.
    ///
    /// The URL may be relative to the page, which makes a versioned same-origin static asset a
    /// suitable model source. The browser runtime must be able to fetch the URL.
    pub async fn from_model_url(model_url: impl AsRef<str>) -> OutlineResult<Self> {
        let model_url = model_url.as_ref();
        if model_url.trim().is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "ONNX model URL must not be empty",
            )
            .into());
        }

        initialize_runtime().await?;
        let session = InferenceSession::create_from_url(model_url)
            .await
            .map_err(|error| js_error("failed to create ONNX Runtime Web session", error))?;
        Self::from_session(session)
    }

    /// Load a model from in-memory ONNX bytes after initializing the browser runtime.
    ///
    /// This is useful when the application fetches and verifies a model itself before creating the
    /// inference session.
    pub async fn from_model_bytes(model_bytes: &[u8]) -> OutlineResult<Self> {
        if model_bytes.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "ONNX model bytes must not be empty",
            )
            .into());
        }

        initialize_runtime().await?;
        let bytes = Uint8Array::from(model_bytes);
        let session = InferenceSession::create_from_bytes(&bytes)
            .await
            .map_err(|error| js_error("failed to create ONNX Runtime Web session", error))?;
        Self::from_session(session)
    }

    fn from_session(session: InferenceSession) -> OutlineResult<Self> {
        let input_name = first_declared_name(&session, "inputNames", "input")?;
        let output_name = first_declared_name(&session, "outputNames", "output")?;
        let model_input_spec = determine_web_model_input_spec(&session, &input_name)?;

        Ok(Self {
            session,
            input_name,
            output_name,
            model_input_spec,
            model_input_size: None,
            input_resize_filter: FilterType::Triangle,
            output_resize_filter: FilterType::Lanczos3,
            mask_processing_defaults: MaskProcessingDefaults::default(),
        })
    }

    /// Override the height and width used for model input.
    ///
    /// This bypasses the shape inferred from the ONNX graph. The caller is responsible for using a
    /// size supported by the model.
    ///
    /// # Panics
    ///
    /// Panics if either dimension is zero.
    pub fn with_model_input_size(mut self, height: usize, width: usize) -> Self {
        self.model_input_size = Some(ModelInputSize::new(height, width));
        self
    }

    /// Set the filter used to resize source images to the model input dimensions.
    pub fn with_input_resize_filter(mut self, filter: FilterType) -> Self {
        self.input_resize_filter = filter;
        self
    }

    /// Set the filter used to resize model mattes back to source image dimensions.
    pub fn with_output_resize_filter(mut self, filter: FilterType) -> Self {
        self.output_resize_filter = filter;
        self
    }

    /// Set the default values used by no-argument matte and mask processing methods.
    pub fn with_mask_processing_defaults(mut self, defaults: MaskProcessingDefaults) -> Self {
        self.mask_processing_defaults = defaults;
        self
    }

    /// Get the default mask-processing parameters returned with inference results.
    pub fn mask_processing_defaults(&self) -> &MaskProcessingDefaults {
        &self.mask_processing_defaults
    }

    /// Run inference for an in-memory RGB image.
    pub async fn for_rgb_image(&self, rgb_image: RgbImage) -> OutlineResult<InferencedMatte> {
        let original_width = rgb_image.width();
        let original_height = rgb_image.height();
        let input_array = prepare_model_input(
            &rgb_image,
            self.model_input_spec,
            self.model_input_size,
            self.input_resize_filter,
        )?;

        let input_dimensions = dimensions_to_js(input_array.shape())?;
        let (input_values, offset) = input_array.into_raw_vec_and_offset();
        if offset != Some(0) {
            return Err(io::Error::other("preprocessed input array is not contiguous").into());
        }
        let input_values = Float32Array::from(input_values.as_slice());
        let input_tensor = Tensor::new("float32", input_values.into(), input_dimensions.into())
            .map_err(|error| js_error("failed to create ONNX Runtime Web input tensor", error))?;

        let feeds = Object::new();
        let assigned = Reflect::set(
            feeds.as_ref(),
            &JsValue::from_str(&self.input_name),
            input_tensor.as_ref(),
        )
        .map_err(|error| js_error("failed to prepare ONNX Runtime Web inputs", error))?;
        if !assigned {
            return Err(io::Error::other("failed to assign ONNX Runtime Web input").into());
        }

        let outputs = self
            .session
            .run(feeds.into())
            .await
            .map_err(|error| js_error("ONNX Runtime Web inference failed", error))?;
        let output = Reflect::get(&outputs, &JsValue::from_str(&self.output_name))
            .map_err(|error| js_error("failed to read ONNX Runtime Web output", error))?;
        if output.is_null() || output.is_undefined() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "ONNX Runtime Web did not return declared output {:?}",
                    self.output_name
                ),
            )
            .into());
        }

        let output_values = Reflect::get(&output, &JsValue::from_str("data"))
            .map_err(|error| js_error("failed to read ONNX Runtime Web output data", error))?
            .dyn_into::<Float32Array>()
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "ONNX Runtime Web output data is not Float32Array",
                )
            })?
            .to_vec();
        let output_dimensions =
            Reflect::get(&output, &JsValue::from_str("dims")).map_err(|error| {
                js_error("failed to read ONNX Runtime Web output dimensions", error)
            })?;
        let output_dimensions = parse_output_dimensions(&output_dimensions)?;
        let output_array = array_from_tensor_parts(&output_dimensions, output_values)?;
        let matte_hw = extract_matte_hw(output_array.view())?;
        let raw_matte = finish_model_matte(
            &matte_hw,
            original_width,
            original_height,
            self.output_resize_filter,
        )?;

        Ok(InferencedMatte::new(
            rgb_image,
            raw_matte,
            self.mask_processing_defaults.clone(),
        ))
    }

    /// Run inference for an in-memory RGBA image.
    ///
    /// The alpha channel is discarded before inference, matching native `Outline`.
    pub async fn for_rgba_image(&self, rgba_image: RgbaImage) -> OutlineResult<InferencedMatte> {
        self.for_dynamic_image(DynamicImage::ImageRgba8(rgba_image))
            .await
    }

    /// Run inference for an in-memory image decoded by the `image` crate.
    pub async fn for_dynamic_image(&self, image: DynamicImage) -> OutlineResult<InferencedMatte> {
        self.for_rgb_image(image.into_rgb8()).await
    }

    /// Decode encoded image bytes and run browser inference.
    ///
    /// Image decoding applies EXIF orientation metadata when present, matching native
    /// `Outline::for_image_bytes`.
    pub async fn for_image_bytes(&self, bytes: &[u8]) -> OutlineResult<InferencedMatte> {
        let rgb = load_rgb_from_memory_with_orientation(bytes)?;
        self.for_rgb_image(rgb).await
    }
}

async fn initialize_runtime() -> OutlineResult<()> {
    // This call only loads the global JavaScript runtime. The returned C-style API is deliberately
    // unused: creating an `ort` Environment would enable ort-web's telemetry bridge and would also
    // enter its currently unusable Session metadata adapter.
    let _unused_api = ort_web::api(ort_web::FEATURE_NONE).await?;
    Ok(())
}

fn first_declared_name(
    session: &InferenceSession,
    property: &str,
    kind: &str,
) -> OutlineResult<String> {
    let names = Reflect::get(session.as_ref(), &JsValue::from_str(property))
        .map_err(|error| js_error("failed to read ONNX Runtime Web model metadata", error))?;
    if !Array::is_array(&names) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("ONNX Runtime Web {property} metadata is not an array"),
        )
        .into());
    }

    let first = Reflect::get(&names, &JsValue::from_f64(0.0))
        .map_err(|error| js_error("failed to read ONNX Runtime Web model metadata", error))?;
    first.as_string().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("ONNX model does not declare an {kind} tensor"),
        )
        .into()
    })
}

fn determine_web_model_input_spec(
    session: &InferenceSession,
    input_name: &str,
) -> OutlineResult<ModelInputSpec> {
    let metadata = Reflect::get(session.as_ref(), &JsValue::from_str("inputMetadata"))
        .map_err(|error| js_error("failed to read ONNX Runtime Web input metadata", error))?;
    let entry = if Array::is_array(&metadata) {
        Reflect::get(&metadata, &JsValue::from_f64(0.0))
    } else {
        Reflect::get(&metadata, &JsValue::from_str(input_name))
    }
    .map_err(|error| js_error("failed to read ONNX Runtime Web input metadata", error))?;

    if entry.is_null() || entry.is_undefined() {
        return Ok(DEFAULT_MODEL_INPUT_SPEC);
    }
    let shape = Reflect::get(&entry, &JsValue::from_str("shape"))
        .map_err(|error| js_error("failed to read ONNX Runtime Web input shape", error))?;
    let Some(dimensions) = parse_metadata_dimensions(&shape) else {
        return Ok(DEFAULT_MODEL_INPUT_SPEC);
    };
    Ok(infer_model_input_spec_from_dims(&dimensions).unwrap_or(DEFAULT_MODEL_INPUT_SPEC))
}

fn parse_metadata_dimensions(value: &JsValue) -> Option<Vec<i64>> {
    if !Array::is_array(value) {
        return None;
    }

    let dimensions = Array::from(value);
    let mut parsed = Vec::with_capacity(dimensions.length() as usize);
    for dimension in dimensions.iter() {
        if let Some(number) = dimension.as_f64() {
            if !number.is_finite()
                || number.fract() != 0.0
                || number < i64::MIN as f64
                || number > i64::MAX as f64
            {
                return None;
            }
            parsed.push(number as i64);
        } else if dimension.as_string().is_some() || dimension.is_null() || dimension.is_undefined()
        {
            parsed.push(-1);
        } else {
            return None;
        }
    }
    Some(parsed)
}

fn parse_output_dimensions(value: &JsValue) -> OutlineResult<Vec<usize>> {
    if !Array::is_array(value) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "ONNX Runtime Web output dimensions are not an array",
        )
        .into());
    }

    let dimensions = Array::from(value);
    let mut parsed = Vec::with_capacity(dimensions.length() as usize);
    for dimension in dimensions.iter() {
        let number = dimension.as_f64().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "ONNX Runtime Web output dimension is not numeric",
            )
        })?;
        if !number.is_finite() || number.fract() != 0.0 || number <= 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid ONNX Runtime Web output dimension {number}"),
            )
            .into());
        }
        parsed.push(usize::try_from(number as u64).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("ONNX Runtime Web output dimension {number} exceeds usize"),
            )
        })?);
    }
    Ok(parsed)
}

fn dimensions_to_js(dimensions: &[usize]) -> OutlineResult<Array> {
    let output = Array::new();
    for &dimension in dimensions {
        let dimension = u32::try_from(dimension).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("input tensor dimension {dimension} exceeds u32"),
            )
        })?;
        output.push(&JsValue::from_f64(f64::from(dimension)));
    }
    Ok(output)
}

fn js_error(context: &str, error: JsValue) -> OutlineError {
    let detail = Reflect::get(&error, &JsValue::from_str("message"))
        .ok()
        .and_then(|value| value.as_string())
        .filter(|value| !value.trim().is_empty())
        .or_else(|| error.as_string())
        .unwrap_or_else(|| format!("{error:?}"));
    io::Error::other(format!("{context}: {detail}")).into()
}
