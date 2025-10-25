pub mod config;
pub mod foreground;
pub mod inference;
pub mod mask;
pub mod tracing;

pub use config::{
    AlphaSource, CutConfig, MaskCommandConfig, MaskOutputKind, MaskProcessingOptions,
    MaskSourcePreference, MattePipelineConfig, ModelOptions, TraceConfig,
};

use image::{GrayImage, ImageReader, RgbImage};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

use crate::inference::{
    determine_model_input_spec, extract_matte_hw, preprocess_image_to_tensor, resize_matte,
};
use crate::mask::{
    array_to_gray_image, blur_then_threshold, dilate_euclidean, fill_mask_holes,
    gray_to_color_image_rgba, threshold_mask,
};
use crate::tracing::trace;

#[derive(Debug)]
pub struct MattePipelineResult {
    pub rgb_image: RgbImage,
    pub raw_matte: GrayImage,
    pub processed_mask: GrayImage,
}

pub fn run_matte_pipeline(
    config: &MattePipelineConfig,
) -> Result<MattePipelineResult, Box<dyn std::error::Error>> {
    let mut builder =
        Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;
    if let Some(n) = config.model.intra_threads {
        builder = builder.with_intra_threads(n)?;
    }
    let mut session = builder.commit_from_file(&config.model.model_path)?;

    let rgb_input = ImageReader::open(&config.model.image_path)?
        .decode()?
        .to_rgb8();
    let orig_w = rgb_input.width();
    let orig_h = rgb_input.height();

    let input_spec = determine_model_input_spec(&session);
    let input_tensor =
        preprocess_image_to_tensor(&rgb_input, config.model.model_resize_filter, input_spec)?;
    let outputs = session.run(ort::inputs![input_tensor])?;
    let matte = outputs[0].try_extract_array::<f32>()?;
    let matte_hw = extract_matte_hw(matte)?;
    let matte_orig = resize_matte(&matte_hw, orig_w, orig_h, config.model.matte_resize_filter)?;
    let raw_matte = array_to_gray_image(&matte_orig);

    let mut processed_mask = if config.mask_processing.blur {
        blur_then_threshold(
            &raw_matte,
            config.mask_processing.blur_sigma,
            config.mask_processing.mask_threshold,
        )
    } else {
        threshold_mask(&raw_matte, config.mask_processing.mask_threshold)
    };

    if config.mask_processing.dilate {
        processed_mask = dilate_euclidean(&processed_mask, config.mask_processing.dilation_radius);
    }

    if config.mask_processing.fill_holes {
        processed_mask = fill_mask_holes(&processed_mask);
    }

    Ok(MattePipelineResult {
        rgb_image: rgb_input,
        raw_matte,
        processed_mask,
    })
}

pub fn select_mask<'a>(
    result: &'a MattePipelineResult,
    preference: MaskSourcePreference,
) -> &'a GrayImage {
    match preference {
        MaskSourcePreference::Raw => &result.raw_matte,
        MaskSourcePreference::Processed | MaskSourcePreference::Auto => &result.processed_mask,
    }
}

pub fn select_alpha<'a>(result: &'a MattePipelineResult, source: AlphaSource) -> &'a GrayImage {
    match source {
        AlphaSource::Raw => &result.raw_matte,
        AlphaSource::Processed => &result.processed_mask,
    }
}

pub fn trace_to_svg_string(
    mask_image: &GrayImage,
    trace_config: &TraceConfig,
) -> Result<String, Box<dyn std::error::Error>> {
    let color_img = gray_to_color_image_rgba(mask_image, None, trace_config.invert_svg);
    let svg = trace(color_img, trace_config)?;
    Ok(svg.to_string())
}
