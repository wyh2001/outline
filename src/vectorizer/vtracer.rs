use image::GrayImage;
use visioncortex::PathSimplifyMode;
use vtracer::{ColorImage, ColorMode, Config, Hierarchical, SvgFile, convert};

use crate::mask::gray_to_color_image_rgba;
use crate::{OutlineError, OutlineResult};

use super::MaskVectorizer;

/// Options for tracing the mask to SVG using VTracer.
#[derive(Debug, Clone)]
pub struct TraceOptions {
    pub tracer_color_mode: ColorMode,
    pub tracer_hierarchical: Hierarchical,
    pub tracer_mode: PathSimplifyMode,
    pub tracer_filter_speckle: usize,
    pub tracer_color_precision: i32,
    pub tracer_layer_difference: i32,
    pub tracer_corner_threshold: i32,
    pub tracer_length_threshold: f64,
    pub tracer_max_iterations: usize,
    pub tracer_splice_threshold: i32,
    pub tracer_path_precision: Option<u32>,
    pub invert_svg: bool,
}

impl Default for TraceOptions {
    fn default() -> Self {
        Self {
            tracer_color_mode: ColorMode::Binary,
            tracer_hierarchical: Hierarchical::Stacked,
            tracer_mode: PathSimplifyMode::Spline,
            tracer_filter_speckle: 4,
            tracer_color_precision: 6,
            tracer_layer_difference: 16,
            tracer_corner_threshold: 60,
            tracer_length_threshold: 4.0,
            tracer_max_iterations: 10,
            tracer_splice_threshold: 45,
            tracer_path_precision: Some(2),
            invert_svg: false,
        }
    }
}

/// VTracer-based SVG vectorizer implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct VtracerSvgVectorizer;

impl MaskVectorizer for VtracerSvgVectorizer {
    type Options = TraceOptions;
    type Output = String;

    fn vectorize(&self, mask: &GrayImage, options: &Self::Options) -> OutlineResult<Self::Output> {
        trace_to_svg_string(mask, options)
    }
}

/// The helper function that uses VTracer to trace a grayscale mask to an SVG string.
pub fn trace_to_svg_string(
    mask_image: &GrayImage,
    options: &TraceOptions,
) -> OutlineResult<String> {
    let color_img = gray_to_color_image_rgba(mask_image, None, options.invert_svg);
    let svg_file = trace(color_img, options)?;
    Ok(svg_file.to_string())
}

/// Trace a ColorImage into an SVG using VTracer with the given options.
pub fn trace(img: ColorImage, options: &TraceOptions) -> OutlineResult<SvgFile> {
    let cfg = Config {
        color_mode: options.tracer_color_mode.clone(),
        hierarchical: options.tracer_hierarchical.clone(),
        mode: options.tracer_mode,
        filter_speckle: options.tracer_filter_speckle,
        color_precision: options.tracer_color_precision,
        layer_difference: options.tracer_layer_difference,
        corner_threshold: options.tracer_corner_threshold,
        length_threshold: options.tracer_length_threshold,
        max_iterations: options.tracer_max_iterations,
        splice_threshold: options.tracer_splice_threshold,
        path_precision: options.tracer_path_precision,
    };

    let svg_file = convert(img, cfg).map_err(OutlineError::Trace)?;
    Ok(svg_file)
}
