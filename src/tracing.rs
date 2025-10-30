#[cfg(feature = "vectorizer-vtracer")]
use vtracer::{ColorImage, Config, SvgFile, convert};

use crate::{OutlineError, OutlineResult, TraceOptions};

/// Traces a ColorImage into an SVG using the specified TraceOptions.
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
