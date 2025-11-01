use std::fs;

use outline::{MaskProcessingOptions, OutlineResult, TraceOptions, VtracerSvgVectorizer};

use crate::cli::{GlobalOptions, MaskSourceArg, TraceCommand};

use super::utils::{build_outline, derive_svg_path};

/// The main function to run the trace command.
pub fn run(global: &GlobalOptions, cmd: TraceCommand) -> OutlineResult<()> {
    let outline = build_outline(
        global,
        &cmd.mask_processing,
        MaskProcessingOptions::default(),
    );
    let session = outline.for_image(&cmd.input)?;
    let matte = session.matte();
    let output_path = cmd
        .output
        .clone()
        .unwrap_or_else(|| derive_svg_path(&cmd.input));

    let mut options = TraceOptions::default();
    options.tracer_color_mode = cmd.trace_options.color_mode.into();
    options.tracer_hierarchical = cmd.trace_options.hierarchy.into();
    options.tracer_mode = cmd.trace_options.mode.into();
    options.tracer_filter_speckle = cmd.trace_options.filter_speckle;
    options.tracer_color_precision = cmd.trace_options.color_precision;
    options.tracer_layer_difference = cmd.trace_options.layer_difference;
    options.tracer_corner_threshold = cmd.trace_options.corner_threshold;
    options.tracer_length_threshold = cmd.trace_options.length_threshold;
    options.tracer_max_iterations = cmd.trace_options.max_iterations;
    options.tracer_splice_threshold = cmd.trace_options.splice_threshold;
    if let Some(path_precision) = cmd.trace_options.path_precision {
        options.tracer_path_precision = Some(path_precision);
    }
    if cmd.trace_options.no_path_precision {
        options.tracer_path_precision = None;
    }
    options.invert_svg = cmd.trace_options.invert_svg;

    let vectorizer = VtracerSvgVectorizer;
    let defaults = MaskProcessingOptions::default();
    let processing_requested = cmd.mask_processing.binary
        || cmd.mask_processing.blur.is_some()
        || cmd.mask_processing.dilate.is_some()
        || cmd.mask_processing.fill_holes
        || cmd.mask_processing.mask_threshold != defaults.mask_threshold;

    let mask_source = match cmd.mask_source {
        MaskSourceArg::Auto => {
            if processing_requested {
                MaskSourceArg::Processed
            } else {
                MaskSourceArg::Raw
            }
        }
        other => other,
    };

    if matches!(mask_source, MaskSourceArg::Processed)
        && !cmd.mask_processing.binary
        && (cmd.mask_processing.dilate.is_some() || cmd.mask_processing.fill_holes)
    {
        eprintln!(
            "Warning: --no-binary disables thresholding, but dilation/fill-holes assume a hard mask; tracing output may be unexpected."
        );
    }

    let svg = match mask_source {
        MaskSourceArg::Raw => matte.trace(&vectorizer, &options)?,
        MaskSourceArg::Processed => matte
            .clone()
            .processed(None)?
            .trace(&vectorizer, &options)?,
        MaskSourceArg::Auto => unreachable!(),
    };
    fs::write(&output_path, &svg)?;
    println!("SVG saved to {}", output_path.display());

    Ok(())
}
