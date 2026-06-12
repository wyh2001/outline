use std::fs;

use outline::{OutlineResult, VtracerSvgVectorizer};

use crate::cli::{GlobalOptions, MaskSourceArg, TraceCommand};

use super::utils::{
    build_outline, derive_svg_path, mask_pipeline_from_args, processing_requested,
    resolve_mask_source_arg,
};

/// The main function to run the trace command.
pub fn run(global: &GlobalOptions, cmd: TraceCommand) -> OutlineResult<()> {
    let outline = build_outline(global);
    let session = outline.for_image(&cmd.input)?;
    let matte = session.matte();
    let output_path = cmd
        .output
        .clone()
        .unwrap_or_else(|| derive_svg_path(&cmd.input));

    let options = (&cmd.trace_options).into();

    let vectorizer = VtracerSvgVectorizer;
    let processing_requested = processing_requested(&cmd.mask_processing);
    let mask_pipeline = mask_pipeline_from_args(&cmd.mask_processing);

    let mask_source = resolve_mask_source_arg(cmd.mask_source, processing_requested);

    let svg = match mask_source {
        MaskSourceArg::Raw => matte.trace(&vectorizer, &options)?,
        MaskSourceArg::Processed => matte
            .clone()
            .processed_with(&mask_pipeline)?
            .trace(&vectorizer, &options)?,
        MaskSourceArg::Auto => unreachable!(),
    };
    fs::write(&output_path, &svg)?;
    println!("SVG saved to {}", output_path.display());

    Ok(())
}
