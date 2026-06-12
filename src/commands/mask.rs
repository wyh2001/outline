use outline::OutlineResult;

use crate::cli::{GlobalOptions, MaskCommand, MaskExportSource};

use super::utils::{
    build_outline, derive_variant_path, mask_pipeline_from_args, processing_requested,
    resolve_mask_export_source,
};

/// The main function to run the mask command.
pub fn run(global: &GlobalOptions, cmd: MaskCommand) -> OutlineResult<()> {
    let outline = build_outline(global);
    let session = outline.for_image(&cmd.input)?;
    let matte = session.matte();
    let mask_pipeline = mask_pipeline_from_args(&cmd.mask_processing);
    let mask_source =
        resolve_mask_export_source(cmd.mask_source, processing_requested(&cmd.mask_processing));

    let default_suffix = match mask_source {
        MaskExportSource::Processed => "mask",
        MaskExportSource::Raw => "matte",
        MaskExportSource::Auto => unreachable!(),
    };
    let output_path = cmd
        .output
        .clone()
        .unwrap_or_else(|| derive_variant_path(&cmd.input, default_suffix, "png"));

    match mask_source {
        MaskExportSource::Processed => {
            let mask = matte.clone().processed_with(&mask_pipeline)?;
            mask.save(&output_path)?;
            println!("Processed mask PNG saved to {}", output_path.display());
        }
        MaskExportSource::Auto => unreachable!(),
        MaskExportSource::Raw => {
            matte.save(&output_path)?;
            println!("Matte PNG saved to {}", output_path.display());
        }
    }

    Ok(())
}
