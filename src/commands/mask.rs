use outline::OutlineResult;

use crate::cli::{BinaryOption, GlobalOptions, MaskCommand, MaskExportSource, MaskProcessingArgs};

use super::utils::{build_outline, derive_variant_path, processing_requested};

fn resolve_mask_source(requested: MaskExportSource, args: &MaskProcessingArgs) -> MaskExportSource {
    match requested {
        MaskExportSource::Auto => {
            if processing_requested(args) {
                MaskExportSource::Processed
            } else {
                MaskExportSource::Raw
            }
        }
        other => other,
    }
}

/// The main function to run the mask command.
pub fn run(global: &GlobalOptions, cmd: MaskCommand) -> OutlineResult<()> {
    let outline = build_outline(global, &cmd.mask_processing);
    let session = outline.for_image(&cmd.input)?;
    let matte = session.matte();
    let mask_source = resolve_mask_source(cmd.mask_source, &cmd.mask_processing);

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
            if cmd.mask_processing.binary == BinaryOption::Disabled
                && (cmd.mask_processing.dilate.is_some() || cmd.mask_processing.fill_holes)
            {
                eprintln!(
                    "Warning: --no-binary disables thresholding, but dilation/fill-holes assume a hard mask; output may be unexpected."
                );
            }
            let mask = matte.clone().processed()?;
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
