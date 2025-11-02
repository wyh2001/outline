use outline::{MaskProcessingOptions, OutlineResult};

use crate::cli::{BinaryOption, GlobalOptions, MaskCommand, MaskExportSource};

use super::utils::{build_outline, derive_variant_path};

/// The main function to run the mask command.
pub fn run(global: &GlobalOptions, cmd: MaskCommand) -> OutlineResult<()> {
    let outline = build_outline(global, &cmd.mask_processing);
    let session = outline.for_image(&cmd.input)?;
    let matte = session.matte();
    let defaults = MaskProcessingOptions::default();
    let processing_requested = cmd.mask_processing.binary == BinaryOption::Enabled
        || cmd.mask_processing.blur.is_some()
        || cmd.mask_processing.dilate.is_some()
        || cmd.mask_processing.fill_holes
        || cmd.mask_processing.mask_threshold != defaults.mask_threshold;

    let mask_source = match cmd.mask_source {
        MaskExportSource::Auto => {
            if processing_requested {
                MaskExportSource::Processed
            } else {
                MaskExportSource::Raw
            }
        }
        other => other,
    };

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
            let mask = matte.clone().processed(None)?;
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
