use outline::OutlineResult;

use crate::cli::{GlobalOptions, MaskCommand, MaskExportSource};

use super::utils::{build_outline, derive_variant_path};

/// The main function to run the mask command.
pub fn run(global: &GlobalOptions, cmd: MaskCommand) -> OutlineResult<()> {
    let outline = build_outline(global, &cmd.mask_processing);
    let session = outline.for_image(&cmd.input)?;
    let matte = session.matte();
    let default_suffix = match cmd.mask_source {
        MaskExportSource::Processed => "mask",
        MaskExportSource::Raw => "matte",
    };
    let output_path = cmd
        .output
        .clone()
        .unwrap_or_else(|| derive_variant_path(&cmd.input, default_suffix, "png"));

    match cmd.mask_source {
        MaskExportSource::Processed => {
            if !cmd.mask_processing.binary
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
        MaskExportSource::Raw => {
            matte.save(&output_path)?;
            println!("Matte PNG saved to {}", output_path.display());
        }
    }

    Ok(())
}
