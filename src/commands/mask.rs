use outline::OutlineResult;

use crate::cli::{GlobalOptions, MaskCommand};

use super::utils::{build_outline, derive_variant_path};

/// The main function to run the mask command.
pub fn run(global: &GlobalOptions, cmd: MaskCommand) -> OutlineResult<()> {
    let outline = build_outline(global, &cmd.mask_processing);
    let session = outline.for_image(&cmd.input)?;
    let matte = session.matte();
    let default_suffix = if cmd.binary { "mask" } else { "matte" };
    let output_path = cmd
        .output
        .clone()
        .unwrap_or_else(|| derive_variant_path(&cmd.input, default_suffix, "png"));

    if cmd.binary {
        let mask = matte.clone().processed(None)?;
        mask.save(&output_path)?;
        println!("Processed mask PNG saved to {}", output_path.display());
    } else {
        matte.save(&output_path)?;
        println!("Matte PNG saved to {}", output_path.display());
    }

    Ok(())
}
