use outline::{MaskHandle, MatteHandle, OutlineResult};

use crate::cli::{AlphaFromArg, CutCommand, GlobalOptions};

use super::utils::{build_outline, derive_variant_path};

/// The main function to run the cut command.
pub fn run(global: &GlobalOptions, cmd: CutCommand) -> OutlineResult<()> {
    let outline = build_outline(global, &cmd.mask_processing);
    let session = outline.for_image(&cmd.input)?;
    let matte = session.matte();
    let output_path = cmd
        .output
        .clone()
        .unwrap_or_else(|| derive_variant_path(&cmd.input, "foreground", "png"));

    let save_mask_path = match &cmd.save_mask {
        Some(Some(path)) => Some(path.clone()),
        Some(None) => Some(derive_variant_path(&cmd.input, "matte", "png")),
        None => None,
    };

    let save_processed_mask_path = match &cmd.save_processed_mask {
        Some(Some(path)) => Some(path.clone()),
        Some(None) => Some(derive_variant_path(&cmd.input, "mask", "png")),
        None => None,
    };

    let mut processed_mask: Option<MaskHandle> = None;

    let mut ensure_processed = |matte: &MatteHandle| -> OutlineResult<MaskHandle> {
        if let Some(mask) = &processed_mask {
            Ok(mask.clone())
        } else {
            let mask = matte.clone().processed(None)?;
            processed_mask = Some(mask.clone());
            Ok(mask)
        }
    };

    let foreground = match cmd.alpha_source {
        AlphaFromArg::Raw => matte.foreground()?,
        AlphaFromArg::Processed => ensure_processed(&matte)?.foreground()?,
    };

    foreground.save(&output_path)?;
    println!("Foreground PNG saved to {}", output_path.display());

    if let Some(path) = &save_mask_path {
        matte.clone().save(path)?;
        println!("Matte PNG saved to {}", path.display());
    }

    if let Some(path) = &save_processed_mask_path {
        ensure_processed(&matte)?.save(path)?;
        println!("Processed mask PNG saved to {}", path.display());
    }

    Ok(())
}
