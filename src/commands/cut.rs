use outline::{MaskHandle, MaskProcessingOptions, MatteHandle, OutlineResult};

use crate::cli::{AlphaFromArg, CutCommand, GlobalOptions};

use super::utils::{build_outline, derive_variant_path};

/// The main function to run the cut command.
pub fn run(global: &GlobalOptions, cmd: CutCommand) -> OutlineResult<()> {
    let mut defaults = MaskProcessingOptions::default();
    defaults.binary = false; // Default to no binary for cut command specifically
    let outline = build_outline(global, &cmd.mask_processing, defaults.clone());
    let session = outline.for_image(&cmd.input)?;
    let matte = session.matte();
    let output_path = cmd
        .output
        .clone()
        .unwrap_or_else(|| derive_variant_path(&cmd.input, "foreground", "png"));

    let save_mask_path = match &cmd.export_matte {
        Some(Some(path)) => Some(path.clone()),
        Some(None) => Some(derive_variant_path(&cmd.input, "matte", "png")),
        None => None,
    };

    let save_processed_mask_path = match &cmd.export_mask {
        Some(Some(path)) => Some(path.clone()),
        Some(None) => Some(derive_variant_path(&cmd.input, "mask", "png")),
        None => None,
    };

    let mut processed_mask: Option<MaskHandle> = None;
    let processing_requested = cmd.mask_processing.binary
        || cmd.mask_processing.blur.is_some()
        || cmd.mask_processing.dilate.is_some()
        || cmd.mask_processing.fill_holes
        || cmd.mask_processing.mask_threshold != defaults.mask_threshold;

    let alpha_source = match cmd.alpha_source {
        AlphaFromArg::Auto => {
            if processing_requested {
                AlphaFromArg::Processed
            } else {
                AlphaFromArg::Raw
            }
        }
        other => other,
    };

    let needs_processed_mask =
        matches!(alpha_source, AlphaFromArg::Processed) || cmd.export_mask.is_some();
    if needs_processed_mask
        && !cmd.mask_processing.binary
        && (cmd.mask_processing.dilate.is_some() || cmd.mask_processing.fill_holes)
    {
        eprintln!(
            "Warning: --no-binary disables thresholding, but dilation/fill-holes assume a hard mask; processed output may be unexpected."
        );
    }

    let mut ensure_processed = |matte: &MatteHandle| -> OutlineResult<MaskHandle> {
        if let Some(mask) = &processed_mask {
            Ok(mask.clone())
        } else {
            let mask = matte.clone().processed(None)?;
            processed_mask = Some(mask.clone());
            Ok(mask)
        }
    };

    let foreground = match alpha_source {
        AlphaFromArg::Raw => matte.foreground()?,
        AlphaFromArg::Processed => ensure_processed(&matte)?.foreground()?,
        AlphaFromArg::Auto => unreachable!(),
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
