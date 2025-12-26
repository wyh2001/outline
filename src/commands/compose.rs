use outline::{
    ForegroundHandle, MaskAlphaMode, MaskFill, MaskHandle, OutlineResult, alpha_composite,
    create_rgba_layer_from_mask,
};

use crate::cli::{BgAlphaModeArg, ComposeCommand, GlobalOptions, MaskSourceArg};

use super::utils::{
    build_outline, derive_variant_path, processing_requested, resolve_export_path,
    resolve_mask_source_arg, warn_if_soft_conflict,
};

/// Run the compose command.
pub fn run(global: &GlobalOptions, cmd: ComposeCommand) -> OutlineResult<()> {
    let outline = build_outline(global, &cmd.mask_processing);
    let session = outline.for_image(&cmd.input)?;
    let matte = session.matte();

    let output_path = cmd
        .output
        .clone()
        .unwrap_or_else(|| derive_variant_path(&cmd.input, "composite", "png"));

    let proc_requested = processing_requested(&cmd.mask_processing);

    // Resolve fg/bg mask source with compose-specific auto strategy:
    // - fg: auto -> raw (soft edges for natural foreground)
    // - bg: auto -> processed if processing requested (hard edges for clean background)
    let fg_src = match cmd.fg_mask_source {
        MaskSourceArg::Auto => MaskSourceArg::Raw,
        other => other,
    };
    let bg_src = resolve_mask_source_arg(cmd.bg_mask_source, proc_requested);

    debug_assert!(
        !matches!(fg_src, MaskSourceArg::Auto),
        "fg_src should be resolved from Auto"
    );
    debug_assert!(
        !matches!(bg_src, MaskSourceArg::Auto),
        "bg_src should be resolved from Auto"
    );

    // Warn if background uses processed but binary is disabled with dilate/fill-holes
    if matches!(bg_src, MaskSourceArg::Processed) {
        warn_if_soft_conflict(&cmd.mask_processing, "background layer");
    }

    // Pre-compute processed mask if any branch needs it
    let needs_processed = matches!(fg_src, MaskSourceArg::Processed)
        || matches!(bg_src, MaskSourceArg::Processed)
        || cmd.export_mask.is_some();

    let processed: Option<MaskHandle> = if needs_processed {
        Some(matte.clone().processed()?)
    } else {
        None
    };

    // Generate foreground
    let foreground: ForegroundHandle = match (fg_src, processed.as_ref()) {
        (MaskSourceArg::Raw, _) => matte.foreground()?,
        (MaskSourceArg::Processed, Some(proc)) => proc.foreground()?,
        (MaskSourceArg::Processed, None) => unreachable!("processed guaranteed"),
        (MaskSourceArg::Auto, _) => unreachable!("auto resolved"),
    };

    // Build MaskFill for background layer
    let bg_alpha_mode = match cmd.bg_alpha_mode {
        BgAlphaModeArg::UseMask => MaskAlphaMode::UseMask,
        BgAlphaModeArg::Scale => MaskAlphaMode::Scale(cmd.bg_alpha_scale),
        BgAlphaModeArg::Solid => MaskAlphaMode::Solid(cmd.bg_solid_alpha),
    };
    let bg_fill = MaskFill::new(cmd.bg_color).with_alpha_mode(bg_alpha_mode);

    // Create background layer
    let bg_layer = match (bg_src, processed.as_ref()) {
        (MaskSourceArg::Raw, _) => create_rgba_layer_from_mask(session.raw_matte(), bg_fill),
        (MaskSourceArg::Processed, Some(proc)) => {
            create_rgba_layer_from_mask(proc.image(), bg_fill)
        }
        (MaskSourceArg::Processed, None) => unreachable!("processed guaranteed"),
        (MaskSourceArg::Auto, _) => unreachable!("auto resolved"),
    };

    // Composite foreground over background (Porter-Duff over)
    let result = alpha_composite(&bg_layer, foreground.image());

    // Save final result
    result.save(&output_path)?;
    println!("Composite PNG saved to {}", output_path.display());

    // Optional exports
    if let Some(path) = resolve_export_path(&cmd.export_foreground, &cmd.input, "foreground") {
        foreground.save(&path)?;
        println!("Foreground PNG saved to {}", path.display());
    }

    if let Some(path) = resolve_export_path(&cmd.export_matte, &cmd.input, "matte") {
        session.raw_matte().save(&path)?;
        println!("Matte PNG saved to {}", path.display());
    }

    if let Some(path) = resolve_export_path(&cmd.export_mask, &cmd.input, "mask") {
        let proc = processed
            .as_ref()
            .expect("processed mask must exist when export_mask is requested");
        proc.save(&path)?;
        println!("Processed mask PNG saved to {}", path.display());
    }

    if let Some(path) = resolve_export_path(&cmd.export_bg_layer, &cmd.input, "bg-layer") {
        bg_layer.save(&path)?;
        println!("Background layer PNG saved to {}", path.display());
    }

    Ok(())
}
