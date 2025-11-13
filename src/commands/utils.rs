use std::path::{Path, PathBuf};

use outline::{MaskProcessingOptions, Outline};

use crate::cli::{
    AlphaFromArg, BinaryOption, GlobalOptions, MaskExportSource, MaskProcessingArgs, MaskSourceArg,
};

/// The convenience function to build an Outline instance with the input global and mask processing options.
pub fn build_outline(global: &GlobalOptions, mask_args: &MaskProcessingArgs) -> Outline {
    let mask_processing = mask_args.into();
    Outline::new(global.model.clone())
        .with_input_resize_filter(global.input_resample_filter.into())
        .with_output_resize_filter(global.output_resample_filter.into())
        .with_intra_threads(global.intra_threads)
        .with_default_mask_processing(mask_processing)
}

/// Derive a variant file path by appending a suffix before the extension.
pub fn derive_variant_path(input: &Path, suffix: &str, extension: &str) -> PathBuf {
    let mut derived = input.to_path_buf();
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| suffix.to_string());
    let filename = format!("{}-{}.{}", stem, suffix, extension);
    derived.set_file_name(filename);
    derived
}

/// Derive an SVG file path by changing the extension to "svg".
pub fn derive_svg_path(input: &Path) -> PathBuf {
    let mut path = input.to_path_buf();
    path.set_extension("svg");
    path
}

/// Determine if any mask processing is requested based on the provided arguments.
pub fn processing_requested(args: &MaskProcessingArgs) -> bool {
    let derived: MaskProcessingOptions = args.into();
    derived != MaskProcessingOptions::default()
}

/// Emit a warning when dilation/fill-holes are requested but thresholding is disabled.
pub fn warn_if_soft_conflict(args: &MaskProcessingArgs, context: &str) {
    if args.binary == BinaryOption::Disabled && (args.dilate.is_some() || args.fill_holes) {
        eprintln!(
            "Warning: --no-binary disables thresholding, but dilation/fill-holes assume a hard mask; {} may be unexpected.",
            context
        );
    }
}

/// Resolve alpha source with Auto behavior.
pub fn resolve_alpha_source(requested: AlphaFromArg, processing_requested: bool) -> AlphaFromArg {
    match requested {
        AlphaFromArg::Auto => {
            if processing_requested {
                AlphaFromArg::Processed
            } else {
                AlphaFromArg::Raw
            }
        }
        other => other,
    }
}

/// Resolve mask source arg with Auto behavior (trace command).
pub fn resolve_mask_source_arg(
    requested: MaskSourceArg,
    processing_requested: bool,
) -> MaskSourceArg {
    match requested {
        MaskSourceArg::Auto => {
            if processing_requested {
                MaskSourceArg::Processed
            } else {
                MaskSourceArg::Raw
            }
        }
        other => other,
    }
}

/// Resolve mask export source with Auto behavior (mask command).
pub fn resolve_mask_export_source(
    requested: MaskExportSource,
    processing_requested: bool,
) -> MaskExportSource {
    match requested {
        MaskExportSource::Auto => {
            if processing_requested {
                MaskExportSource::Processed
            } else {
                MaskExportSource::Raw
            }
        }
        other => other,
    }
}
