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

/// Resolve an export path from an optional double-Option field.
/// Returns Some(path) if export is requested, None otherwise.
pub fn resolve_export_path(
    opt: &Option<Option<PathBuf>>,
    input: &Path,
    suffix: &str,
) -> Option<PathBuf> {
    opt.as_ref().map(|inner| {
        inner
            .clone()
            .unwrap_or_else(|| derive_variant_path(input, suffix, "png"))
    })
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

/// Check if there's a conflict between soft mask mode and operations that assume hard masks.
/// Returns true if --no-binary is set but dilation or fill-holes are requested.
pub fn has_soft_conflict(args: &MaskProcessingArgs) -> bool {
    args.binary == BinaryOption::Disabled && (args.dilate.is_some() || args.fill_holes)
}

/// Emit a warning when dilation/fill-holes are requested but thresholding is disabled.
pub fn warn_if_soft_conflict(args: &MaskProcessingArgs, context: &str) {
    if has_soft_conflict(args) {
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

#[cfg(test)]
mod tests {
    use super::*;

    mod derive_variant_path {
        use super::*;

        #[test]
        fn basic() {
            let input = Path::new("/path/to/image.png");
            let result = derive_variant_path(input, "matte", "png");
            assert_eq!(result, PathBuf::from("/path/to/image-matte.png"));
        }

        #[test]
        fn different_extension() {
            let input = Path::new("/path/to/photo.jpg");
            let result = derive_variant_path(input, "mask", "png");
            assert_eq!(result, PathBuf::from("/path/to/photo-mask.png"));
        }

        #[test]
        fn no_extension() {
            let input = Path::new("/path/to/image");
            let result = derive_variant_path(input, "foreground", "png");
            assert_eq!(result, PathBuf::from("/path/to/image-foreground.png"));
        }

        #[test]
        fn relative_path() {
            let input = Path::new("image.png");
            let result = derive_variant_path(input, "composite", "png");
            assert_eq!(result, PathBuf::from("image-composite.png"));
        }

        #[test]
        fn multi_dot_name() {
            // file_stem() returns "archive.tar", not "archive"
            let input = Path::new("/path/to/archive.tar.gz");
            let result = derive_variant_path(input, "mask", "png");
            assert_eq!(result, PathBuf::from("/path/to/archive.tar-mask.png"));
        }
    }

    mod resolve_export_path {
        use super::*;

        #[test]
        fn none_returns_none() {
            let opt: Option<Option<PathBuf>> = None;
            let input = Path::new("/path/to/image.png");
            let result = resolve_export_path(&opt, input, "matte");
            assert_eq!(result, None);
        }

        #[test]
        fn some_none_uses_default() {
            let opt: Option<Option<PathBuf>> = Some(None);
            let input = Path::new("/path/to/image.png");
            let result = resolve_export_path(&opt, input, "matte");
            assert_eq!(result, Some(PathBuf::from("/path/to/image-matte.png")));
        }

        #[test]
        fn some_some_uses_custom() {
            let custom_path = PathBuf::from("/custom/output.png");
            let opt: Option<Option<PathBuf>> = Some(Some(custom_path.clone()));
            let input = Path::new("/path/to/image.png");
            let result = resolve_export_path(&opt, input, "matte");
            assert_eq!(result, Some(custom_path));
        }

        #[test]
        fn different_suffixes() {
            let opt: Option<Option<PathBuf>> = Some(None);
            let input = Path::new("photo.jpg");

            assert_eq!(
                resolve_export_path(&opt, input, "foreground"),
                Some(PathBuf::from("photo-foreground.png"))
            );
            assert_eq!(
                resolve_export_path(&opt, input, "mask"),
                Some(PathBuf::from("photo-mask.png"))
            );
            assert_eq!(
                resolve_export_path(&opt, input, "bg-layer"),
                Some(PathBuf::from("photo-bg-layer.png"))
            );
        }
    }

    mod derive_svg_path {
        use super::*;

        #[test]
        fn changes_extension() {
            let input = Path::new("/path/to/image.png");
            let result = derive_svg_path(input);
            assert_eq!(result, PathBuf::from("/path/to/image.svg"));
        }

        #[test]
        fn no_extension() {
            let input = Path::new("/path/to/image");
            let result = derive_svg_path(input);
            assert_eq!(result, PathBuf::from("/path/to/image.svg"));
        }
    }

    mod resolve_alpha_source {
        use super::*;

        #[test]
        fn auto_with_processing() {
            let result = resolve_alpha_source(AlphaFromArg::Auto, true);
            assert!(matches!(result, AlphaFromArg::Processed));
        }

        #[test]
        fn auto_without_processing() {
            let result = resolve_alpha_source(AlphaFromArg::Auto, false);
            assert!(matches!(result, AlphaFromArg::Raw));
        }

        #[test]
        fn explicit_raw() {
            let result = resolve_alpha_source(AlphaFromArg::Raw, true);
            assert!(matches!(result, AlphaFromArg::Raw));
        }

        #[test]
        fn explicit_processed() {
            let result = resolve_alpha_source(AlphaFromArg::Processed, false);
            assert!(matches!(result, AlphaFromArg::Processed));
        }
    }

    mod resolve_mask_source_arg {
        use super::*;

        #[test]
        fn auto_with_processing() {
            let result = resolve_mask_source_arg(MaskSourceArg::Auto, true);
            assert!(matches!(result, MaskSourceArg::Processed));
        }

        #[test]
        fn auto_without_processing() {
            let result = resolve_mask_source_arg(MaskSourceArg::Auto, false);
            assert!(matches!(result, MaskSourceArg::Raw));
        }

        #[test]
        fn explicit_raw() {
            let result = resolve_mask_source_arg(MaskSourceArg::Raw, true);
            assert!(matches!(result, MaskSourceArg::Raw));
        }

        #[test]
        fn explicit_processed() {
            let result = resolve_mask_source_arg(MaskSourceArg::Processed, false);
            assert!(matches!(result, MaskSourceArg::Processed));
        }
    }

    mod resolve_mask_export_source {
        use super::*;

        #[test]
        fn auto_with_processing() {
            let result = resolve_mask_export_source(MaskExportSource::Auto, true);
            assert!(matches!(result, MaskExportSource::Processed));
        }

        #[test]
        fn auto_without_processing() {
            let result = resolve_mask_export_source(MaskExportSource::Auto, false);
            assert!(matches!(result, MaskExportSource::Raw));
        }

        #[test]
        fn explicit_raw() {
            let result = resolve_mask_export_source(MaskExportSource::Raw, true);
            assert!(matches!(result, MaskExportSource::Raw));
        }

        #[test]
        fn explicit_processed() {
            let result = resolve_mask_export_source(MaskExportSource::Processed, false);
            assert!(matches!(result, MaskExportSource::Processed));
        }
    }

    mod has_soft_conflict {
        use super::*;

        fn make_args(
            binary: BinaryOption,
            dilate: Option<f32>,
            fill_holes: bool,
        ) -> MaskProcessingArgs {
            MaskProcessingArgs {
                blur: None,
                mask_threshold: 120,
                binary,
                dilate,
                fill_holes,
            }
        }

        #[test]
        fn no_conflict_when_binary_enabled() {
            let args = make_args(BinaryOption::Enabled, Some(5.0), true);
            assert!(!has_soft_conflict(&args));
        }

        #[test]
        fn no_conflict_when_binary_auto() {
            let args = make_args(BinaryOption::Auto, Some(5.0), true);
            assert!(!has_soft_conflict(&args));
        }

        #[test]
        fn no_conflict_when_disabled_without_dilate_or_fill() {
            let args = make_args(BinaryOption::Disabled, None, false);
            assert!(!has_soft_conflict(&args));
        }

        #[test]
        fn conflict_when_disabled_with_dilate() {
            let args = make_args(BinaryOption::Disabled, Some(5.0), false);
            assert!(has_soft_conflict(&args));
        }

        #[test]
        fn conflict_when_disabled_with_fill_holes() {
            let args = make_args(BinaryOption::Disabled, None, true);
            assert!(has_soft_conflict(&args));
        }

        #[test]
        fn conflict_when_disabled_with_both() {
            let args = make_args(BinaryOption::Disabled, Some(5.0), true);
            assert!(has_soft_conflict(&args));
        }
    }
}
