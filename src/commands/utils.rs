use std::path::{Path, PathBuf};

use outline::{MaskProcessingOptions, Outline};

use crate::cli::{GlobalOptions, MaskProcessingArgs};

/// The convenience function to build an Outline instance with the input global and mask processing options.
pub fn build_outline(global: &GlobalOptions, mask_args: &MaskProcessingArgs) -> Outline {
    let mask_processing: MaskProcessingOptions = mask_args.into();
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
