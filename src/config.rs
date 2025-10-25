use std::path::PathBuf;

use image::imageops::FilterType;
use visioncortex::PathSimplifyMode;
use vtracer::{ColorMode, Hierarchical, Preset};

#[derive(Debug, Clone)]
pub struct ModelOptions {
    pub model_path: PathBuf,
    pub image_path: PathBuf,
    pub model_resize_filter: FilterType,
    pub matte_resize_filter: FilterType,
    pub intra_threads: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct MaskProcessingOptions {
    pub blur: bool,
    pub blur_sigma: f32,
    pub mask_threshold: u8,
    pub dilate: bool,
    pub dilation_radius: f32,
    pub fill_holes: bool,
}

#[derive(Debug, Clone)]
pub struct MattePipelineConfig {
    pub model: ModelOptions,
    pub mask_processing: MaskProcessingOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaSource {
    Raw,
    Processed,
}

#[derive(Debug, Clone)]
pub struct CutConfig {
    pub pipeline: MattePipelineConfig,
    pub output_path: PathBuf,
    pub save_mask_path: Option<PathBuf>,
    pub save_processed_mask_path: Option<PathBuf>,
    pub alpha_source: AlphaSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskOutputKind {
    Raw,
    Processed,
}

#[derive(Debug, Clone)]
pub struct MaskCommandConfig {
    pub pipeline: MattePipelineConfig,
    pub output_path: PathBuf,
    pub kind: MaskOutputKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskSourcePreference {
    Raw,
    Processed,
    Auto,
}

#[derive(Debug, Clone)]
pub struct TraceConfig {
    pub pipeline: MattePipelineConfig,
    pub svg_path: PathBuf,
    pub mask_preference: MaskSourcePreference,
    pub tracer_preset: Preset,
    pub tracer_color_mode: ColorMode,
    pub tracer_hierarchical: Hierarchical,
    pub tracer_mode: PathSimplifyMode,
    pub tracer_filter_speckle: usize,
    pub tracer_path_precision: Option<u32>,
    pub invert_svg: bool,
}
