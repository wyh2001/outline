use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use image::imageops::FilterType;
use outline::MaskProcessingOptions;
use visioncortex::PathSimplifyMode;
use vtracer::{ColorMode, Hierarchical};

/// Command line interface definition.
#[derive(Parser, Debug)]
#[command(author, version, about, propagate_version = true)]
pub struct Cli {
    #[command(flatten)]
    pub global: GlobalOptions,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Args, Debug)]
pub struct GlobalOptions {
    /// ONNX model path
    #[arg(short = 'm', long, default_value = "model.onnx")]
    pub model: PathBuf,
    /// Intra-op thread count for ORT (None to let ORT decide)
    #[arg(long)]
    pub intra_threads: Option<usize>,
    /// Filter used when resizing the input before inference
    #[arg(long = "input-resample-filter", value_enum, default_value_t = ResampleFilter::Triangle)]
    pub input_resample_filter: ResampleFilter,
    /// Filter used when resizing the matte back to the original resolution
    #[arg(long = "output-resample-filter", value_enum, default_value_t = ResampleFilter::Lanczos3)]
    pub output_resample_filter: ResampleFilter,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Export only the matte/mask as a PNG
    Mask(MaskCommand),
    /// Remove the background and export the foreground PNG
    Cut(CutCommand),
    /// Trace the subject into an SVG outline
    Trace(TraceCommand),
}

/// Resampling filters for image resizing.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ResampleFilter {
    Nearest,
    Triangle,
    CatmullRom,
    Gaussian,
    Lanczos3,
}

impl From<ResampleFilter> for FilterType {
    /// Convert ResampleFilter to image::imageops::FilterType.
    fn from(value: ResampleFilter) -> Self {
        match value {
            ResampleFilter::Nearest => FilterType::Nearest,
            ResampleFilter::Triangle => FilterType::Triangle,
            ResampleFilter::CatmullRom => FilterType::CatmullRom,
            ResampleFilter::Gaussian => FilterType::Gaussian,
            ResampleFilter::Lanczos3 => FilterType::Lanczos3,
        }
    }
}

#[derive(Args, Debug)]
pub struct MaskCommand {
    /// Input image path
    pub input: PathBuf,
    /// Output path (defaults to `<name>-matte.png` or `<name>-mask.png`)
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Export the processed binary mask instead of the raw matte
    #[arg(long)]
    pub binary: bool,
    #[command(flatten)]
    pub mask_processing: MaskProcessingArgs,
}

#[derive(Args, Debug)]
pub struct CutCommand {
    /// Input image path
    pub input: PathBuf,
    /// Foreground PNG output path (defaults to `<name>-foreground.png`)
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Save the raw matte alongside the foreground PNG
    #[arg(long = "export-matte", value_name = "PATH", num_args = 0..=1)]
    pub export_matte: Option<Option<PathBuf>>,
    /// Save the processed binary mask alongside the foreground PNG
    #[arg(long = "export-mask", value_name = "PATH", num_args = 0..=1)]
    pub export_mask: Option<Option<PathBuf>>,
    /// Select which mask is used for the foreground alpha channel
    #[arg(long = "alpha-source", value_enum, default_value_t = AlphaFromArg::Raw)]
    pub alpha_source: AlphaFromArg,
    #[command(flatten)]
    pub mask_processing: MaskProcessingArgs,
}

#[derive(Args, Debug)]
pub struct TraceCommand {
    /// Input image path
    pub input: PathBuf,
    /// Output SVG path (defaults to input name with `.svg`)
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Which mask to use for tracing (auto prefers processed)
    #[arg(long = "mask-source", value_enum, default_value_t = MaskSourceArg::Auto)]
    pub mask_source: MaskSourceArg,
    #[command(flatten)]
    pub mask_processing: MaskProcessingArgs,
    #[command(flatten)]
    pub trace_options: TraceOptionsArgs,
}

#[derive(Args, Debug)]
pub struct MaskProcessingArgs {
    /// Enable gaussian blur before thresholding
    #[arg(long)]
    pub blur: bool,
    /// Sigma used when gaussian blur is enabled
    #[arg(long = "blur-sigma", default_value_t = 6.0)]
    pub blur_sigma: f32,
    /// Threshold applied to the matte (0-255 or 0.0-1.0)
    #[arg(long = "mask-threshold", default_value_t = 120, value_parser = parse_mask_threshold)]
    pub mask_threshold: u8,
    /// Enable dilation after thresholding
    #[arg(long)]
    pub dilate: bool,
    /// Dilation radius in pixels
    #[arg(long = "dilation-radius", default_value_t = 5.0)]
    pub dilation_radius: f32,
    /// Fill enclosed holes in the mask before vectorization
    #[arg(long = "fill-holes")]
    pub fill_holes: bool,
}

impl From<&MaskProcessingArgs> for MaskProcessingOptions {
    fn from(args: &MaskProcessingArgs) -> Self {
        Self {
            blur: args.blur,
            blur_sigma: args.blur_sigma,
            mask_threshold: args.mask_threshold,
            dilate: args.dilate,
            dilation_radius: args.dilation_radius,
            fill_holes: args.fill_holes,
        }
    }
}

fn parse_mask_threshold(value: &str) -> Result<u8, String> {
    if let Ok(int_value) = value.parse::<u8>() {
        return Ok(int_value);
    }

    let float_value = value
        .parse::<f32>()
        .map_err(|_| format!("mask threshold must be numeric (0-255 or 0.0-1.0), got `{value}`"))?;

    if (0.0..=1.0).contains(&float_value) {
        let scaled = (float_value * 255.0).round() as i32;
        return Ok(scaled.clamp(0, 255) as u8);
    }

    if float_value.fract().abs() <= f32::EPSILON && (0.0..=255.0).contains(&float_value) {
        return Ok(float_value as u8);
    }

    Err(format!(
        "mask threshold {value} is out of range; expected 0-255 or 0.0-1.0"
    ))
}

/// The argument to specify which alpha source to use.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum AlphaFromArg {
    Raw,
    Processed,
}

/// The argument to specify which mask source to use.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum MaskSourceArg {
    Raw,
    Processed,
    Auto,
}

/// Tracing color modes for SVG vectorization.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum TracerColorMode {
    Color,
    Binary,
}

impl From<TracerColorMode> for ColorMode {
    /// Convert TracerColorMode to vtracer::ColorMode.
    fn from(value: TracerColorMode) -> Self {
        match value {
            TracerColorMode::Color => ColorMode::Color,
            TracerColorMode::Binary => ColorMode::Binary,
        }
    }
}

/// Hierarchical tracing modes for SVG vectorization.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum TracerHierarchy {
    Stacked,
    Cutout,
}

impl From<TracerHierarchy> for Hierarchical {
    /// Convert TracerHierarchy to vtracer::Hierarchical.
    fn from(value: TracerHierarchy) -> Self {
        match value {
            TracerHierarchy::Stacked => Hierarchical::Stacked,
            TracerHierarchy::Cutout => Hierarchical::Cutout,
        }
    }
}

/// Path simplification modes for SVG vectorization.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum TracerMode {
    None,
    Polygon,
    Spline,
}

impl From<TracerMode> for PathSimplifyMode {
    /// Convert TracerMode to vtracer::PathSimplifyMode.
    fn from(value: TracerMode) -> Self {
        match value {
            TracerMode::None => PathSimplifyMode::None,
            TracerMode::Polygon => PathSimplifyMode::Polygon,
            TracerMode::Spline => PathSimplifyMode::Spline,
        }
    }
}

#[derive(Args, Debug)]
pub struct TraceOptionsArgs {
    /// Tracing color mode
    #[arg(long = "color-mode", value_enum, default_value_t = TracerColorMode::Binary)]
    pub color_mode: TracerColorMode,
    /// Hierarchical tracing mode
    #[arg(long = "hierarchy", value_enum, default_value_t = TracerHierarchy::Stacked)]
    pub hierarchy: TracerHierarchy,
    /// Path simplification mode
    #[arg(long = "mode", value_enum, default_value_t = TracerMode::Spline)]
    pub mode: TracerMode,
    /// Speckle filter size used by the tracer
    #[arg(long = "filter-speckle", default_value_t = 4)]
    pub filter_speckle: usize,
    /// Color precision override (significant bits per RGB channel)
    #[arg(long = "color-precision", default_value_t = 6)]
    pub color_precision: i32,
    /// Layer difference / gradient step override
    #[arg(long = "layer-difference", default_value_t = 16)]
    pub layer_difference: i32,
    /// Corner threshold override in degrees
    #[arg(long = "corner-threshold", default_value_t = 60)]
    pub corner_threshold: i32,
    /// Segment length threshold override
    #[arg(long = "length-threshold", default_value_t = 4.0)]
    pub length_threshold: f64,
    /// Maximum subdivision iterations override
    #[arg(long = "max-iterations", default_value_t = 10)]
    pub max_iterations: usize,
    /// Splice threshold override in degrees
    #[arg(long = "splice-threshold", default_value_t = 45)]
    pub splice_threshold: i32,
    /// Path precision override (decimal places)
    #[arg(long = "path-precision")]
    pub path_precision: Option<u32>,
    /// Disable explicit path precision override
    #[arg(long = "no-path-precision")]
    pub no_path_precision: bool,
    /// Invert foreground/background in the output SVG
    #[arg(long = "invert-svg")]
    pub invert_svg: bool,
}
