use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum, ValueHint};
use image::imageops::FilterType;
use outline::{MaskProcessingOptions, TraceOptions};
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
    #[arg(
        short = 'm',
        long,
        global = true,
        env = outline::ENV_MODEL_PATH,
        value_hint = ValueHint::FilePath,
        default_value = outline::DEFAULT_MODEL_PATH
    )]
    pub model: PathBuf,
    /// Intra-op thread count for ORT (None to let ORT decide)
    #[arg(long, global = true)]
    pub intra_threads: Option<usize>,
    /// Filter used when resizing the input before inference
    #[arg(long = "input-resample-filter", value_enum, default_value_t = ResampleFilter::Triangle, global = true)]
    pub input_resample_filter: ResampleFilter,
    /// Filter used when resizing the matte back to the original resolution
    #[arg(long = "output-resample-filter", value_enum, default_value_t = ResampleFilter::Lanczos3, global = true)]
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
    /// Compose foreground over a filled background layer
    Compose(ComposeCommand),
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
    /// Select which mask to export
    #[arg(long = "mask-source", value_enum, default_value_t = MaskExportSource::Auto)]
    pub mask_source: MaskExportSource,
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
    #[arg(long = "alpha-source", value_enum, default_value_t = AlphaFromArg::Auto)]
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
pub struct ComposeCommand {
    /// Input image path
    pub input: PathBuf,

    /// Output path (defaults to `<name>-composite.png`)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    // Background layer options
    /// Background fill color (#RGB, #RRGGBB, #RRGGBBAA, r,g,b, or r,g,b,a)
    #[arg(long = "bg-color", default_value = "#FFFFFFFF", value_parser = parse_rgba_color)]
    pub bg_color: [u8; 4],

    /// Background layer alpha mode
    #[arg(long = "bg-alpha-mode", value_enum, default_value_t = BgAlphaModeArg::UseMask)]
    pub bg_alpha_mode: BgAlphaModeArg,

    /// Scale factor for Scale alpha mode
    #[arg(long = "bg-alpha-scale", default_value_t = 1.0)]
    pub bg_alpha_scale: f32,

    /// Fixed alpha for Solid alpha mode (0-255)
    #[arg(long = "bg-solid-alpha", default_value_t = 255)]
    pub bg_solid_alpha: u8,

    /// Which mask to use for background layer (default: auto -> processed if processing requested)
    #[arg(long = "bg-mask-source", value_enum, default_value_t = MaskSourceArg::Auto)]
    pub bg_mask_source: MaskSourceArg,

    // Foreground options
    /// Which mask to use for foreground alpha (default: auto -> raw for soft edges)
    #[arg(long = "fg-mask-source", value_enum, default_value_t = MaskSourceArg::Auto)]
    pub fg_mask_source: MaskSourceArg,

    // Shared mask processing
    #[command(flatten)]
    pub mask_processing: MaskProcessingArgs,

    // Optional exports
    /// Export foreground PNG
    #[arg(long = "export-foreground", value_name = "PATH", num_args = 0..=1)]
    pub export_foreground: Option<Option<PathBuf>>,

    /// Export raw matte PNG
    #[arg(long = "export-matte", value_name = "PATH", num_args = 0..=1)]
    pub export_matte: Option<Option<PathBuf>>,

    /// Export processed mask PNG
    #[arg(long = "export-mask", value_name = "PATH", num_args = 0..=1)]
    pub export_mask: Option<Option<PathBuf>>,

    /// Export background layer PNG
    #[arg(long = "export-bg-layer", value_name = "PATH", num_args = 0..=1)]
    pub export_bg_layer: Option<Option<PathBuf>>,
}

#[derive(Args, Debug)]
pub struct MaskProcessingArgs {
    /// Enable gaussian blur before thresholding (optionally override sigma)
    #[arg(long = "blur", value_name = "SIGMA", num_args = 0..=1, default_missing_value = "6.0")]
    pub blur: Option<f32>,
    /// Threshold applied to the matte (0-255 or 0.0-1.0)
    #[arg(long = "mask-threshold", default_value_t = 120, value_parser = parse_mask_threshold)]
    pub mask_threshold: u8,
    /// Apply thresholding to produce a binary mask (use `--binary enabled|disabled|auto` to choose behaviour)
    #[arg(
        long = "binary",
        value_enum,
        default_value_t = BinaryOption::Auto,
        num_args = 0..=1,
        default_missing_value = "enabled"
    )]
    pub binary: BinaryOption,
    #[arg(long = "dilate", value_name = "RADIUS", num_args = 0..=1, default_missing_value = "5.0")]
    pub dilate: Option<f32>,
    /// Fill enclosed holes in the mask before vectorization
    #[arg(long = "fill-holes")]
    pub fill_holes: bool,
}

impl From<&MaskProcessingArgs> for MaskProcessingOptions {
    fn from(args: &MaskProcessingArgs) -> Self {
        let defaults = MaskProcessingOptions::default();
        Self {
            binary: (args.binary == BinaryOption::Auto
                && (args.dilate.is_some() || args.fill_holes))
                || args.binary == BinaryOption::Enabled,
            blur: args.blur.is_some(),
            blur_sigma: args.blur.unwrap_or(defaults.blur_sigma),
            mask_threshold: args.mask_threshold,
            dilate: args.dilate.is_some(),
            dilation_radius: args.dilate.unwrap_or(defaults.dilation_radius),
            fill_holes: args.fill_holes,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum MaskExportSource {
    Auto,
    Raw,
    Processed,
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

/// Parse an RGBA color from string.
/// Supported formats: #RGB, #RRGGBB, #RRGGBBAA, r,g,b, r,g,b,a
fn parse_rgba_color(s: &str) -> Result<[u8; 4], String> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix('#') {
        parse_hex_color(hex)
    } else {
        parse_csv_color(s)
    }
}

/// Parse a single hex byte from a slice, optionally expanding single chars (e.g., "F" -> "FF").
fn parse_hex_byte(hex: &str, start: usize, len: usize) -> Result<u8, String> {
    let slice = &hex[start..start + len];
    let expanded = if len == 1 {
        // #RGB format: expand single char to double (e.g., "F" -> "FF")
        [slice, slice].concat()
    } else {
        slice.to_string()
    };
    u8::from_str_radix(&expanded, 16).map_err(|_| format!("invalid hex digit in color: {}", slice))
}

fn parse_hex_color(hex: &str) -> Result<[u8; 4], String> {
    // Validate ASCII before byte-slicing to avoid panic on multi-byte UTF-8 characters
    if !hex.is_ascii() {
        return Err("hex color must contain only ASCII characters".to_string());
    }

    match hex.len() {
        3 => Ok([
            parse_hex_byte(hex, 0, 1)?,
            parse_hex_byte(hex, 1, 1)?,
            parse_hex_byte(hex, 2, 1)?,
            255,
        ]),
        6 => Ok([
            parse_hex_byte(hex, 0, 2)?,
            parse_hex_byte(hex, 2, 2)?,
            parse_hex_byte(hex, 4, 2)?,
            255,
        ]),
        8 => Ok([
            parse_hex_byte(hex, 0, 2)?,
            parse_hex_byte(hex, 2, 2)?,
            parse_hex_byte(hex, 4, 2)?,
            parse_hex_byte(hex, 6, 2)?,
        ]),
        _ => Err(format!(
            "invalid hex color length: expected 3, 6, or 8, got {}",
            hex.len()
        )),
    }
}

fn parse_csv_color(s: &str) -> Result<[u8; 4], String> {
    let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();

    let parse = |p: &str, name: &str| -> Result<u8, String> {
        p.parse()
            .map_err(|_| format!("invalid {} component: {}", name, p))
    };

    match parts.as_slice() {
        [r, g, b] => Ok([parse(r, "R")?, parse(g, "G")?, parse(b, "B")?, 255]),
        [r, g, b, a] => Ok([
            parse(r, "R")?,
            parse(g, "G")?,
            parse(b, "B")?,
            parse(a, "A")?,
        ]),
        _ => Err(format!(
            "expected 3 or 4 color components (r,g,b or r,g,b,a), got {}",
            parts.len()
        )),
    }
}

/// The argument to specify if binary mask processing is enabled.
#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum BinaryOption {
    Enabled,
    Disabled,
    Auto,
}

/// The argument to specify which alpha source to use.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum AlphaFromArg {
    Raw,
    Processed,
    Auto,
}

/// The argument to specify which mask source to use.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum MaskSourceArg {
    Raw,
    Processed,
    Auto,
}

/// Alpha mode for background layer fill in compose command.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum BgAlphaModeArg {
    /// Use the mask value directly as alpha.
    #[default]
    UseMask,
    /// Scale the mask alpha by a factor.
    Scale,
    /// Treat any non-zero mask value as a solid alpha.
    Solid,
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
    #[arg(long = "path-precision", conflicts_with = "no_path_precision")]
    pub path_precision: Option<u32>,
    /// Disable explicit path precision override
    #[arg(long = "no-path-precision")]
    pub no_path_precision: bool,
    /// Invert foreground/background in the output SVG
    #[arg(long = "invert-svg")]
    pub invert_svg: bool,
}

impl From<&TraceOptionsArgs> for TraceOptions {
    fn from(args: &TraceOptionsArgs) -> Self {
        let default_opts = TraceOptions::default();
        let tracer_path_precision = if args.no_path_precision {
            None
        } else {
            args.path_precision.or(default_opts.tracer_path_precision)
        };
        Self {
            tracer_color_mode: args.color_mode.into(),
            tracer_hierarchical: args.hierarchy.into(),
            tracer_mode: args.mode.into(),
            tracer_filter_speckle: args.filter_speckle,
            tracer_color_precision: args.color_precision,
            tracer_layer_difference: args.layer_difference,
            tracer_corner_threshold: args.corner_threshold,
            tracer_length_threshold: args.length_threshold,
            tracer_max_iterations: args.max_iterations,
            tracer_splice_threshold: args.splice_threshold,
            tracer_path_precision,
            invert_svg: args.invert_svg,
        }
    }
}
