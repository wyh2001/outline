use std::fs;
use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand, ValueEnum};
use image::imageops::FilterType;

use outline::{MaskHandle, MaskProcessingOptions, MatteHandle, Outline, OutlineResult};
#[cfg(feature = "vectorizer-vtracer")]
use outline::{TraceOptions, VtracerSvgVectorizer};
use visioncortex::PathSimplifyMode;
use vtracer::{ColorMode, Hierarchical};

/// Resampling filters for image resizing.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum ResampleFilter {
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

/// Tracing color modes for SVG vectorization.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum TracerColorMode {
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
enum TracerHierarchy {
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
enum TracerMode {
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

/// Command line interface definition.
#[derive(Parser, Debug)]
#[command(author, version, about, propagate_version = true)]
struct Cli {
    #[command(flatten)]
    global: GlobalOptions,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Args, Debug)]
struct GlobalOptions {
    /// ONNX model path
    #[arg(short = 'm', long, default_value = "model.onnx")]
    model: PathBuf,
    /// Intra-op thread count for ORT (None to let ORT decide)
    #[arg(long)]
    intra_threads: Option<usize>,
    /// Filter used when resizing the input before inference
    #[arg(long = "model-filter", value_enum, default_value_t = ResampleFilter::Triangle)]
    model_filter: ResampleFilter,
    /// Filter used when resizing the matte back to the original resolution
    #[arg(long = "matte-filter", value_enum, default_value_t = ResampleFilter::Lanczos3)]
    matte_filter: ResampleFilter,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Export only the matte/mask as a PNG
    Mask(MaskCommand),
    /// Remove the background and export the foreground PNG
    Cut(CutCommand),
    /// Trace the subject into an SVG outline
    Trace(TraceCommand),
}

#[derive(Args, Debug)]
struct MaskCommand {
    /// Input image path
    input: PathBuf,
    /// Output path (defaults to `<name>-matte.png` or `<name>-mask.png`)
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Export the processed binary mask instead of the raw matte
    #[arg(long)]
    binary: bool,
    #[command(flatten)]
    mask_processing: MaskProcessingArgs,
}

#[derive(Args, Debug)]
struct CutCommand {
    /// Input image path
    input: PathBuf,
    /// Foreground PNG output path (defaults to `<name>-foreground.png`)
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Save the raw matte alongside the foreground PNG
    #[arg(long = "save-mask", value_name = "PATH", num_args = 0..=1)]
    save_mask: Option<Option<PathBuf>>,
    /// Save the processed binary mask alongside the foreground PNG
    #[arg(long = "save-processed-mask", value_name = "PATH", num_args = 0..=1)]
    save_processed_mask: Option<Option<PathBuf>>,
    /// Select which mask is used for the foreground alpha channel
    #[arg(long = "alpha-from", value_enum, default_value_t = AlphaFromArg::Raw)]
    alpha_from: AlphaFromArg,
    #[command(flatten)]
    mask_processing: MaskProcessingArgs,
}

#[derive(Args, Debug)]
struct TraceCommand {
    /// Input image path
    input: PathBuf,
    /// Output SVG path (defaults to input name with `.svg`)
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Which mask to use for tracing (auto prefers processed)
    #[arg(long = "mask-source", value_enum, default_value_t = MaskSourceArg::Auto)]
    mask_source: MaskSourceArg,
    #[command(flatten)]
    mask_processing: MaskProcessingArgs,
    #[command(flatten)]
    trace_options: TraceOptionsArgs,
}

#[derive(Args, Debug)]
struct MaskProcessingArgs {
    /// Enable gaussian blur before thresholding
    #[arg(long)]
    blur: bool,
    /// Sigma used when gaussian blur is enabled
    #[arg(long = "blur-sigma", default_value_t = 6.0)]
    blur_sigma: f32,
    /// Threshold applied to the matte (0-255)
    #[arg(long = "mask-threshold", default_value_t = 120)]
    mask_threshold: u8,
    /// Enable dilation after thresholding
    #[arg(long)]
    dilate: bool,
    /// Dilation radius in pixels
    #[arg(long = "dilation-radius", default_value_t = 5.0)]
    dilation_radius: f32,
    /// Fill enclosed holes in the mask before vectorization
    #[arg(long = "fill-holes")]
    fill_holes: bool,
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

#[derive(Args, Debug)]
struct TraceOptionsArgs {
    /// Tracing color mode
    #[arg(long = "color-mode", value_enum, default_value_t = TracerColorMode::Binary)]
    color_mode: TracerColorMode,
    /// Hierarchical tracing mode
    #[arg(long = "hierarchy", value_enum, default_value_t = TracerHierarchy::Stacked)]
    hierarchy: TracerHierarchy,
    /// Path simplification mode
    #[arg(long = "mode", value_enum, default_value_t = TracerMode::Spline)]
    mode: TracerMode,
    /// Speckle filter size used by the tracer
    #[arg(long = "filter-speckle", default_value_t = 4)]
    filter_speckle: usize,
    /// Color precision override (significant bits per RGB channel)
    #[arg(long = "color-precision", default_value_t = 6)]
    color_precision: i32,
    /// Layer difference / gradient step override
    #[arg(long = "layer-difference", default_value_t = 16)]
    layer_difference: i32,
    /// Corner threshold override in degrees
    #[arg(long = "corner-threshold", default_value_t = 60)]
    corner_threshold: i32,
    /// Segment length threshold override
    #[arg(long = "length-threshold", default_value_t = 4.0)]
    length_threshold: f64,
    /// Maximum subdivision iterations override
    #[arg(long = "max-iterations", default_value_t = 10)]
    max_iterations: usize,
    /// Splice threshold override in degrees
    #[arg(long = "splice-threshold", default_value_t = 45)]
    splice_threshold: i32,
    /// Path precision override (decimal places)
    #[arg(long = "path-precision")]
    path_precision: Option<u32>,
    /// Disable explicit path precision override
    #[arg(long = "no-path-precision")]
    no_path_precision: bool,
    /// Invert foreground/background in the output SVG
    #[arg(long = "invert-svg")]
    invert_svg: bool,
}

/// The argument to specify which alpha source to use.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum AlphaFromArg {
    Raw,
    Processed,
}

/// The argument to specify which mask source to use.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum MaskSourceArg {
    Raw,
    Processed,
    Auto,
}

fn main() -> OutlineResult<()> {
    let cli = Cli::parse();
    match &cli.command {
        Commands::Mask(cmd) => handle_mask(&cli.global, cmd),
        Commands::Cut(cmd) => handle_cut(&cli.global, cmd),
        Commands::Trace(cmd) => handle_trace(&cli.global, cmd),
    }
}

/// Handle the 'mask' command.
fn handle_mask(global: &GlobalOptions, cmd: &MaskCommand) -> OutlineResult<()> {
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

/// Handles the 'cut' command.
fn handle_cut(global: &GlobalOptions, cmd: &CutCommand) -> OutlineResult<()> {
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

    let foreground = match cmd.alpha_from {
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

/// Handle the 'trace' command.
fn handle_trace(global: &GlobalOptions, cmd: &TraceCommand) -> OutlineResult<()> {
    let outline = build_outline(global, &cmd.mask_processing);
    let session = outline.for_image(&cmd.input)?;
    let matte = session.matte();
    let output_path = cmd
        .output
        .clone()
        .unwrap_or_else(|| derive_svg_path(&cmd.input));

    let mut options = TraceOptions::default();
    options.tracer_color_mode = cmd.trace_options.color_mode.into();
    options.tracer_hierarchical = cmd.trace_options.hierarchy.into();
    options.tracer_mode = cmd.trace_options.mode.into();
    options.tracer_filter_speckle = cmd.trace_options.filter_speckle;
    options.tracer_color_precision = cmd.trace_options.color_precision;
    options.tracer_layer_difference = cmd.trace_options.layer_difference;
    options.tracer_corner_threshold = cmd.trace_options.corner_threshold;
    options.tracer_length_threshold = cmd.trace_options.length_threshold;
    options.tracer_max_iterations = cmd.trace_options.max_iterations;
    options.tracer_splice_threshold = cmd.trace_options.splice_threshold;
    if let Some(path_precision) = cmd.trace_options.path_precision {
        options.tracer_path_precision = Some(path_precision);
    }
    if cmd.trace_options.no_path_precision {
        options.tracer_path_precision = None;
    }
    options.invert_svg = cmd.trace_options.invert_svg;

    let vectorizer = VtracerSvgVectorizer;
    let svg = match cmd.mask_source {
        MaskSourceArg::Raw => matte.trace(&vectorizer, &options)?,
        MaskSourceArg::Processed => matte
            .clone()
            .processed(None)?
            .trace(&vectorizer, &options)?,
        MaskSourceArg::Auto => matte
            .clone()
            .processed(None)?
            .trace(&vectorizer, &options)?,
    };
    fs::write(&output_path, &svg)?;
    println!("SVG saved to {}", output_path.display());

    Ok(())
}

/// The convenience function to build an Outline with the input global and mask processing options.
fn build_outline(global: &GlobalOptions, mask_args: &MaskProcessingArgs) -> Outline {
    let mask_processing: MaskProcessingOptions = mask_args.into();
    Outline::new(global.model.clone())
        .with_input_resize_filter(global.model_filter.into())
        .with_output_resize_filter(global.matte_filter.into())
        .with_intra_threads(global.intra_threads)
        .with_default_mask_processing(mask_processing)
}

/// Derive a variant file path by appending a suffix before the extension.
fn derive_variant_path(input: &Path, suffix: &str, extension: &str) -> PathBuf {
    let mut derived = input.to_path_buf();
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| suffix.to_string());
    let filename = format!("{}-{}.{}", stem, suffix, extension);
    derived.set_file_name(filename);
    derived
}

/// Derive an SVG file path by changing the extension to `.svg`.
fn derive_svg_path(input: &Path) -> PathBuf {
    let mut path = input.to_path_buf();
    path.set_extension("svg");
    path
}
