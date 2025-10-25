use std::fs;
use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand, ValueEnum};
use image::imageops::FilterType;

use outline::foreground::export_foreground;
use outline::run_matte_pipeline;
use outline::{
    AlphaSource, MaskProcessingOptions, MaskSourcePreference, MattePipelineConfig, ModelOptions,
    TraceConfig, select_alpha, select_mask, trace_to_svg_string,
};
use visioncortex::PathSimplifyMode;
use vtracer::{ColorMode, Hierarchical, Preset};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ResampleFilter {
    Nearest,
    Triangle,
    CatmullRom,
    Gaussian,
    Lanczos3,
}

impl From<ResampleFilter> for FilterType {
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

#[derive(Clone, Copy, Debug, ValueEnum)]
enum TracerPreset {
    Bw,
    Poster,
    Photo,
}

impl From<TracerPreset> for Preset {
    fn from(value: TracerPreset) -> Self {
        match value {
            TracerPreset::Bw => Preset::Bw,
            TracerPreset::Poster => Preset::Poster,
            TracerPreset::Photo => Preset::Photo,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum TracerColorMode {
    Color,
    Binary,
}

impl From<TracerColorMode> for ColorMode {
    fn from(value: TracerColorMode) -> Self {
        match value {
            TracerColorMode::Color => ColorMode::Color,
            TracerColorMode::Binary => ColorMode::Binary,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum TracerHierarchy {
    Stacked,
    Cutout,
}

impl From<TracerHierarchy> for Hierarchical {
    fn from(value: TracerHierarchy) -> Self {
        match value {
            TracerHierarchy::Stacked => Hierarchical::Stacked,
            TracerHierarchy::Cutout => Hierarchical::Cutout,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum TracerMode {
    None,
    Polygon,
    Spline,
}

impl From<TracerMode> for PathSimplifyMode {
    fn from(value: TracerMode) -> Self {
        match value {
            TracerMode::None => PathSimplifyMode::None,
            TracerMode::Polygon => PathSimplifyMode::Polygon,
            TracerMode::Spline => PathSimplifyMode::Spline,
        }
    }
}

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

#[derive(Args, Debug)]
struct TraceOptionsArgs {
    /// Tracing preset applied before overrides
    #[arg(long = "preset", value_enum, default_value_t = TracerPreset::Bw)]
    preset: TracerPreset,
    /// Tracing color mode
    #[arg(long = "color-mode", value_enum, default_value_t = TracerColorMode::Binary)]
    color_mode: TracerColorMode,
    /// Hierarchical tracing mode
    #[arg(long = "hierarchy", value_enum, default_value_t = TracerHierarchy::Cutout)]
    hierarchy: TracerHierarchy,
    /// Path simplification mode
    #[arg(long = "mode", value_enum, default_value_t = TracerMode::Spline)]
    mode: TracerMode,
    /// Speckle filter size used by the tracer
    #[arg(long = "filter-speckle", default_value_t = 4)]
    filter_speckle: usize,
    /// Path precision override (decimal places); set --no-path-precision to disable
    #[arg(long = "path-precision", default_value = "8")]
    path_precision: Option<u32>,
    /// Disable explicit path precision override
    #[arg(long = "no-path-precision")]
    no_path_precision: bool,
    /// Invert foreground/background in the output SVG
    #[arg(long = "invert-svg")]
    invert_svg: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum AlphaFromArg {
    Raw,
    Processed,
}

impl From<AlphaFromArg> for AlphaSource {
    fn from(value: AlphaFromArg) -> Self {
        match value {
            AlphaFromArg::Raw => AlphaSource::Raw,
            AlphaFromArg::Processed => AlphaSource::Processed,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MaskSourceArg {
    Raw,
    Processed,
    Auto,
}

impl From<MaskSourceArg> for MaskSourcePreference {
    fn from(value: MaskSourceArg) -> Self {
        match value {
            MaskSourceArg::Raw => MaskSourcePreference::Raw,
            MaskSourceArg::Processed => MaskSourcePreference::Processed,
            MaskSourceArg::Auto => MaskSourcePreference::Auto,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match &cli.command {
        Commands::Mask(cmd) => run_mask(&cli.global, cmd),
        Commands::Cut(cmd) => run_cut(&cli.global, cmd),
        Commands::Trace(cmd) => run_trace(&cli.global, cmd),
    }
}

fn run_mask(global: &GlobalOptions, cmd: &MaskCommand) -> Result<(), Box<dyn std::error::Error>> {
    let pipeline_config = build_pipeline_config(global, &cmd.input, &cmd.mask_processing);
    let result = run_matte_pipeline(&pipeline_config)?;

    let default_suffix = if cmd.binary { "mask" } else { "matte" };
    let output_path = cmd
        .output
        .clone()
        .unwrap_or_else(|| derive_variant_path(&cmd.input, default_suffix, "png"));

    if cmd.binary {
        result.processed_mask.save(&output_path)?;
        println!("Processed mask PNG saved to {}", output_path.display());
    } else {
        result.raw_matte.save(&output_path)?;
        println!("Matte PNG saved to {}", output_path.display());
    }

    Ok(())
}

fn run_cut(global: &GlobalOptions, cmd: &CutCommand) -> Result<(), Box<dyn std::error::Error>> {
    let pipeline_config = build_pipeline_config(global, &cmd.input, &cmd.mask_processing);
    let result = run_matte_pipeline(&pipeline_config)?;

    if let Some(selection) = &cmd.save_mask {
        let path = selection
            .clone()
            .unwrap_or_else(|| derive_variant_path(&cmd.input, "matte", "png"));
        result.raw_matte.save(&path)?;
        println!("Matte PNG saved to {}", path.display());
    }

    if let Some(selection) = &cmd.save_processed_mask {
        let path = selection
            .clone()
            .unwrap_or_else(|| derive_variant_path(&cmd.input, "mask", "png"));
        result.processed_mask.save(&path)?;
        println!("Processed mask PNG saved to {}", path.display());
    }

    let output_path = cmd
        .output
        .clone()
        .unwrap_or_else(|| derive_variant_path(&cmd.input, "foreground", "png"));

    let alpha_image = select_alpha(&result, cmd.alpha_from.into());
    export_foreground(&result.rgb_image, alpha_image, &output_path)?;
    println!("Foreground PNG saved to {}", output_path.display());

    Ok(())
}

fn run_trace(global: &GlobalOptions, cmd: &TraceCommand) -> Result<(), Box<dyn std::error::Error>> {
    let pipeline_config = build_pipeline_config(global, &cmd.input, &cmd.mask_processing);
    let output_path = cmd
        .output
        .clone()
        .unwrap_or_else(|| derive_svg_path(&cmd.input));

    let trace_config = TraceConfig {
        pipeline: pipeline_config,
        svg_path: output_path.clone(),
        mask_preference: cmd.mask_source.into(),
        tracer_preset: cmd.trace_options.preset.into(),
        tracer_color_mode: cmd.trace_options.color_mode.into(),
        tracer_hierarchical: cmd.trace_options.hierarchy.into(),
        tracer_mode: cmd.trace_options.mode.into(),
        tracer_filter_speckle: cmd.trace_options.filter_speckle,
        tracer_path_precision: if cmd.trace_options.no_path_precision {
            None
        } else {
            cmd.trace_options.path_precision
        },
        invert_svg: cmd.trace_options.invert_svg,
    };

    let result = run_matte_pipeline(&trace_config.pipeline)?;
    let mask_image = select_mask(&result, trace_config.mask_preference);
    let svg = trace_to_svg_string(mask_image, &trace_config)?;
    fs::write(&trace_config.svg_path, svg)?;
    println!("SVG saved to {}", trace_config.svg_path.display());

    Ok(())
}

fn build_pipeline_config(
    global: &GlobalOptions,
    input: &Path,
    mask_args: &MaskProcessingArgs,
) -> MattePipelineConfig {
    MattePipelineConfig {
        model: ModelOptions {
            model_path: global.model.clone(),
            image_path: input.to_path_buf(),
            model_resize_filter: global.model_filter.into(),
            matte_resize_filter: global.matte_filter.into(),
            intra_threads: global.intra_threads,
        },
        mask_processing: MaskProcessingOptions {
            blur: mask_args.blur,
            blur_sigma: mask_args.blur_sigma,
            mask_threshold: mask_args.mask_threshold,
            dilate: mask_args.dilate,
            dilation_radius: mask_args.dilation_radius,
            fill_holes: mask_args.fill_holes,
        },
    }
}

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

fn derive_svg_path(input: &Path) -> PathBuf {
    let mut path = input.to_path_buf();
    path.set_extension("svg");
    path
}
