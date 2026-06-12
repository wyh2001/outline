use std::ffi::OsString;
use std::path::PathBuf;

use clap::{
    ArgMatches, Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum, ValueHint,
    error::ErrorKind,
};
use image::imageops::FilterType;
use outline::{
    ErosionBorderMode, MaskPipeline, MaskProcessingDefaults, ModelInputSize, TraceOptions,
};
use visioncortex::PathSimplifyMode;
use vtracer::{ColorMode, Hierarchical};

// Hard-coded for clap missing values so bare options keep their index.
// Tests keep these synced with `MaskProcessingDefaults`.
const DEFAULT_BLUR_SIGMA: &str = "6.0";
const DEFAULT_MASK_THRESHOLD: &str = "120";
const DEFAULT_MASK_THRESHOLD_VALUE: u8 = 120;
const DEFAULT_DILATION_RADIUS: &str = "5.0";
const DEFAULT_EROSION_RADIUS: &str = "5.0";

/// Command line interface definition.
#[derive(Parser, Debug)]
#[command(author, version, about, propagate_version = true)]
pub struct Cli {
    #[command(flatten)]
    pub global: GlobalOptions,

    #[command(subcommand)]
    pub command: Commands,
}

impl Cli {
    pub fn parse() -> Self {
        match Self::try_parse_from(std::env::args_os()) {
            Ok(cli) => cli,
            Err(err) => err.exit(),
        }
    }

    pub fn try_parse_from<I, T>(itr: I) -> Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString> + Clone,
    {
        let matches = Self::command().try_get_matches_from(itr)?;
        let mut cli = <Self as FromArgMatches>::from_arg_matches(&matches)?;
        cli.populate_ordered_mask_steps(&matches)?;
        Ok(cli)
    }

    fn populate_ordered_mask_steps(&mut self, matches: &ArgMatches) -> Result<(), clap::Error> {
        let Some((_, command_matches)) = matches.subcommand() else {
            return Ok(());
        };

        match &mut self.command {
            Commands::Mask(cmd) => cmd.mask_processing.populate_ordered_steps(command_matches),
            Commands::Cut(cmd) => cmd.mask_processing.populate_ordered_steps(command_matches),
            Commands::Trace(cmd) => cmd.mask_processing.populate_ordered_steps(command_matches),
            #[cfg(feature = "fetch-model")]
            Commands::FetchModel(_) => Ok(()),
        }
    }
}

#[derive(Args, Debug)]
pub struct GlobalOptions {
    /// ONNX model path
    #[arg(
        short = 'm',
        long,
        global = true,
        env = outline::ENV_MODEL_PATH,
        value_hint = ValueHint::FilePath
    )]
    pub model: Option<PathBuf>,
    /// Intra-op thread count for ORT (None to let ORT decide)
    #[arg(long, global = true)]
    pub intra_threads: Option<usize>,
    /// Override model input size when it cannot be inferred
    #[arg(
        long = "model-input-size",
        value_name = "HEIGHTxWIDTH",
        value_parser = parse_model_input_size,
        global = true
    )]
    pub model_input_size: Option<ModelInputSize>,
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
    /// Download the default model from the network
    #[cfg(feature = "fetch-model")]
    FetchModel(FetchModelCommand),
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

/// Command to download the default model.
#[cfg(feature = "fetch-model")]
#[derive(Args, Debug, Clone)]
pub struct FetchModelCommand {
    /// Output path for the downloaded model
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    pub output: Option<PathBuf>,
    /// Overwrite existing model file
    #[arg(long)]
    pub force: bool,
}

#[derive(Args, Debug)]
pub struct MaskProcessingArgs {
    /// Apply gaussian blur (optionally override sigma)
    #[arg(
        long = "blur",
        value_name = "SIGMA",
        num_args = 0..=1,
        default_missing_value = DEFAULT_BLUR_SIGMA
    )]
    pub blur: Option<f32>,
    /// Apply thresholding to binarize the mask (0-255 or 0.0-1.0, optionally override threshold value)
    #[arg(
        long = "threshold",
        value_name = "VALUE",
        num_args = 0..=1,
        value_parser = parse_mask_threshold,
        default_missing_value = DEFAULT_MASK_THRESHOLD
    )]
    pub threshold: Option<u8>,
    /// Disable implicit threshold insertion before hard-mask operations
    #[arg(long = "no-implicit-threshold")]
    pub no_implicit_threshold: bool,
    #[arg(
        long = "dilate",
        value_name = "RADIUS",
        num_args = 0..=1,
        default_missing_value = DEFAULT_DILATION_RADIUS
    )]
    pub dilate: Option<f32>,
    #[arg(
        long = "erode",
        value_name = "RADIUS",
        num_args = 0..=1,
        default_missing_value = DEFAULT_EROSION_RADIUS
    )]
    pub erode: Option<f32>,
    /// How erosion treats pixels outside the image bounds
    #[arg(
        long = "erode-border",
        value_enum,
        value_name = "MODE",
        requires = "erode"
    )]
    pub erode_border: Option<ErosionBorderArg>,
    /// Fill enclosed holes in the mask before vectorization (optionally override threshold value)
    #[arg(
        long = "fill-holes",
        value_name = "THRESHOLD",
        num_args = 0..=1,
        value_parser = parse_mask_threshold,
        default_missing_value = DEFAULT_MASK_THRESHOLD
    )]
    pub fill_holes: Option<u8>,
    #[arg(skip)]
    pub(crate) ordered_steps: Vec<CliMaskProcessingStep>,
}

impl MaskProcessingArgs {
    fn populate_ordered_steps(&mut self, matches: &ArgMatches) -> Result<(), clap::Error> {
        let mut entries = Vec::new();
        if let Some(sigma) = self.blur
            && let Some(index) = matches.index_of("blur")
        {
            entries.push((index, CliMaskProcessingStep::Blur(sigma)));
        }
        if let Some(value) = self.threshold
            && let Some(index) = matches.index_of("threshold")
        {
            entries.push((index, CliMaskProcessingStep::Threshold(value)));
        }
        if let Some(radius) = self.dilate
            && let Some(index) = matches.index_of("dilate")
        {
            entries.push((index, CliMaskProcessingStep::Dilate(radius)));
        }
        if let Some(radius) = self.erode
            && let Some(index) = matches.index_of("erode")
        {
            entries.push((
                index,
                CliMaskProcessingStep::Erode {
                    radius,
                    border_mode: self.erode_border.map(Into::into),
                },
            ));
        }
        if let Some(threshold) = self.fill_holes
            && let Some(index) = matches.index_of("fill_holes")
        {
            entries.push((index, CliMaskProcessingStep::FillHoles(threshold)));
        }

        entries.sort_by_key(|(index, _)| *index);
        let user_steps = entries.into_iter().map(|(_, step)| step).collect();
        self.ordered_steps = normalize_mask_steps(user_steps, self.no_implicit_threshold)?;
        Ok(())
    }
}

// Insert implicit threshold steps before hard-mask operations when needed, unless the user opts out.
fn normalize_mask_steps(
    user_steps: Vec<CliMaskProcessingStep>,
    no_implicit_threshold: bool,
) -> Result<Vec<CliMaskProcessingStep>, clap::Error> {
    let mut steps = Vec::with_capacity(user_steps.len());
    let mut mask_state = MaskState::Soft;

    for step in user_steps {
        let spec = step.spec();

        if spec.requires_hard_mask && mask_state == MaskState::Soft {
            if no_implicit_threshold {
                return Err(hard_mask_required_error(spec.option_name));
            }
            steps.push(CliMaskProcessingStep::Threshold(
                DEFAULT_MASK_THRESHOLD_VALUE,
            ));
        }

        steps.push(step);
        mask_state = spec.mask_state_after;
    }

    Ok(steps)
}

// Generate a clap error for when a hard-mask operation is requested without a preceding threshold.
fn hard_mask_required_error(operation: &'static str) -> clap::Error {
    clap::Error::raw(
        ErrorKind::ArgumentConflict,
        format!(
            "`--{operation}` requires a hard mask; add `--threshold` before it or remove `--no-implicit-threshold`"
        ),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskState {
    Soft, // Mask with continuous values (e.g., after blur)
    Hard, // Binary mask with only 0 and 255 values (e.g., after threshold)
}

// Specification for each mask processing step, used to determine when implicit thresholds are needed.
#[derive(Debug, Clone, Copy)]
struct MaskStepSpec {
    option_name: &'static str,
    requires_hard_mask: bool,
    mask_state_after: MaskState,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum CliMaskProcessingStep {
    Blur(f32),
    Threshold(u8),
    Dilate(f32),
    Erode {
        radius: f32,
        border_mode: Option<ErosionBorderMode>,
    },
    FillHoles(u8),
}

impl CliMaskProcessingStep {
    fn spec(self) -> MaskStepSpec {
        match self {
            Self::Blur(_) => MaskStepSpec {
                option_name: "blur",
                requires_hard_mask: false,
                mask_state_after: MaskState::Soft,
            },
            Self::Threshold(_) => MaskStepSpec {
                option_name: "threshold",
                requires_hard_mask: false,
                mask_state_after: MaskState::Hard,
            },
            Self::Dilate(_) => MaskStepSpec {
                option_name: "dilate",
                requires_hard_mask: true,
                mask_state_after: MaskState::Hard,
            },
            Self::Erode { .. } => MaskStepSpec {
                option_name: "erode",
                requires_hard_mask: true,
                mask_state_after: MaskState::Hard,
            },
            Self::FillHoles(_) => MaskStepSpec {
                option_name: "fill-holes",
                requires_hard_mask: true,
                mask_state_after: MaskState::Hard,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CliMaskProcessingRequest {
    steps: Vec<CliMaskProcessingStep>,
}

impl CliMaskProcessingRequest {
    pub(crate) fn from_args(args: &MaskProcessingArgs) -> Self {
        if args.ordered_steps.is_empty() {
            assert!(
                args.blur.is_none()
                    && args.threshold.is_none()
                    && args.dilate.is_none()
                    && args.erode.is_none()
                    && args.erode_border.is_none()
                    && args.fill_holes.is_none(),
                "MaskProcessingArgs must be populated through Cli::try_parse_from before conversion"
            );
        }

        Self {
            steps: args.ordered_steps.clone(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub(crate) fn to_pipeline(&self) -> MaskPipeline {
        let defaults = MaskProcessingDefaults::default();
        let mut pipeline = MaskPipeline::new();

        for step in &self.steps {
            pipeline = match *step {
                CliMaskProcessingStep::Blur(sigma) => pipeline.blur_with(sigma),
                CliMaskProcessingStep::Threshold(value) => pipeline.threshold_with(value),
                CliMaskProcessingStep::Dilate(radius) => pipeline.dilate_with(radius),
                CliMaskProcessingStep::Erode {
                    radius,
                    border_mode,
                } => pipeline.erode_with_border_mode(
                    radius,
                    border_mode.unwrap_or(defaults.erosion_border_mode),
                ),
                CliMaskProcessingStep::FillHoles(threshold) => pipeline.fill_holes_with(threshold),
            };
        }

        pipeline
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum ErosionBorderArg {
    OutsideIsBackground,
    OutsideIsUnknown,
}

impl From<ErosionBorderArg> for ErosionBorderMode {
    fn from(value: ErosionBorderArg) -> Self {
        match value {
            ErosionBorderArg::OutsideIsBackground => ErosionBorderMode::OutsideIsBackground,
            ErosionBorderArg::OutsideIsUnknown => ErosionBorderMode::OutsideIsUnknown,
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

fn parse_model_input_size(value: &str) -> Result<ModelInputSize, String> {
    let Some((height, width)) = value.split_once(['x', 'X']) else {
        return Err(format!(
            "model input size must be HEIGHTxWIDTH, got `{value}`"
        ));
    };

    let height = height
        .parse::<usize>()
        .map_err(|_| format!("model input height must be an integer, got `{height}`"))?;
    let width = width
        .parse::<usize>()
        .map_err(|_| format!("model input width must be an integer, got `{width}`"))?;

    if height == 0 || width == 0 {
        return Err(format!("model input size must be non-zero, got `{value}`"));
    }

    Ok(ModelInputSize::new(height, width))
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

#[cfg(test)]
mod tests {
    use super::*;

    mod parse_mask_threshold {
        use super::*;

        mod unit {
            use super::*;

            #[test]
            fn integer_values() {
                assert_eq!(parse_mask_threshold("0").unwrap(), 0);
                assert_eq!(parse_mask_threshold("128").unwrap(), 128);
                assert_eq!(parse_mask_threshold("255").unwrap(), 255);
            }

            #[test]
            fn float_zero_to_one_scaled() {
                assert_eq!(parse_mask_threshold("0.0").unwrap(), 0);
                assert_eq!(parse_mask_threshold("1.0").unwrap(), 255);
                assert_eq!(parse_mask_threshold("0.5").unwrap(), 128);
                // 0.25 * 255 = 63.75, rounds to 64
                assert_eq!(parse_mask_threshold("0.25").unwrap(), 64);
            }

            #[test]
            fn integer_as_float() {
                // "120.0" should be treated as integer 120
                assert_eq!(parse_mask_threshold("120.0").unwrap(), 120);
                assert_eq!(parse_mask_threshold("255.0").unwrap(), 255);
                assert_eq!(parse_mask_threshold("0.0").unwrap(), 0);
            }

            #[test]
            fn out_of_range_rejected() {
                assert!(parse_mask_threshold("256").is_err());
                assert!(parse_mask_threshold("-1").is_err());
                assert!(parse_mask_threshold("1.1").is_err());
                assert!(parse_mask_threshold("-0.1").is_err());
                assert!(parse_mask_threshold("255.5").is_err());
            }

            #[test]
            fn non_numeric_rejected() {
                assert!(parse_mask_threshold("abc").is_err());
                assert!(parse_mask_threshold("").is_err());
                assert!(parse_mask_threshold("12a").is_err());
            }

            #[test]
            fn integer_one_not_scaled() {
                // "1" parses as u8 first, so it stays 1 (not scaled to 255)
                assert_eq!(parse_mask_threshold("1").unwrap(), 1);
            }

            #[test]
            fn rounding_near_half() {
                // 0.499 * 255 = 127.245, rounds to 127
                assert_eq!(parse_mask_threshold("0.499").unwrap(), 127);
                // 0.501 * 255 = 127.755, rounds to 128
                assert_eq!(parse_mask_threshold("0.501").unwrap(), 128);
                // 0.5 * 255 = 127.5, rounds to 128 (round half up)
                assert_eq!(parse_mask_threshold("0.5").unwrap(), 128);
            }

            #[test]
            fn rounding_edge_cases() {
                // 0.002 * 255 = 0.51, rounds to 1
                assert_eq!(parse_mask_threshold("0.002").unwrap(), 1);
                // 0.001 * 255 = 0.255, rounds to 0
                assert_eq!(parse_mask_threshold("0.001").unwrap(), 0);
                // 0.998 * 255 = 254.49, rounds to 254
                assert_eq!(parse_mask_threshold("0.998").unwrap(), 254);
                // 0.999 * 255 = 254.745, rounds to 255
                assert_eq!(parse_mask_threshold("0.999").unwrap(), 255);
            }
        }

        mod prop {
            use super::*;
            use proptest::prelude::*;

            proptest! {
                #[test]
                fn float_0_to_1_always_parses_successfully(f in 0.0f32..=1.0f32) {
                    let s = format!("{:.6}", f);
                    let result = parse_mask_threshold(&s);
                    prop_assert!(result.is_ok());
                }

                #[test]
                fn valid_u8_always_parses_as_threshold(v in 0u8..=255u8) {
                    let s = v.to_string();
                    let result = parse_mask_threshold(&s).unwrap();
                    prop_assert_eq!(result, v);
                }
            }
        }
    }

    mod parse_model_input_size {
        use super::*;

        #[test]
        fn parses_taller_size() {
            let size = parse_model_input_size("1024x768").unwrap();
            assert_eq!(size.height(), 1024);
            assert_eq!(size.width(), 768);
        }

        #[test]
        fn parses_wider_size() {
            let size = parse_model_input_size("768x1024").unwrap();
            assert_eq!(size.height(), 768);
            assert_eq!(size.width(), 1024);
        }

        #[test]
        fn parses_uppercase_separator() {
            let size = parse_model_input_size("768X1024").unwrap();
            assert_eq!(size.height(), 768);
            assert_eq!(size.width(), 1024);
        }

        #[test]
        fn rejects_missing_separator() {
            assert!(parse_model_input_size("1024").is_err());
        }

        #[test]
        fn rejects_zero_dimension() {
            assert!(parse_model_input_size("0x1024").is_err());
            assert!(parse_model_input_size("1024x0").is_err());
        }
    }

    mod from_implementations {
        use super::*;

        mod unit {
            use super::*;

            #[test]
            fn resample_filter_to_filter_type() {
                assert!(matches!(
                    FilterType::from(ResampleFilter::Nearest),
                    FilterType::Nearest
                ));
                assert!(matches!(
                    FilterType::from(ResampleFilter::Triangle),
                    FilterType::Triangle
                ));
                assert!(matches!(
                    FilterType::from(ResampleFilter::CatmullRom),
                    FilterType::CatmullRom
                ));
                assert!(matches!(
                    FilterType::from(ResampleFilter::Gaussian),
                    FilterType::Gaussian
                ));
                assert!(matches!(
                    FilterType::from(ResampleFilter::Lanczos3),
                    FilterType::Lanczos3
                ));
            }

            #[test]
            fn tracer_color_mode_to_color_mode() {
                assert!(matches!(
                    ColorMode::from(TracerColorMode::Color),
                    ColorMode::Color
                ));
                assert!(matches!(
                    ColorMode::from(TracerColorMode::Binary),
                    ColorMode::Binary
                ));
            }

            #[test]
            fn tracer_hierarchy_to_hierarchical() {
                assert!(matches!(
                    Hierarchical::from(TracerHierarchy::Stacked),
                    Hierarchical::Stacked
                ));
                assert!(matches!(
                    Hierarchical::from(TracerHierarchy::Cutout),
                    Hierarchical::Cutout
                ));
            }

            #[test]
            fn tracer_mode_to_path_simplify_mode() {
                assert!(matches!(
                    PathSimplifyMode::from(TracerMode::None),
                    PathSimplifyMode::None
                ));
                assert!(matches!(
                    PathSimplifyMode::from(TracerMode::Polygon),
                    PathSimplifyMode::Polygon
                ));
                assert!(matches!(
                    PathSimplifyMode::from(TracerMode::Spline),
                    PathSimplifyMode::Spline
                ));
            }
        }
    }

    mod mask_processing_args_conversion {
        use super::*;
        use outline::MaskOperation;

        fn default_args() -> MaskProcessingArgs {
            MaskProcessingArgs {
                blur: None,
                threshold: None,
                no_implicit_threshold: false,
                dilate: None,
                erode: None,
                erode_border: None,
                fill_holes: None,
                ordered_steps: vec![],
            }
        }

        fn request(args: &MaskProcessingArgs) -> CliMaskProcessingRequest {
            CliMaskProcessingRequest::from_args(args)
        }

        fn pipeline(args: &MaskProcessingArgs) -> MaskPipeline {
            request(args).to_pipeline()
        }

        mod unit {
            use super::*;

            #[test]
            fn cli_default_missing_values_match_processing_defaults() {
                let defaults = MaskProcessingDefaults::default();

                assert_eq!(
                    DEFAULT_BLUR_SIGMA.parse::<f32>().unwrap(),
                    defaults.blur_sigma
                );
                assert_eq!(
                    parse_mask_threshold(DEFAULT_MASK_THRESHOLD).unwrap(),
                    defaults.mask_threshold
                );
                assert_eq!(DEFAULT_MASK_THRESHOLD_VALUE, defaults.mask_threshold);
                assert_eq!(
                    DEFAULT_DILATION_RADIUS.parse::<f32>().unwrap(),
                    defaults.dilation_radius
                );
                assert_eq!(
                    DEFAULT_EROSION_RADIUS.parse::<f32>().unwrap(),
                    defaults.erosion_radius
                );
            }

            #[test]
            fn auto_without_hard_mask_ops_yields_empty_pipeline() {
                let args = default_args();
                let pipeline = pipeline(&args);

                assert!(pipeline.is_empty());
            }

            #[test]
            fn threshold_step_materializes_threshold() {
                let args = MaskProcessingArgs {
                    ordered_steps: vec![CliMaskProcessingStep::Threshold(200)],
                    ..default_args()
                };
                let pipeline = pipeline(&args);

                assert!(matches!(
                    pipeline.operations(),
                    [MaskOperation::Threshold { value: 200 }]
                ));
            }

            #[test]
            fn threshold_then_fill_holes_materializes_in_order() {
                let args = MaskProcessingArgs {
                    ordered_steps: vec![
                        CliMaskProcessingStep::Threshold(120),
                        CliMaskProcessingStep::FillHoles(120),
                    ],
                    ..default_args()
                };
                let pipeline = pipeline(&args);

                assert!(matches!(
                    pipeline.operations(),
                    [
                        MaskOperation::Threshold { value: 120 },
                        MaskOperation::FillHoles { threshold: 120 }
                    ]
                ));
            }

            #[test]
            fn threshold_then_dilate_materializes_in_order() {
                let args = MaskProcessingArgs {
                    ordered_steps: vec![
                        CliMaskProcessingStep::Threshold(120),
                        CliMaskProcessingStep::Dilate(5.0),
                    ],
                    ..default_args()
                };
                let pipeline = pipeline(&args);

                assert!(matches!(
                    pipeline.operations(),
                    [
                        MaskOperation::Threshold { value: 120 },
                        MaskOperation::Dilate { radius }
                    ] if (*radius - 5.0).abs() < f32::EPSILON
                ));
            }

            #[test]
            fn threshold_then_erode_materializes_in_order() {
                let args = MaskProcessingArgs {
                    ordered_steps: vec![
                        CliMaskProcessingStep::Threshold(120),
                        CliMaskProcessingStep::Erode {
                            radius: 5.0,
                            border_mode: None,
                        },
                    ],
                    ..default_args()
                };
                let pipeline = pipeline(&args);

                assert!(matches!(
                    pipeline.operations(),
                    [
                        MaskOperation::Threshold { value: 120 },
                        MaskOperation::Erode { radius, border_mode }
                    ] if (*radius - 5.0).abs() < f32::EPSILON
                        && *border_mode == ErosionBorderMode::OutsideIsBackground
                ));
            }

            #[test]
            fn fill_holes_without_threshold_materializes_as_requested() {
                let args = MaskProcessingArgs {
                    ordered_steps: vec![CliMaskProcessingStep::FillHoles(120)],
                    ..default_args()
                };
                let pipeline = pipeline(&args);

                assert!(matches!(
                    pipeline.operations(),
                    [MaskOperation::FillHoles { threshold: 120 }]
                ));
            }

            #[test]
            fn threshold_only_materializes_as_requested() {
                let args = MaskProcessingArgs {
                    ordered_steps: vec![CliMaskProcessingStep::Threshold(120)],
                    ..default_args()
                };
                let pipeline = pipeline(&args);

                assert!(matches!(
                    pipeline.operations(),
                    [MaskOperation::Threshold { value: 120 }]
                ));
            }

            #[test]
            fn blur_request_adds_pipeline_sigma() {
                let args = MaskProcessingArgs {
                    ordered_steps: vec![CliMaskProcessingStep::Blur(10.0)],
                    ..default_args()
                };
                let pipeline = pipeline(&args);

                assert!(matches!(
                    pipeline.operations(),
                    [MaskOperation::Blur { sigma }] if (*sigma - 10.0).abs() < f32::EPSILON
                ));
            }

            #[test]
            fn blur_flag_only_uses_default_sigma_when_materialized() {
                let args = MaskProcessingArgs {
                    ordered_steps: vec![CliMaskProcessingStep::Blur(6.0)],
                    ..default_args()
                };
                let pipeline = pipeline(&args);

                assert!(matches!(
                    pipeline.operations(),
                    [MaskOperation::Blur { sigma }] if (*sigma - 6.0).abs() < f32::EPSILON
                ));
            }

            #[test]
            fn dilate_request_adds_threshold_and_radius() {
                let args = MaskProcessingArgs {
                    ordered_steps: vec![
                        CliMaskProcessingStep::Threshold(120),
                        CliMaskProcessingStep::Dilate(8.0),
                    ],
                    ..default_args()
                };
                let pipeline = pipeline(&args);

                assert!(matches!(
                    pipeline.operations(),
                    [
                        MaskOperation::Threshold { value: 120 },
                        MaskOperation::Dilate { radius }
                    ] if (*radius - 8.0).abs() < f32::EPSILON
                ));
            }

            #[test]
            fn dilate_flag_only_uses_default_radius_when_materialized() {
                let args = MaskProcessingArgs {
                    ordered_steps: vec![
                        CliMaskProcessingStep::Threshold(120),
                        CliMaskProcessingStep::Dilate(5.0),
                    ],
                    ..default_args()
                };
                let pipeline = pipeline(&args);

                assert!(matches!(
                    pipeline.operations(),
                    [
                        MaskOperation::Threshold { value: 120 },
                        MaskOperation::Dilate { radius }
                    ] if (*radius - 5.0).abs() < f32::EPSILON
                ));
            }

            #[test]
            fn erode_request_adds_threshold_and_radius() {
                let args = MaskProcessingArgs {
                    ordered_steps: vec![
                        CliMaskProcessingStep::Threshold(120),
                        CliMaskProcessingStep::Erode {
                            radius: 3.0,
                            border_mode: None,
                        },
                    ],
                    ..default_args()
                };
                let pipeline = pipeline(&args);

                assert!(matches!(
                    pipeline.operations(),
                    [
                        MaskOperation::Threshold { value: 120 },
                        MaskOperation::Erode { radius, border_mode }
                    ] if (*radius - 3.0).abs() < f32::EPSILON
                        && *border_mode == ErosionBorderMode::default()
                ));
            }

            #[test]
            fn erode_flag_only_uses_default_radius_when_materialized() {
                let args = MaskProcessingArgs {
                    ordered_steps: vec![
                        CliMaskProcessingStep::Threshold(120),
                        CliMaskProcessingStep::Erode {
                            radius: 5.0,
                            border_mode: None,
                        },
                    ],
                    ..default_args()
                };
                let pipeline = pipeline(&args);

                assert!(matches!(
                    pipeline.operations(),
                    [
                        MaskOperation::Threshold { value: 120 },
                        MaskOperation::Erode { radius, border_mode }
                    ] if (*radius - 5.0).abs() < f32::EPSILON
                        && *border_mode == ErosionBorderMode::default()
                ));
            }

            #[test]
            fn erode_border_passed_through() {
                let args = MaskProcessingArgs {
                    ordered_steps: vec![
                        CliMaskProcessingStep::Threshold(120),
                        CliMaskProcessingStep::Erode {
                            radius: 3.0,
                            border_mode: Some(ErosionBorderMode::OutsideIsUnknown),
                        },
                    ],
                    ..default_args()
                };
                let pipeline = pipeline(&args);

                assert!(matches!(
                    pipeline.operations(),
                    [
                        MaskOperation::Threshold { value: 120 },
                        MaskOperation::Erode { radius, border_mode }
                    ] if (*radius - 3.0).abs() < f32::EPSILON
                        && *border_mode == ErosionBorderMode::OutsideIsUnknown
                ));
            }

            #[test]
            fn threshold_passed_through() {
                let args = MaskProcessingArgs {
                    ordered_steps: vec![CliMaskProcessingStep::Threshold(200)],
                    ..default_args()
                };
                let pipeline = pipeline(&args);

                assert!(matches!(
                    pipeline.operations(),
                    [MaskOperation::Threshold { value: 200 }]
                ));
            }

            #[test]
            #[should_panic(expected = "MaskProcessingArgs must be populated")]
            fn raw_operation_fields_fail_fast() {
                let args = MaskProcessingArgs {
                    blur: Some(10.0),
                    ..default_args()
                };

                let _ = pipeline(&args);
            }
        }
    }

    mod trace_options_args_conversion {
        use super::*;

        fn default_trace_args() -> TraceOptionsArgs {
            TraceOptionsArgs {
                color_mode: TracerColorMode::Binary,
                hierarchy: TracerHierarchy::Stacked,
                mode: TracerMode::Spline,
                filter_speckle: 4,
                color_precision: 6,
                layer_difference: 16,
                corner_threshold: 60,
                length_threshold: 4.0,
                max_iterations: 10,
                splice_threshold: 45,
                path_precision: None,
                no_path_precision: false,
                invert_svg: false,
            }
        }

        mod unit {
            use super::*;

            #[test]
            fn no_path_precision_clears_default() {
                let args = TraceOptionsArgs {
                    no_path_precision: true,
                    ..default_trace_args()
                };
                let opts = TraceOptions::from(&args);
                assert!(opts.tracer_path_precision.is_none());
            }

            #[test]
            fn path_precision_overrides_default() {
                let args = TraceOptionsArgs {
                    path_precision: Some(5),
                    ..default_trace_args()
                };
                let opts = TraceOptions::from(&args);
                assert_eq!(opts.tracer_path_precision, Some(5));
            }

            #[test]
            fn default_path_precision_used() {
                let args = default_trace_args();
                let opts = TraceOptions::from(&args);
                let default_opts = TraceOptions::default();
                assert_eq!(
                    opts.tracer_path_precision,
                    default_opts.tracer_path_precision
                );
            }

            #[test]
            fn invert_svg_passed_through() {
                let args = TraceOptionsArgs {
                    invert_svg: true,
                    ..default_trace_args()
                };
                let opts = TraceOptions::from(&args);
                assert!(opts.invert_svg);
            }

            #[test]
            fn enum_fields_converted() {
                let args = TraceOptionsArgs {
                    color_mode: TracerColorMode::Color,
                    hierarchy: TracerHierarchy::Cutout,
                    mode: TracerMode::Polygon,
                    ..default_trace_args()
                };
                let opts = TraceOptions::from(&args);
                assert!(matches!(opts.tracer_color_mode, ColorMode::Color));
                assert!(matches!(opts.tracer_hierarchical, Hierarchical::Cutout));
                assert!(matches!(opts.tracer_mode, PathSimplifyMode::Polygon));
            }

            #[test]
            fn conflicting_no_path_precision_and_path_precision() {
                // clap prevents this via conflicts_with, but test pure function priority
                let args = TraceOptionsArgs {
                    no_path_precision: true,
                    path_precision: Some(5),
                    ..default_trace_args()
                };
                let opts = TraceOptions::from(&args);
                // no_path_precision takes priority
                assert!(opts.tracer_path_precision.is_none());
            }
        }
    }

    mod clap_integration {
        use super::*;
        use std::path::Path;

        macro_rules! parse_cmd {
            ($args:expr, $variant:ident) => {{
                let cli = Cli::try_parse_from($args).unwrap();
                match cli.command {
                    Commands::$variant(cmd) => cmd,
                    _ => panic!("expected {} command", stringify!($variant)),
                }
            }};
        }

        // Option<Option<PathBuf>> three-state semantics
        mod optional_path_semantics {
            use super::*;

            mod unit {
                use super::*;

                #[test]
                fn export_matte_absent_is_none() {
                    let cmd = parse_cmd!(["outline", "cut", "in.png"], Cut);
                    assert!(cmd.export_matte.is_none());
                }

                #[test]
                fn export_matte_flag_only_is_some_none() {
                    let cmd = parse_cmd!(["outline", "cut", "in.png", "--export-matte"], Cut);
                    assert!(matches!(cmd.export_matte, Some(None)));
                }

                #[test]
                fn export_matte_with_path_is_some_some() {
                    let cmd = parse_cmd!(
                        ["outline", "cut", "in.png", "--export-matte", "out.png"],
                        Cut
                    );
                    assert!(
                        matches!(&cmd.export_matte, Some(Some(p)) if p == Path::new("out.png"))
                    );
                }

                #[test]
                fn export_mask_absent_is_none() {
                    let cmd = parse_cmd!(["outline", "cut", "in.png"], Cut);
                    assert!(cmd.export_mask.is_none());
                }

                #[test]
                fn export_mask_flag_only_is_some_none() {
                    let cmd = parse_cmd!(["outline", "cut", "in.png", "--export-mask"], Cut);
                    assert!(matches!(cmd.export_mask, Some(None)));
                }

                #[test]
                fn export_mask_with_path_is_some_some() {
                    let cmd = parse_cmd!(
                        ["outline", "cut", "in.png", "--export-mask", "mask.png"],
                        Cut
                    );
                    assert!(
                        matches!(&cmd.export_mask, Some(Some(p)) if p == Path::new("mask.png"))
                    );
                }
            }
        }

        // Optional argument value behavior
        mod optional_argument_values {
            use super::*;

            mod unit {
                use super::*;

                #[test]
                fn blur_absent_is_none() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png"], Mask);
                    assert_eq!(cmd.mask_processing.blur, None);
                }

                #[test]
                fn blur_flag_only_records_default_request() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--blur"], Mask);
                    assert_eq!(cmd.mask_processing.blur, Some(6.0));
                }

                #[test]
                fn blur_with_value_records_explicit_value() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--blur", "10.0"], Mask);
                    assert_eq!(cmd.mask_processing.blur, Some(10.0));
                }

                #[test]
                fn threshold_absent_is_none() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png"], Mask);
                    assert_eq!(cmd.mask_processing.threshold, None);
                }

                #[test]
                fn threshold_flag_only_records_default_request() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--threshold"], Mask);
                    assert_eq!(cmd.mask_processing.threshold, Some(120));
                }

                #[test]
                fn threshold_with_value_records_explicit_value() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--threshold", "200"], Mask);
                    assert_eq!(cmd.mask_processing.threshold, Some(200));
                }

                #[test]
                fn no_implicit_threshold_flag_is_recorded() {
                    let cmd = parse_cmd!(
                        ["outline", "mask", "in.png", "--no-implicit-threshold"],
                        Mask
                    );
                    assert!(cmd.mask_processing.no_implicit_threshold);
                }

                #[test]
                fn dilate_absent_is_none() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png"], Mask);
                    assert_eq!(cmd.mask_processing.dilate, None);
                }

                #[test]
                fn dilate_flag_only_records_default_request() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--dilate"], Mask);
                    assert_eq!(cmd.mask_processing.dilate, Some(5.0));
                }

                #[test]
                fn dilate_with_value_records_explicit_value() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--dilate", "8.0"], Mask);
                    assert_eq!(cmd.mask_processing.dilate, Some(8.0));
                }

                #[test]
                fn erode_absent_is_none() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png"], Mask);
                    assert_eq!(cmd.mask_processing.erode, None);
                }

                #[test]
                fn erode_flag_only_records_default_request() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--erode"], Mask);
                    assert_eq!(cmd.mask_processing.erode, Some(5.0));
                }

                #[test]
                fn erode_with_value_records_explicit_value() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--erode", "3.0"], Mask);
                    assert_eq!(cmd.mask_processing.erode, Some(3.0));
                }

                #[test]
                fn fill_holes_absent_is_none() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png"], Mask);
                    assert_eq!(cmd.mask_processing.fill_holes, None);
                }

                #[test]
                fn fill_holes_flag_only_records_default_request() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--fill-holes"], Mask);
                    assert_eq!(cmd.mask_processing.fill_holes, Some(120));
                }

                #[test]
                fn fill_holes_with_value_records_explicit_value() {
                    let cmd =
                        parse_cmd!(["outline", "mask", "in.png", "--fill-holes", "180"], Mask);
                    assert_eq!(cmd.mask_processing.fill_holes, Some(180));
                }

                #[test]
                fn erode_border_outside_is_unknown_parses() {
                    let cmd = parse_cmd!(
                        [
                            "outline",
                            "mask",
                            "in.png",
                            "--erode",
                            "--erode-border",
                            "outside-is-unknown"
                        ],
                        Mask
                    );
                    assert_eq!(
                        cmd.mask_processing.erode_border,
                        Some(ErosionBorderArg::OutsideIsUnknown)
                    );
                }

                #[test]
                fn erode_border_requires_erode() {
                    let result = Cli::try_parse_from([
                        "outline",
                        "mask",
                        "in.png",
                        "--erode-border",
                        "outside-is-unknown",
                    ]);
                    assert!(result.is_err());
                }
            }
        }

        mod ordered_mask_processing {
            use super::*;
            use outline::MaskOperation;

            fn pipeline(args: &MaskProcessingArgs) -> MaskPipeline {
                CliMaskProcessingRequest::from_args(args).to_pipeline()
            }

            mod unit {
                use super::*;

                #[test]
                fn fixed_flags_preserve_cross_operation_order() {
                    let cmd = parse_cmd!(
                        [
                            "outline", "mask", "in.png", "--blur", "2.0", "--dilate", "5.0",
                            "--erode", "1.0"
                        ],
                        Mask
                    );
                    let pipeline = pipeline(&cmd.mask_processing);

                    assert!(matches!(
                        pipeline.operations(),
                        [
                            MaskOperation::Blur { sigma: first_sigma },
                            MaskOperation::Threshold { value: 120 },
                            MaskOperation::Dilate { radius },
                            MaskOperation::Erode { radius: erode_radius, border_mode },
                        ] if (*first_sigma - 2.0).abs() < f32::EPSILON
                            && (*radius - 5.0).abs() < f32::EPSILON
                            && (*erode_radius - 1.0).abs() < f32::EPSILON
                            && *border_mode == ErosionBorderMode::default()
                    ));
                }

                #[test]
                fn repeated_operation_flag_is_rejected() {
                    let result = Cli::try_parse_from([
                        "outline", "mask", "in.png", "--blur", "2.0", "--blur", "8.0",
                    ]);

                    assert!(result.is_err());
                }

                #[test]
                fn flag_only_blur_is_preserved_in_ordered_pipeline() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--blur"], Mask);
                    let pipeline = pipeline(&cmd.mask_processing);

                    assert!(matches!(
                        pipeline.operations(),
                        [MaskOperation::Blur { sigma }] if (*sigma - 6.0).abs() < f32::EPSILON
                    ));
                }

                #[test]
                fn flag_only_dilate_is_preserved_in_ordered_pipeline() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--dilate"], Mask);
                    let pipeline = pipeline(&cmd.mask_processing);

                    assert!(matches!(
                        pipeline.operations(),
                        [
                            MaskOperation::Threshold { value: 120 },
                            MaskOperation::Dilate { radius },
                        ] if (*radius - 5.0).abs() < f32::EPSILON
                    ));
                }

                #[test]
                fn flag_only_erode_is_preserved_in_ordered_pipeline() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--erode"], Mask);
                    let pipeline = pipeline(&cmd.mask_processing);

                    assert!(matches!(
                        pipeline.operations(),
                        [
                            MaskOperation::Threshold { value: 120 },
                            MaskOperation::Erode { radius, border_mode },
                        ] if (*radius - 5.0).abs() < f32::EPSILON
                            && *border_mode == ErosionBorderMode::default()
                    ));
                }

                #[test]
                fn flag_only_fill_holes_is_preserved_in_ordered_pipeline() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--fill-holes"], Mask);
                    let pipeline = pipeline(&cmd.mask_processing);

                    assert!(matches!(
                        pipeline.operations(),
                        [
                            MaskOperation::Threshold { value: 120 },
                            MaskOperation::FillHoles { threshold: 120 },
                        ]
                    ));
                }

                #[test]
                fn fill_holes_threshold_is_passed_through() {
                    let cmd = parse_cmd!(
                        [
                            "outline",
                            "mask",
                            "in.png",
                            "--threshold",
                            "--fill-holes",
                            "180"
                        ],
                        Mask
                    );
                    let pipeline = pipeline(&cmd.mask_processing);

                    assert!(matches!(
                        pipeline.operations(),
                        [
                            MaskOperation::Threshold { value: 120 },
                            MaskOperation::FillHoles { threshold: 180 },
                        ]
                    ));
                }

                #[test]
                fn auto_threshold_is_inserted_before_first_hard_mask_operation() {
                    let cmd = parse_cmd!(
                        [
                            "outline", "mask", "in.png", "--dilate", "5.0", "--blur", "2.0"
                        ],
                        Mask
                    );
                    let pipeline = pipeline(&cmd.mask_processing);

                    assert!(matches!(
                        pipeline.operations(),
                        [
                            MaskOperation::Threshold { value: 120 },
                            MaskOperation::Dilate { radius },
                            MaskOperation::Blur { sigma },
                        ] if (*radius - 5.0).abs() < f32::EPSILON
                            && (*sigma - 2.0).abs() < f32::EPSILON
                    ));
                }

                #[test]
                fn threshold_alone_materializes_threshold() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--threshold", "200"], Mask);
                    let pipeline = pipeline(&cmd.mask_processing);

                    assert!(matches!(
                        pipeline.operations(),
                        [MaskOperation::Threshold { value: 200 }]
                    ));
                }

                #[test]
                fn threshold_prevents_duplicate_implicit_threshold() {
                    let cmd = parse_cmd!(
                        [
                            "outline",
                            "mask",
                            "in.png",
                            "--threshold",
                            "200",
                            "--dilate",
                            "5.0"
                        ],
                        Mask
                    );
                    let pipeline = pipeline(&cmd.mask_processing);

                    assert!(matches!(
                        pipeline.operations(),
                        [
                            MaskOperation::Threshold { value: 200 },
                            MaskOperation::Dilate { radius },
                        ] if (*radius - 5.0).abs() < f32::EPSILON
                    ));
                }

                #[test]
                fn late_threshold_does_not_prevent_earlier_implicit_threshold() {
                    let cmd = parse_cmd!(
                        [
                            "outline",
                            "mask",
                            "in.png",
                            "--dilate",
                            "5.0",
                            "--threshold",
                            "200"
                        ],
                        Mask
                    );
                    let pipeline = pipeline(&cmd.mask_processing);

                    assert!(matches!(
                        pipeline.operations(),
                        [
                            MaskOperation::Threshold { value: 120 },
                            MaskOperation::Dilate { radius },
                            MaskOperation::Threshold { value: 200 },
                        ] if (*radius - 5.0).abs() < f32::EPSILON
                    ));
                }

                #[test]
                fn no_implicit_threshold_rejects_hard_mask_operation_without_threshold() {
                    let result = Cli::try_parse_from([
                        "outline",
                        "mask",
                        "in.png",
                        "--no-implicit-threshold",
                        "--dilate",
                        "5.0",
                    ]);

                    assert!(result.is_err());
                }

                #[test]
                fn no_implicit_threshold_accepts_explicit_threshold_before_hard_mask_operation() {
                    let cmd = parse_cmd!(
                        [
                            "outline",
                            "mask",
                            "in.png",
                            "--no-implicit-threshold",
                            "--threshold",
                            "--dilate",
                            "5.0"
                        ],
                        Mask
                    );
                    let pipeline = pipeline(&cmd.mask_processing);

                    assert!(matches!(
                        pipeline.operations(),
                        [
                            MaskOperation::Threshold { value: 120 },
                            MaskOperation::Dilate { radius },
                        ] if (*radius - 5.0).abs() < f32::EPSILON
                    ));
                }

                #[test]
                fn no_implicit_threshold_rejects_hard_mask_operation_after_blur() {
                    let result = Cli::try_parse_from([
                        "outline",
                        "mask",
                        "in.png",
                        "--no-implicit-threshold",
                        "--threshold",
                        "--blur",
                        "2.0",
                        "--dilate",
                        "5.0",
                    ]);

                    assert!(result.is_err());
                }

                #[test]
                fn explicit_threshold_uses_flag_position() {
                    let cmd = parse_cmd!(
                        [
                            "outline",
                            "mask",
                            "in.png",
                            "--blur",
                            "2.0",
                            "--threshold",
                            "--dilate",
                            "5.0"
                        ],
                        Mask
                    );
                    let pipeline = pipeline(&cmd.mask_processing);

                    assert!(matches!(
                        pipeline.operations(),
                        [
                            MaskOperation::Blur { sigma },
                            MaskOperation::Threshold { value: 120 },
                            MaskOperation::Dilate { radius },
                        ] if (*sigma - 2.0).abs() < f32::EPSILON
                            && (*radius - 5.0).abs() < f32::EPSILON
                    ));
                }

                #[test]
                fn blur_after_threshold_softens_mask_and_triggers_later_implicit_threshold() {
                    let cmd = parse_cmd!(
                        [
                            "outline",
                            "mask",
                            "in.png",
                            "--threshold",
                            "--blur",
                            "2.0",
                            "--dilate",
                            "5.0"
                        ],
                        Mask
                    );
                    let pipeline = pipeline(&cmd.mask_processing);

                    assert!(matches!(
                        pipeline.operations(),
                        [
                            MaskOperation::Threshold { value: 120 },
                            MaskOperation::Blur { sigma },
                            MaskOperation::Threshold { value: 120 },
                            MaskOperation::Dilate { radius },
                        ] if (*sigma - 2.0).abs() < f32::EPSILON
                            && (*radius - 5.0).abs() < f32::EPSILON
                    ));
                }

                #[test]
                fn mask_pipeline_string_is_not_supported() {
                    let result = Cli::try_parse_from([
                        "outline",
                        "mask",
                        "in.png",
                        "--mask-pipeline",
                        "blur 2; dilate 5",
                    ]);

                    assert!(result.is_err());
                }
            }
        }

        // conflicts_with behavior
        mod conflicts_with {
            use super::*;

            mod unit {
                use super::*;

                #[test]
                fn path_precision_and_no_path_precision_conflict() {
                    let result = Cli::try_parse_from([
                        "outline",
                        "trace",
                        "in.png",
                        "--path-precision",
                        "5",
                        "--no-path-precision",
                    ]);
                    assert!(result.is_err());
                }

                #[test]
                fn path_precision_alone_ok() {
                    let cmd = parse_cmd!(
                        ["outline", "trace", "in.png", "--path-precision", "5"],
                        Trace
                    );
                    assert_eq!(cmd.trace_options.path_precision, Some(5));
                    assert!(!cmd.trace_options.no_path_precision);
                }

                #[test]
                fn no_path_precision_alone_ok() {
                    let cmd =
                        parse_cmd!(["outline", "trace", "in.png", "--no-path-precision"], Trace);
                    assert!(cmd.trace_options.no_path_precision);
                    assert!(cmd.trace_options.path_precision.is_none());
                }
            }
        }

        // Threshold value_parser
        mod threshold_parsing {
            use super::*;

            mod unit {
                use super::*;

                #[test]
                fn integer_threshold() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--threshold", "200"], Mask);
                    assert_eq!(cmd.mask_processing.threshold, Some(200));
                }

                #[test]
                fn float_threshold_scaled() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--threshold", "0.5"], Mask);
                    assert_eq!(cmd.mask_processing.threshold, Some(128));
                }

                #[test]
                fn flag_only_threshold_uses_default_when_materialized() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--threshold"], Mask);
                    let pipeline =
                        CliMaskProcessingRequest::from_args(&cmd.mask_processing).to_pipeline();

                    assert!(matches!(
                        pipeline.operations(),
                        [outline::MaskOperation::Threshold { value: 120 }]
                    ));
                }

                #[test]
                fn invalid_threshold_rejected() {
                    let result =
                        Cli::try_parse_from(["outline", "mask", "in.png", "--threshold", "1.5"]);
                    assert!(result.is_err());
                }
            }
        }

        // ValueEnum parsing for global options.
        mod value_enum_parsing {
            use super::*;

            mod unit {
                use super::*;

                #[test]
                fn input_resample_filter_gaussian() {
                    let cli = Cli::try_parse_from([
                        "outline",
                        "mask",
                        "in.png",
                        "--input-resample-filter",
                        "gaussian",
                    ])
                    .unwrap();
                    assert!(matches!(
                        cli.global.input_resample_filter,
                        ResampleFilter::Gaussian
                    ));
                }

                #[test]
                fn output_resample_filter_nearest() {
                    let cli = Cli::try_parse_from([
                        "outline",
                        "mask",
                        "in.png",
                        "--output-resample-filter",
                        "nearest",
                    ])
                    .unwrap();
                    assert!(matches!(
                        cli.global.output_resample_filter,
                        ResampleFilter::Nearest
                    ));
                }

                #[test]
                fn model_input_size_override() {
                    let cli = Cli::try_parse_from([
                        "outline",
                        "mask",
                        "in.png",
                        "--model-input-size",
                        "1024x768",
                    ])
                    .unwrap();
                    let size = cli.global.model_input_size.unwrap();
                    assert_eq!(size.height(), 1024);
                    assert_eq!(size.width(), 768);
                }

                #[test]
                fn resample_filter_all_variants() {
                    for (name, expected) in [
                        ("nearest", ResampleFilter::Nearest),
                        ("triangle", ResampleFilter::Triangle),
                        ("catmull-rom", ResampleFilter::CatmullRom),
                        ("gaussian", ResampleFilter::Gaussian),
                        ("lanczos3", ResampleFilter::Lanczos3),
                    ] {
                        let cli = Cli::try_parse_from([
                            "outline",
                            "mask",
                            "in.png",
                            "--input-resample-filter",
                            name,
                        ])
                        .unwrap();
                        assert!(
                            matches!(cli.global.input_resample_filter, ref f if std::mem::discriminant(f) == std::mem::discriminant(&expected)),
                            "failed for {name}"
                        );
                    }
                }

                #[test]
                fn invalid_resample_filter_rejected() {
                    let result = Cli::try_parse_from([
                        "outline",
                        "mask",
                        "in.png",
                        "--input-resample-filter",
                        "invalid",
                    ]);
                    assert!(result.is_err());
                }
            }
        }

        /// Integration tests for model path resolution priority through clap.
        ///
        /// Priority: --model flag > OUTLINE_MODEL_PATH env var > (downstream: cached > default)
        mod model_path_priority {
            use super::*;
            use std::sync::Mutex;

            // Serialize env-var tests so they don't race each other.
            static ENV_LOCK: Mutex<()> = Mutex::new(());

            #[test]
            fn flag_sets_model() {
                let cli = Cli::try_parse_from(["outline", "-m", "flag.onnx", "mask", "input.png"])
                    .unwrap();
                assert_eq!(cli.global.model, Some(PathBuf::from("flag.onnx")));
            }

            #[test]
            fn env_var_sets_model() {
                let _lock = ENV_LOCK.lock().unwrap();
                // SAFETY: serialized by ENV_LOCK; no other thread reads this var concurrently.
                unsafe { std::env::set_var(outline::ENV_MODEL_PATH, "env.onnx") };
                let cli = Cli::try_parse_from(["outline", "mask", "input.png"]).unwrap();
                unsafe { std::env::remove_var(outline::ENV_MODEL_PATH) };
                assert_eq!(cli.global.model, Some(PathBuf::from("env.onnx")));
            }

            #[test]
            fn flag_overrides_env_var() {
                let _lock = ENV_LOCK.lock().unwrap();
                // SAFETY: serialized by ENV_LOCK; no other thread reads this var concurrently.
                unsafe { std::env::set_var(outline::ENV_MODEL_PATH, "env.onnx") };
                let cli = Cli::try_parse_from(["outline", "-m", "flag.onnx", "mask", "input.png"])
                    .unwrap();
                unsafe { std::env::remove_var(outline::ENV_MODEL_PATH) };
                assert_eq!(cli.global.model, Some(PathBuf::from("flag.onnx")));
            }

            #[test]
            fn neither_flag_nor_env_gives_none() {
                let _lock = ENV_LOCK.lock().unwrap();
                // SAFETY: serialized by ENV_LOCK; no other thread reads this var concurrently.
                unsafe { std::env::remove_var(outline::ENV_MODEL_PATH) };
                let cli = Cli::try_parse_from(["outline", "mask", "input.png"]).unwrap();
                assert_eq!(cli.global.model, None);
            }
        }
    }
}
