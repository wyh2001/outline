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
        value_hint = ValueHint::FilePath
    )]
    pub model: Option<PathBuf>,
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
    /// Download the default model from the network
    #[cfg(feature = "model-fetch")]
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
#[cfg(feature = "model-fetch")]
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

        fn default_args() -> MaskProcessingArgs {
            MaskProcessingArgs {
                blur: None,
                mask_threshold: 120,
                binary: BinaryOption::Auto,
                dilate: None,
                fill_holes: false,
            }
        }

        mod unit {
            use super::*;

            #[test]
            fn auto_no_dilate_no_fill_holes_yields_binary_false() {
                let args = default_args();
                let opts = MaskProcessingOptions::from(&args);
                assert!(!opts.binary);
            }

            #[test]
            fn auto_with_fill_holes_yields_binary_true() {
                let args = MaskProcessingArgs {
                    fill_holes: true,
                    ..default_args()
                };
                let opts = MaskProcessingOptions::from(&args);
                assert!(opts.binary);
            }

            #[test]
            fn auto_with_dilate_yields_binary_true() {
                let args = MaskProcessingArgs {
                    dilate: Some(5.0),
                    ..default_args()
                };
                let opts = MaskProcessingOptions::from(&args);
                assert!(opts.binary);
            }

            #[test]
            fn disabled_with_fill_holes_yields_binary_false() {
                let args = MaskProcessingArgs {
                    binary: BinaryOption::Disabled,
                    fill_holes: true,
                    ..default_args()
                };
                let opts = MaskProcessingOptions::from(&args);
                assert!(!opts.binary);
            }

            #[test]
            fn enabled_always_yields_binary_true() {
                let args = MaskProcessingArgs {
                    binary: BinaryOption::Enabled,
                    ..default_args()
                };
                let opts = MaskProcessingOptions::from(&args);
                assert!(opts.binary);
            }

            #[test]
            fn blur_flags_and_sigma() {
                let args = MaskProcessingArgs {
                    blur: Some(10.0),
                    ..default_args()
                };
                let opts = MaskProcessingOptions::from(&args);
                assert!(opts.blur);
                assert!((opts.blur_sigma - 10.0).abs() < f32::EPSILON);
            }

            #[test]
            fn dilate_flags_and_radius() {
                let args = MaskProcessingArgs {
                    dilate: Some(8.0),
                    ..default_args()
                };
                let opts = MaskProcessingOptions::from(&args);
                assert!(opts.dilate);
                assert!((opts.dilation_radius - 8.0).abs() < f32::EPSILON);
            }

            #[test]
            fn threshold_passed_through() {
                let args = MaskProcessingArgs {
                    mask_threshold: 200,
                    ..default_args()
                };
                let opts = MaskProcessingOptions::from(&args);
                assert_eq!(opts.mask_threshold, 200);
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
        use clap::Parser;
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

        // default_missing_value behavior
        mod default_missing_value {
            use super::*;

            mod unit {
                use super::*;

                #[test]
                fn blur_flag_only_uses_default_sigma() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--blur"], Mask);
                    assert_eq!(cmd.mask_processing.blur, Some(6.0));
                }

                #[test]
                fn blur_with_value_uses_provided() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--blur", "10.0"], Mask);
                    assert_eq!(cmd.mask_processing.blur, Some(10.0));
                }

                #[test]
                fn binary_flag_only_becomes_enabled() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--binary"], Mask);
                    assert_eq!(cmd.mask_processing.binary, BinaryOption::Enabled);
                }

                #[test]
                fn binary_disabled_explicit() {
                    let cmd =
                        parse_cmd!(["outline", "mask", "in.png", "--binary", "disabled"], Mask);
                    assert_eq!(cmd.mask_processing.binary, BinaryOption::Disabled);
                }

                #[test]
                fn dilate_flag_only_uses_default_radius() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--dilate"], Mask);
                    assert_eq!(cmd.mask_processing.dilate, Some(5.0));
                }

                #[test]
                fn dilate_with_value_uses_provided() {
                    let cmd = parse_cmd!(["outline", "mask", "in.png", "--dilate", "8.0"], Mask);
                    assert_eq!(cmd.mask_processing.dilate, Some(8.0));
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
                    let cmd = parse_cmd!(
                        ["outline", "mask", "in.png", "--mask-threshold", "200"],
                        Mask
                    );
                    assert_eq!(cmd.mask_processing.mask_threshold, 200);
                }

                #[test]
                fn float_threshold_scaled() {
                    let cmd = parse_cmd!(
                        ["outline", "mask", "in.png", "--mask-threshold", "0.5"],
                        Mask
                    );
                    assert_eq!(cmd.mask_processing.mask_threshold, 128);
                }

                #[test]
                fn invalid_threshold_rejected() {
                    let result = Cli::try_parse_from([
                        "outline",
                        "mask",
                        "in.png",
                        "--mask-threshold",
                        "1.5",
                    ]);
                    assert!(result.is_err());
                }
            }
        }

        // ValueEnum parsing for global options and compose command
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

                #[test]
                fn invalid_bg_alpha_mode_rejected() {
                    let result = Cli::try_parse_from([
                        "outline",
                        "compose",
                        "in.png",
                        "--bg-alpha-mode",
                        "invalid",
                    ]);
                    assert!(result.is_err());
                }
            }
        }
    }
}
