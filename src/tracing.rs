use vtracer::{ColorImage, Config, SvgFile, convert};

use crate::config::TraceConfig;

pub fn trace(img: ColorImage, config: &TraceConfig) -> Result<SvgFile, Box<dyn std::error::Error>> {
    let mut cfg = Config::from_preset(config.tracer_preset.clone());
    cfg.color_mode = config.tracer_color_mode.clone();
    cfg.hierarchical = config.tracer_hierarchical.clone();
    cfg.mode = config.tracer_mode;
    cfg.filter_speckle = config.tracer_filter_speckle;
    cfg.path_precision = config.tracer_path_precision;

    let svg_file =
        convert(img, cfg).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    Ok(svg_file)
}
