use std::fs;
use std::path::Path;
use image::imageops::FilterType;
use image::{GrayImage, ImageReader, Luma, RgbImage};
use imageproc::distance_transform::euclidean_squared_distance_transform;
use imageproc::filter::gaussian_blur_f32;
use ndarray::{Array2, Array4, Axis, Ix2, Ix3, Ix4};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use visioncortex::PathSimplifyMode;
use vtracer::ColorMode::Binary;
use vtracer::{convert, ColorImage, Config, Hierarchical, Preset, SvgFile};

fn pick_modnet_size(w: u32, h: u32) -> (u32, u32) {
    let short_target = 512.0f32;
    let short_edge = w.min(h) as f32;
    let scale = short_target / short_edge;
    let align32 = |value: u32| ((value + 31) / 32) * 32;
    let mut new_w = ((w as f32) * scale).round() as u32;
    let mut new_h = ((h as f32) * scale).round() as u32;
    new_w = new_w.max(32);
    new_h = new_h.max(32);
    (align32(new_w), align32(new_h))
}

fn preprocess_modnet_to_tensor(rgb: &RgbImage) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
    let (target_w, target_h) = pick_modnet_size(rgb.width(), rgb.height());
    let resized = image::imageops::resize(rgb, target_w, target_h, FilterType::Triangle);
    let w = resized.width() as usize;
    let h = resized.height() as usize;
    let mut nchw = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let pixel = resized.get_pixel(x as u32, y as u32);
            let norm = 1.0 / 255.0;
            let r = (f32::from(pixel[0]) * norm - 0.5) / 0.5;
            let g = (f32::from(pixel[1]) * norm - 0.5) / 0.5;
            let b = (f32::from(pixel[2]) * norm - 0.5) / 0.5;
            let idx = y * w + x;
            nchw[idx] = r;
            nchw[h * w + idx] = g;
            nchw[2 * h * w + idx] = b;
        }
    }
    let array = Array4::from_shape_vec((1, 3, h, w), nchw)?;
    Ok(Tensor::from_array(array)?)
}

fn extract_matte_hw(matte: ndarray::ArrayViewD<f32>) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let hw = match matte.ndim() {
        4 => {
            let v4 = matte.into_dimensionality::<Ix4>()?;
            let v3 = v4.index_axis(Axis(0), 0);
            let v2 = v3.index_axis(Axis(0), 0);
            v2.to_owned()
        }
        3 => {
            let v3 = matte.into_dimensionality::<Ix3>()?;
            let v2 = v3.index_axis(Axis(0), 0);
            v2.to_owned()
        }
        2 => matte.into_dimensionality::<Ix2>()?.to_owned(),
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported output shape: {:?}", matte.shape()),
            )
            .into())
        }
    };
    Ok(hw)
}

fn resize_matte(
    matte: &Array2<f32>,
    target_w: u32,
    target_h: u32,
    filter: FilterType,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let src_w = matte.shape()[1] as u32;
    let src_h = matte.shape()[0] as u32;
    let mut buffer = image::ImageBuffer::<Luma<f32>, Vec<f32>>::new(src_w, src_h);
    for (y, row) in matte.axis_iter(Axis(0)).enumerate() {
        for (x, &value) in row.iter().enumerate() {
            buffer.put_pixel(x as u32, y as u32, Luma([value]));
        }
    }
    let resized = image::imageops::resize(&buffer, target_w, target_h, filter);
    let mut out = Array2::<f32>::zeros((target_h as usize, target_w as usize));
    for (x, y, pixel) in resized.enumerate_pixels() {
        out[[y as usize, x as usize]] = pixel[0];
    }
    Ok(out)
}

fn array_to_gray_image(array: &Array2<f32>) -> GrayImage {
    let h = array.shape()[0];
    let w = array.shape()[1];
    let mut gray = GrayImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let value = array[[y, x]].clamp(0.0, 1.0);
            let byte = (value * 255.0 + 0.5) as u8;
            gray.put_pixel(x as u32, y as u32, Luma([byte]));
        }
    }
    gray
}

fn gray_to_color_image_rgba(gray: &GrayImage, threshold: Option<u8>) -> ColorImage {
    let (w, h) = gray.dimensions();
    let (w_usize, h_usize) = (w as usize, h as usize);
    let mut rgba = vec![0u8; 4 * w_usize * h_usize];

    for y in 0..h_usize {
        for x in 0..w_usize {
            let Luma([g]) = gray.get_pixel(x as u32, y as u32);
            let v = if let Some(t) = threshold {
                if *g >= t { 255 } else { 0 }
            } else {
                *g
            };
            let idx = (y * w_usize + x) * 4;
            rgba[idx] = v;
            rgba[idx + 1] = v;
            rgba[idx + 2] = v;
            rgba[idx + 3] = 255;
        }
    }

    ColorImage {
        pixels: rgba,
        width: w_usize,
        height: h_usize,
    }
}

fn trace(img: ColorImage) -> Result<SvgFile, Box<dyn std::error::Error>> {
    let mut cfg = Config::from_preset(Preset::Bw);
    cfg.color_mode = Binary;
    cfg.hierarchical = Hierarchical::Cutout;
    cfg.mode = PathSimplifyMode::Spline;
    cfg.filter_speckle = 4;
    cfg.path_precision = Some(8);

    let svg_file = convert(img, cfg).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    Ok(svg_file)
}

fn blur_then_threshold(gray: &GrayImage, sigma: f32, thr: u8) -> GrayImage {
    let blurred = gaussian_blur_f32(gray, sigma);
    let (w, h) = blurred.dimensions();
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let Luma([v]) = blurred.get_pixel(x, y);
            let bin = if *v >= thr { 255 } else { 0 };
            out.put_pixel(x, y, Luma([bin]));
        }
    }
    out
}

fn dilate_euclidean(mask_bin: &GrayImage, r: f32) -> GrayImage {
    let d2 = euclidean_squared_distance_transform(mask_bin);
    let r2: f64 = (r as f64) * (r as f64);
    let (w, h) = mask_bin.dimensions();
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let d2xy: f64 = d2.get_pixel(x, y)[0];
            let v: u8 = if d2xy <= r2 { 255 } else { 0 };
            out.put_pixel(x, y, Luma([v]));
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = Path::new("model.onnx");
    let image_path = Path::new("test.jfif");
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;
    let rgb_input = ImageReader::open(image_path)?.decode()?.to_rgb8();
    let orig_w = rgb_input.width();
    let orig_h = rgb_input.height();
    let input_tensor = preprocess_modnet_to_tensor(&rgb_input)?;
    let outputs = session.run(ort::inputs![input_tensor])?;
    let matte = outputs[0].try_extract_array::<f32>()?;
    println!("Output shape: {:?}", matte.shape());
    let matte_hw = extract_matte_hw(matte)?;
    let matte_orig = resize_matte(&matte_hw, orig_w, orig_h, FilterType::CatmullRom)?;
    let gray_orig = array_to_gray_image(&matte_orig);
    gray_orig.save("matte_origsize.png")?;
    println!("Saved matte_origsize.png.");
    let sigma = 6.0;
    let thr_out = 120u8;
    let smooth = blur_then_threshold(&gray_orig, sigma, thr_out);
    let dilate = dilate_euclidean(&smooth, 5.0);
    let color_img = gray_to_color_image_rgba(&dilate, Some(128));
    let svg = trace(color_img)?;
    fs::write("outline.svg", &svg.to_string())?;
    Ok(())
}
