use image::{GenericImage, GrayImage, Rgb, RgbImage, Rgba, RgbaImage};

/// Integer bounding box around image content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundingBox {
    /// Left edge of the bounding box in pixels.
    pub x: u32,
    /// Top edge of the bounding box in pixels.
    pub y: u32,
    /// Width of the bounding box in pixels.
    pub width: u32,
    /// Height of the bounding box in pixels.
    pub height: u32,
}

impl BoundingBox {
    /// Create a bounding box from its top-left corner and size.
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Return the exclusive right edge of the bounding box.
    pub const fn right(self) -> u32 {
        self.x
            .checked_add(self.width)
            .expect("bounding box right edge exceeds u32::MAX")
    }

    /// Return the exclusive bottom edge of the bounding box.
    pub const fn bottom(self) -> u32 {
        self.y
            .checked_add(self.height)
            .expect("bounding box bottom edge exceeds u32::MAX")
    }

    /// Expand the bounding box by `padding`, clamped to the image dimensions.
    pub fn expanded_to_fit(self, padding: Padding, image_width: u32, image_height: u32) -> Self {
        let x = self.x.saturating_sub(padding.left);
        let y = self.y.saturating_sub(padding.top);
        let right = self.right().saturating_add(padding.right).min(image_width);
        let bottom = self
            .bottom()
            .saturating_add(padding.bottom)
            .min(image_height);

        Self::new(x, y, right.saturating_sub(x), bottom.saturating_sub(y))
    }
}

/// Per-edge image padding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Padding {
    /// Padding added to the left edge.
    pub left: u32,
    /// Padding added to the top edge.
    pub top: u32,
    /// Padding added to the right edge.
    pub right: u32,
    /// Padding added to the bottom edge.
    pub bottom: u32,
}

impl Padding {
    /// Create padding from individual left, top, right, and bottom values.
    pub const fn new(left: u32, top: u32, right: u32, bottom: u32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    /// Create equal padding for all edges.
    pub const fn uniform(all: u32) -> Self {
        Self::new(all, all, all, all)
    }

    /// Create padding from horizontal and vertical edge values.
    pub const fn symmetric(horizontal: u32, vertical: u32) -> Self {
        Self::new(horizontal, vertical, horizontal, vertical)
    }

    /// Return the combined left and right padding.
    pub const fn horizontal(self) -> u32 {
        self.left
            .checked_add(self.right)
            .expect("horizontal padding exceeds u32::MAX")
    }

    /// Return the combined top and bottom padding.
    pub const fn vertical(self) -> u32 {
        self.top
            .checked_add(self.bottom)
            .expect("vertical padding exceeds u32::MAX")
    }
}

impl From<u32> for Padding {
    fn from(value: u32) -> Self {
        Self::uniform(value)
    }
}

/// Compute the smallest rectangle that contains all mask pixels at or above `threshold`.
pub fn mask_bounding_box(mask: &GrayImage, threshold: u8) -> Option<BoundingBox> {
    let (w, h) = mask.dimensions();
    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;

    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel[0] < threshold {
            continue;
        }
        found = true;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    found.then(|| BoundingBox::new(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
}

/// Compute the smallest rectangle that contains all pixels whose alpha is at or above `threshold`.
pub fn alpha_bounding_box(image: &RgbaImage, threshold: u8) -> Option<BoundingBox> {
    let (w, h) = image.dimensions();
    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;

    for (x, y, pixel) in image.enumerate_pixels() {
        if pixel[3] < threshold {
            continue;
        }
        found = true;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    found.then(|| BoundingBox::new(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
}

pub fn pad_gray_image(image: &GrayImage, padding: Padding, fill: u8) -> GrayImage {
    let (w, h) = image.dimensions();
    let out_width = w
        .checked_add(padding.horizontal())
        .expect("padded image width exceeds u32::MAX");
    let out_height = h
        .checked_add(padding.vertical())
        .expect("padded image height exceeds u32::MAX");
    let mut out = GrayImage::from_pixel(out_width, out_height, image::Luma([fill]));
    out.copy_from(image, padding.left, padding.top)
        .expect("source image should fit in padded image");

    out
}

pub fn pad_rgb_image(image: &RgbImage, padding: Padding, fill: [u8; 3]) -> RgbImage {
    let (w, h) = image.dimensions();
    let out_width = w
        .checked_add(padding.horizontal())
        .expect("padded image width exceeds u32::MAX");
    let out_height = h
        .checked_add(padding.vertical())
        .expect("padded image height exceeds u32::MAX");
    let mut out = RgbImage::from_pixel(out_width, out_height, Rgb(fill));
    out.copy_from(image, padding.left, padding.top)
        .expect("source image should fit in padded image");

    out
}

pub fn pad_rgba_image(image: &RgbaImage, padding: Padding, fill: [u8; 4]) -> RgbaImage {
    let (w, h) = image.dimensions();
    let out_width = w
        .checked_add(padding.horizontal())
        .expect("padded image width exceeds u32::MAX");
    let out_height = h
        .checked_add(padding.vertical())
        .expect("padded image height exceeds u32::MAX");
    let mut out = RgbaImage::from_pixel(out_width, out_height, Rgba(fill));
    out.copy_from(image, padding.left, padding.top)
        .expect("source image should fit in padded image");

    out
}

pub fn crop_gray_image(image: &GrayImage, bounds: BoundingBox) -> GrayImage {
    GrayImage::from_fn(bounds.width, bounds.height, |x, y| {
        *image.get_pixel(bounds.x + x, bounds.y + y)
    })
}

pub fn crop_rgb_image(image: &RgbImage, bounds: BoundingBox) -> RgbImage {
    RgbImage::from_fn(bounds.width, bounds.height, |x, y| {
        *image.get_pixel(bounds.x + x, bounds.y + y)
    })
}

pub fn crop_rgba_image(image: &RgbaImage, bounds: BoundingBox) -> RgbaImage {
    RgbaImage::from_fn(bounds.width, bounds.height, |x, y| {
        *image.get_pixel(bounds.x + x, bounds.y + y)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn mask_bounding_box_finds_smallest_box() {
        let mut image = GrayImage::from_pixel(6, 5, Luma([0]));
        image.put_pixel(2, 1, Luma([255]));
        image.put_pixel(4, 3, Luma([255]));

        let bounds = mask_bounding_box(&image, 1).expect("expected bounding box");
        assert_eq!(bounds, BoundingBox::new(2, 1, 3, 3));
    }

    #[test]
    fn alpha_bounding_box_uses_alpha_channel() {
        let mut image = RgbaImage::from_pixel(4, 4, Rgba([0, 0, 0, 0]));
        image.put_pixel(1, 2, Rgba([10, 20, 30, 200]));

        let bounds = alpha_bounding_box(&image, 1).expect("expected alpha bounds");
        assert_eq!(bounds, BoundingBox::new(1, 2, 1, 1));
    }

    #[test]
    fn pad_gray_image_expands_canvas_without_changing_content() {
        let mut image = GrayImage::from_pixel(2, 2, Luma([0]));
        image.put_pixel(1, 1, Luma([255]));

        let padded = pad_gray_image(&image, Padding::new(1, 2, 3, 4), 0);
        assert_eq!(padded.dimensions(), (6, 8));
        assert_eq!(padded.get_pixel(2, 3)[0], 255);
    }

    #[test]
    #[should_panic(expected = "horizontal padding exceeds u32::MAX")]
    fn padding_horizontal_overflow_panics() {
        let _ = Padding::new(u32::MAX, 0, 1, 0).horizontal();
    }

    #[test]
    #[should_panic(expected = "padded image width exceeds u32::MAX")]
    fn pad_gray_image_width_overflow_panics() {
        let image = GrayImage::from_pixel(1, 1, Luma([0]));
        let _ = pad_gray_image(&image, Padding::new(u32::MAX, 0, 0, 0), 0);
    }

    #[test]
    fn crop_rgba_image_returns_requested_region() {
        let mut image = RgbaImage::from_pixel(4, 4, Rgba([0, 0, 0, 0]));
        image.put_pixel(2, 1, Rgba([1, 2, 3, 4]));

        let cropped = crop_rgba_image(&image, BoundingBox::new(1, 1, 2, 2));
        assert_eq!(cropped.dimensions(), (2, 2));
        assert_eq!(cropped.get_pixel(1, 0).0, [1, 2, 3, 4]);
    }
}
