use image::GrayImage;

use crate::OutlineResult;

/// A trait representing an algorithm that can turn a mask into a vector representation.
pub trait MaskVectorizer {
    /// Configuration options for the vectorizer.
    type Options;
    /// The output type produced by the vectorizer.
    type Output;

    /// Convert a grayscale mask into a vector representation.
    fn vectorize(&self, mask: &GrayImage, options: &Self::Options) -> OutlineResult<Self::Output>;
}

#[cfg(feature = "vectorizer-vtracer")]
pub mod vtracer;
