//! # Outline
//!
//! Image background removal with flexible mask processing options.
//!
//! Powered by ONNX Runtime ([`ort`](https://docs.rs/ort)) and VTracer, and works with U2-Net, BiRefNet,
//! and other ONNX models with a compatible input/output shape.
//!
//! # Quick Start
//!
//! ```no_run
//! use outline::Outline;
//!
//! let outline = Outline::new("model.onnx");
//! let session = outline.for_image("input.png")?;
//! let matte = session.matte();
//!
//! // Compose the foreground directly from the raw matte (soft edges)
//! let foreground = matte.foreground()?;
//! foreground.save("foreground.png")?;
//!
//! // Process the mask and save it
//! let mask = matte.blur().threshold().processed()?;
//! mask.save("mask.png")?;
//! # Ok::<_, outline::OutlineError>(())
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

mod config;
mod error;
mod foreground;
mod inference;
mod mask;
mod vectorizer;

#[doc(inline)]
pub use crate::config::{
    DEFAULT_MODEL_PATH, ENV_MODEL_PATH, InferenceSettings, MaskProcessingOptions,
};
#[doc(inline)]
pub use crate::error::{OutlineError, OutlineResult};
pub use vectorizer::MaskVectorizer;

#[cfg(feature = "vectorizer-vtracer")]
#[cfg_attr(docsrs, doc(cfg(feature = "vectorizer-vtracer")))]
#[doc(inline)]
pub use vectorizer::vtracer::{TraceOptions, VtracerSvgVectorizer, trace_to_svg_string};

use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use image::imageops::FilterType;
use image::{GrayImage, RgbImage, RgbaImage};

use crate::foreground::compose_foreground;
use crate::inference::CachedInferenceSession;
use crate::mask::{MaskOperation, apply_operations, operations_from_options};

/// Entry point for configuring and running background matting inference.
///
/// This is the main interface for loading an ONNX model and processing images to extract
/// foreground subjects. Configure model path, inference settings, and default mask processing
/// options, then call [`for_image`](Outline::for_image) to run inference on individual images.
///
/// Each `Outline` instance lazily initializes and reuses its ONNX Runtime session.
#[derive(Debug)]
pub struct Outline {
    /// Inference settings for model and image handling.
    settings: InferenceSettings,
    /// If nothing is specified and processing is requested, these options will be used.
    default_mask_processing: MaskProcessingOptions,
    /// Lazily initialized cached session for this configured model.
    cached_session: Mutex<Option<Arc<CachedInferenceSession>>>,
}

impl Clone for Outline {
    fn clone(&self) -> Self {
        Self {
            settings: self.settings.clone(),
            default_mask_processing: self.default_mask_processing.clone(),
            cached_session: Mutex::new(None),
        }
    }
}

impl Outline {
    /// Create a new `Outline` instance with the given model path and default settings.
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            settings: InferenceSettings::new(model_path),
            default_mask_processing: MaskProcessingOptions::default(),
            cached_session: Mutex::new(None),
        }
    }

    /// Construct Outline using env var `ENV_MODEL_PATH` or fallback to `DEFAULT_MODEL_PATH`.
    pub fn from_env_or_default() -> Self {
        let resolved = std::env::var_os(ENV_MODEL_PATH)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH));
        Self::new(resolved)
    }

    /// Try constructing Outline strictly from env var; returns error if not set.
    pub fn try_from_env() -> OutlineResult<Self> {
        if let Some(from_env) = std::env::var_os(ENV_MODEL_PATH) {
            return Ok(Self::new(PathBuf::from(from_env)));
        }
        Err(OutlineError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Model path not specified in env {}; set the variable to proceed",
                ENV_MODEL_PATH
            ),
        )))
    }

    /// Set the filter used to resize the input image for the model.
    pub fn with_input_resize_filter(mut self, filter: FilterType) -> Self {
        self.settings.input_resize_filter = filter;
        self
    }

    /// Set the filter used to resize the output matte to the original image size.
    pub fn with_output_resize_filter(mut self, filter: FilterType) -> Self {
        self.settings.output_resize_filter = filter;
        self
    }

    /// Set the number of intra-op threads for the inference.
    pub fn with_intra_threads(mut self, intra_threads: Option<usize>) -> Self {
        if self.settings.intra_threads != intra_threads {
            self.settings.intra_threads = intra_threads;
            self.cached_session = Mutex::new(None);
        }
        self
    }

    /// Set the default mask processing options to use when none are specified.
    pub fn with_default_mask_processing(mut self, options: MaskProcessingOptions) -> Self {
        self.default_mask_processing = options;
        self
    }

    /// Get a reference to the default mask processing options.
    pub fn default_mask_processing(&self) -> &MaskProcessingOptions {
        &self.default_mask_processing
    }

    fn get_or_init_cached_session(&self) -> OutlineResult<Arc<CachedInferenceSession>> {
        let mut cached_session = self
            .cached_session
            .lock()
            .map_err(|_| std::io::Error::other("outline session cache mutex poisoned"))?;

        if let Some(session) = cached_session.as_ref() {
            return Ok(Arc::clone(session));
        }

        let session = Arc::new(CachedInferenceSession::new(&self.settings)?);
        *cached_session = Some(Arc::clone(&session));
        Ok(session)
    }

    /// Run the inference pipeline for a single image, returning the original image, raw matte, and processing options,
    /// wrapped in an `InferencedMatte`.
    pub fn for_image(&self, image_path: impl AsRef<Path>) -> OutlineResult<InferencedMatte> {
        let session = self.get_or_init_cached_session()?;
        let (rgb, matte) = session.run_matte_pipeline(&self.settings, image_path.as_ref())?;
        Ok(InferencedMatte::new(
            rgb,
            matte,
            self.default_mask_processing.clone(),
        ))
    }
}

/// Inference result containing the original RGB image and raw matte prediction.
///
/// Returned by [`Outline::for_image`] after running model inference.
///
/// # Example
/// ```no_run
/// use outline::Outline;
///
/// let outline = Outline::new("model.onnx");
/// let session = outline.for_image("input.png")?;
/// let matte = session.matte();
///
/// // Access the original image and raw matte directly
/// let rgb = session.rgb_image();
/// let raw_matte = session.raw_matte();
///
/// // Compose the foreground from the raw matte
/// let foreground = matte.foreground()?;
/// foreground.save("foreground.png")?;
/// # Ok::<_, outline::OutlineError>(())
/// ```
#[derive(Debug, Clone)]
pub struct InferencedMatte {
    rgb_image: Arc<RgbImage>,
    raw_matte: Arc<GrayImage>,
    default_mask_processing: MaskProcessingOptions,
}

impl InferencedMatte {
    fn new(
        rgb_image: RgbImage,
        raw_matte: GrayImage,
        default_mask_processing: MaskProcessingOptions,
    ) -> Self {
        Self {
            rgb_image: Arc::new(rgb_image),
            raw_matte: Arc::new(raw_matte),
            default_mask_processing,
        }
    }

    /// Get a reference to the original RGB image.
    pub fn rgb_image(&self) -> &RgbImage {
        self.rgb_image.as_ref()
    }

    /// Get a reference to the raw grayscale matte.
    pub fn raw_matte(&self) -> &GrayImage {
        self.raw_matte.as_ref()
    }

    /// Begin building a mask processing pipeline from the raw matte.
    pub fn matte(&self) -> MatteHandle {
        MatteHandle {
            rgb_image: Arc::clone(&self.rgb_image),
            raw_matte: Arc::clone(&self.raw_matte),
            default_mask_processing: self.default_mask_processing.clone(),
            operations: Vec::new(),
        }
    }
}

/// Builder for chaining mask processing operations on the raw matte.
///
/// The raw matte is the soft, grayscale alpha prediction from the model.
///
/// # Example
/// ```no_run
/// use outline::Outline;
///
/// let outline = Outline::new("model.onnx");
/// let session = outline.for_image("input.jpg")?;
///
/// // Chain operations and execute them
/// let mask = session.matte()
///     .blur_with(6.0)          // Smooth edges
///     .threshold_with(120)     // Convert to binary
///     .dilate_with(5.0)        // Expand slightly
///     .processed()?;           // Execute operations
///
/// mask.save("mask.png")?;
/// # Ok::<_, outline::OutlineError>(())
/// ```
#[derive(Debug, Clone)]
pub struct MatteHandle {
    rgb_image: Arc<RgbImage>,
    raw_matte: Arc<GrayImage>,
    default_mask_processing: MaskProcessingOptions,
    operations: Vec<MaskOperation>,
}

impl MatteHandle {
    /// Get the raw grayscale matte.
    pub fn raw(&self) -> GrayImage {
        (*self.raw_matte).clone()
    }

    /// Consume the handle and return the raw grayscale matte.
    pub fn into_image(self) -> GrayImage {
        (*self.raw_matte).clone()
    }

    /// Save the raw grayscale matte to the specified path.
    pub fn save(&self, path: impl AsRef<Path>) -> OutlineResult<()> {
        self.raw_matte.as_ref().save(path)?;
        Ok(())
    }

    /// Add a blur operation using the default sigma.
    pub fn blur(mut self) -> Self {
        let sigma = self.default_mask_processing.blur_sigma;
        self.operations.push(MaskOperation::Blur { sigma });
        self
    }

    /// Add a blur operation with a custom sigma.
    pub fn blur_with(mut self, sigma: f32) -> Self {
        self.operations.push(MaskOperation::Blur { sigma });
        self
    }

    /// Add a threshold operation using the default mask threshold.
    pub fn threshold(mut self) -> Self {
        let value = self.default_mask_processing.mask_threshold;
        self.operations.push(MaskOperation::Threshold { value });
        self
    }

    /// Add a threshold operation with a custom value.
    pub fn threshold_with(mut self, value: u8) -> Self {
        self.operations.push(MaskOperation::Threshold { value });
        self
    }

    /// Add a dilation operation using the default radius.
    ///
    /// **Note**: Dilation typically works best on binary masks. Consider calling
    /// [`threshold`](MatteHandle::threshold) before `dilate` if working with a soft matte.
    pub fn dilate(mut self) -> Self {
        let radius = self.default_mask_processing.dilation_radius;
        self.operations.push(MaskOperation::Dilate { radius });
        self
    }

    /// Add a dilation operation with a custom radius.
    ///
    /// **Note**: Dilation typically works best on binary masks. Consider calling
    /// [`threshold`](MatteHandle::threshold) before `dilate` if working with a soft matte.
    pub fn dilate_with(mut self, radius: f32) -> Self {
        self.operations.push(MaskOperation::Dilate { radius });
        self
    }

    /// Add a hole-filling operation to the processing pipeline.
    ///
    /// **Note**: Hole-filling typically works best on binary masks. Consider calling
    /// [`threshold`](MatteHandle::threshold) before `fill_holes` if working with a soft matte.
    pub fn fill_holes(mut self) -> Self {
        let threshold = self.default_mask_processing.mask_threshold;
        self.operations.push(MaskOperation::FillHoles { threshold });
        self
    }

    /// Process the raw matte with the accumulated operations and default options.
    pub fn processed(self) -> OutlineResult<MaskHandle> {
        self.process_with_options(None)
    }

    /// Process the raw matte with the accumulated operations and custom options.
    pub fn processed_with(self, options: &MaskProcessingOptions) -> OutlineResult<MaskHandle> {
        self.process_with_options(Some(options))
    }

    /// Helper function to process with options.
    fn process_with_options(
        mut self,
        options: Option<&MaskProcessingOptions>,
    ) -> OutlineResult<MaskHandle> {
        let mut ops = std::mem::take(&mut self.operations);
        match options {
            Some(custom) => ops.extend(operations_from_options(custom)),
            None if ops.is_empty() => {
                ops.extend(operations_from_options(&self.default_mask_processing))
            }
            None => {}
        }

        let mask = apply_operations(self.raw_matte.as_ref(), &ops);
        Ok(MaskHandle::new(
            Arc::clone(&self.rgb_image),
            mask,
            self.default_mask_processing,
        ))
    }

    /// Compose the RGBA foreground image from the RGB image and the raw matte.
    pub fn foreground(&self) -> OutlineResult<ForegroundHandle> {
        let rgba = compose_foreground(self.rgb_image.as_ref(), self.raw_matte.as_ref())?;
        Ok(ForegroundHandle { image: rgba })
    }

    /// Trace the raw matte using the specified vectorizer and options.
    pub fn trace<V>(&self, vectorizer: &V, options: &V::Options) -> OutlineResult<V::Output>
    where
        V: MaskVectorizer,
    {
        vectorizer.vectorize(self.raw_matte.as_ref(), options)
    }
}

/// Processed mask image with optional further refinement and output generation.
///
/// Represents a concrete mask image (typically binary after thresholding) produced by executing
/// operations from a [`MatteHandle`].
///
/// # Example
/// ```no_run
/// use outline::Outline;
///
/// let outline = Outline::new("model.onnx");
/// let session = outline.for_image("input.jpg")?;
/// let mask = session.matte().blur().threshold().processed()?;
///
/// // Generate multiple outputs from the mask
/// mask.save("mask.png")?;
/// mask.foreground()?.save("subject.png")?;
/// # Ok::<_, outline::OutlineError>(())
/// ```
///
/// To trace the mask into SVG, pass a [`MaskVectorizer`] implementation to [`MaskHandle::trace`].
/// The crate re-exports `VtracerSvgVectorizer` when the `vectorizer-vtracer` feature is enabled.
#[derive(Debug, Clone)]
pub struct MaskHandle {
    rgb_image: Arc<RgbImage>,
    mask: GrayImage,
    default_mask_processing: MaskProcessingOptions,
    operations: Vec<MaskOperation>,
}

impl MaskHandle {
    fn new(
        rgb_image: Arc<RgbImage>,
        mask: GrayImage,
        default_mask_processing: MaskProcessingOptions,
    ) -> Self {
        Self {
            rgb_image,
            mask,
            default_mask_processing,
            operations: Vec::new(),
        }
    }

    /// Get the raw  mask.
    pub fn raw(&self) -> GrayImage {
        self.mask.clone()
    }

    /// Get a reference to the mask.
    pub fn image(&self) -> &GrayImage {
        &self.mask
    }

    /// Consume the handle and return the mask.
    pub fn into_image(self) -> GrayImage {
        self.mask
    }

    /// Save the mask to the specified path.
    pub fn save(&self, path: impl AsRef<Path>) -> OutlineResult<()> {
        self.mask.save(path)?;
        Ok(())
    }

    /// Add a blur operation using the default sigma.
    pub fn blur(mut self) -> Self {
        let sigma = self.default_mask_processing.blur_sigma;
        self.operations.push(MaskOperation::Blur { sigma });
        self
    }

    /// Add a blur operation with a custom sigma.
    pub fn blur_with(mut self, sigma: f32) -> Self {
        self.operations.push(MaskOperation::Blur { sigma });
        self
    }

    /// Add a threshold operation using the default mask threshold.
    pub fn threshold(mut self) -> Self {
        let value = self.default_mask_processing.mask_threshold;
        self.operations.push(MaskOperation::Threshold { value });
        self
    }

    /// Add a threshold operation with a custom value.
    pub fn threshold_with(mut self, value: u8) -> Self {
        self.operations.push(MaskOperation::Threshold { value });
        self
    }

    /// Add a dilation operation using the default radius.
    ///
    /// **Note**: Dilation typically works best on binary masks. If this mask is still grayscale,
    /// consider calling [`threshold`](MaskHandle::threshold) first.
    pub fn dilate(mut self) -> Self {
        let radius = self.default_mask_processing.dilation_radius;
        self.operations.push(MaskOperation::Dilate { radius });
        self
    }

    /// Add a dilation operation with a custom radius.
    ///
    /// **Note**: Dilation typically works best on binary masks. If this mask is still grayscale,
    /// consider calling [`threshold`](MaskHandle::threshold) first.
    pub fn dilate_with(mut self, radius: f32) -> Self {
        self.operations.push(MaskOperation::Dilate { radius });
        self
    }

    /// Add a hole-filling operation to the processing pipeline.
    ///
    /// **Note**: Hole-filling typically works best on binary masks. If this mask is still grayscale,
    /// consider calling [`threshold`](MaskHandle::threshold) first.
    pub fn fill_holes(mut self) -> Self {
        let threshold = self.default_mask_processing.mask_threshold;
        self.operations.push(MaskOperation::FillHoles { threshold });
        self
    }

    /// Process the mask with the accumulated operations and default options.
    pub fn processed(self) -> OutlineResult<MaskHandle> {
        self.process_with_options(None)
    }

    /// Process the mask with the accumulated operations and custom options.
    pub fn processed_with(self, options: &MaskProcessingOptions) -> OutlineResult<MaskHandle> {
        self.process_with_options(Some(options))
    }

    /// Helper function to process with options.
    fn process_with_options(
        mut self,
        options: Option<&MaskProcessingOptions>,
    ) -> OutlineResult<MaskHandle> {
        let mut ops = std::mem::take(&mut self.operations);
        match options {
            Some(custom) => ops.extend(operations_from_options(custom)),
            None if ops.is_empty() => {
                ops.extend(operations_from_options(&self.default_mask_processing))
            }
            None => {}
        }

        let mask = apply_operations(&self.mask, &ops);
        Ok(MaskHandle::new(
            self.rgb_image,
            mask,
            self.default_mask_processing,
        ))
    }

    /// Compose the RGBA foreground image from the RGB image and the current mask.
    pub fn foreground(&self) -> OutlineResult<ForegroundHandle> {
        let rgba = compose_foreground(self.rgb_image.as_ref(), &self.mask)?;
        Ok(ForegroundHandle { image: rgba })
    }

    /// Trace the current mask using the specified vectorizer and options.
    pub fn trace<V>(&self, vectorizer: &V, options: &V::Options) -> OutlineResult<V::Output>
    where
        V: MaskVectorizer,
    {
        vectorizer.vectorize(&self.mask, options)
    }
}

/// Composed RGBA foreground image with transparent background.
///
/// Final output produced by composing the original RGB image with a mask as the alpha channel.
/// The mask's grayscale values map to alpha, producing smooth or hard edges depending on processing.
/// Obtain by calling [`foreground`](MatteHandle::foreground) on a [`MatteHandle`] or [`MaskHandle`].
///
/// # Example
/// ```no_run
/// use outline::Outline;
///
/// let outline = Outline::new("model.onnx");
/// let session = outline.for_image("input.jpg")?;
///
/// // Soft edges from raw matte
/// let soft = session.matte().foreground()?;
/// soft.save("soft-cutout.png")?;
///
/// // Hard edges from processed mask
/// let hard = session.matte()
///     .blur()
///     .threshold()
///     .processed()?
///     .foreground()?;
/// hard.save("hard-cutout.png")?;
/// # Ok::<_, outline::OutlineError>(())
/// ```
pub struct ForegroundHandle {
    image: RgbaImage,
}

impl ForegroundHandle {
    /// Get a reference to the RGBA foreground image.
    pub fn image(&self) -> &RgbaImage {
        &self.image
    }

    /// Consume the handle and return the RGBA foreground image.
    pub fn into_image(self) -> RgbaImage {
        self.image
    }

    /// Save the RGBA foreground image to the specified path.
    pub fn save(&self, path: impl AsRef<Path>) -> OutlineResult<()> {
        self.image.save(path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Mutex;
    use tempfile::NamedTempFile;

    // Serialize env-var tests so they don't race each other.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    // Embed the tiny ORT-format identity model into test binaries so tests remain self-contained.
    const ORT_IDENTITY_MODEL_BYTES: &[u8] = include_bytes!("../tests/fixtures/identity.ort");

    fn ort_identity_model_file() -> NamedTempFile {
        let mut file = tempfile::Builder::new()
            .suffix(".ort")
            .tempfile()
            .expect("failed to create temporary identity model");
        file.write_all(ORT_IDENTITY_MODEL_BYTES)
            .expect("failed to write temporary identity model");
        file.flush()
            .expect("failed to flush temporary identity model");
        file
    }

    mod outline_new {
        use super::*;

        #[test]
        fn user_value_is_stored_directly() {
            let outline = Outline::new("/explicit/model.onnx");
            assert_eq!(
                outline.settings.model_path,
                PathBuf::from("/explicit/model.onnx")
            );
        }

        #[test]
        fn user_value_ignores_env_var() {
            let _lock = ENV_LOCK.lock().unwrap();
            // SAFETY: serialized by ENV_LOCK; no other thread reads this var concurrently.
            unsafe { std::env::set_var(ENV_MODEL_PATH, "env.onnx") };
            let outline = Outline::new("user.onnx");
            unsafe { std::env::remove_var(ENV_MODEL_PATH) };
            assert_eq!(outline.settings.model_path, PathBuf::from("user.onnx"));
        }
    }

    mod outline_from_env_or_default {
        use super::*;

        #[test]
        fn uses_env_var_when_set() {
            let _lock = ENV_LOCK.lock().unwrap();
            // SAFETY: serialized by ENV_LOCK; no other thread reads this var concurrently.
            unsafe { std::env::set_var(ENV_MODEL_PATH, "/from/env.onnx") };
            let outline = Outline::from_env_or_default();
            unsafe { std::env::remove_var(ENV_MODEL_PATH) };
            assert_eq!(outline.settings.model_path, PathBuf::from("/from/env.onnx"));
        }

        #[test]
        fn falls_back_to_default_when_env_unset() {
            let _lock = ENV_LOCK.lock().unwrap();
            // SAFETY: serialized by ENV_LOCK; no other thread reads this var concurrently.
            unsafe { std::env::remove_var(ENV_MODEL_PATH) };
            let outline = Outline::from_env_or_default();
            assert_eq!(
                outline.settings.model_path,
                PathBuf::from(DEFAULT_MODEL_PATH)
            );
        }
    }

    mod outline_try_from_env {
        use super::*;

        #[test]
        fn succeeds_when_env_set() {
            let _lock = ENV_LOCK.lock().unwrap();
            // SAFETY: serialized by ENV_LOCK; no other thread reads this var concurrently.
            unsafe { std::env::set_var(ENV_MODEL_PATH, "/from/env.onnx") };
            let result = Outline::try_from_env();
            unsafe { std::env::remove_var(ENV_MODEL_PATH) };
            let outline = result.expect("should succeed when env is set");
            assert_eq!(outline.settings.model_path, PathBuf::from("/from/env.onnx"));
        }

        #[test]
        fn errors_when_env_unset() {
            let _lock = ENV_LOCK.lock().unwrap();
            // SAFETY: serialized by ENV_LOCK; no other thread reads this var concurrently.
            unsafe { std::env::remove_var(ENV_MODEL_PATH) };
            let result = Outline::try_from_env();
            assert!(result.is_err());
        }
    }

    mod outline_session_cache {
        use super::*;

        #[test]
        fn session_is_reused_within_same_outline() {
            let model = ort_identity_model_file();
            let outline = Outline::new(model.path());

            let first = outline
                .get_or_init_cached_session()
                .expect("should initialize cached session");
            let second = outline
                .get_or_init_cached_session()
                .expect("should reuse cached session");

            assert!(Arc::ptr_eq(&first, &second));
        }

        #[test]
        fn clone_starts_with_fresh_session_cache() {
            let model = ort_identity_model_file();
            let outline = Outline::new(model.path());
            let original = outline
                .get_or_init_cached_session()
                .expect("should initialize cached session");

            let cloned = outline.clone();
            let cloned_session = cloned
                .get_or_init_cached_session()
                .expect("should initialize cloned cached session");

            assert!(!Arc::ptr_eq(&original, &cloned_session));
        }

        #[test]
        fn non_session_settings_keep_cached_session() {
            let model = ort_identity_model_file();
            let outline = Outline::new(model.path());
            let cached = outline
                .get_or_init_cached_session()
                .expect("should initialize cached session");

            let outline = outline.with_input_resize_filter(FilterType::Nearest);
            let reused = outline
                .get_or_init_cached_session()
                .expect("should reuse cached session for non-session setting changes");

            assert!(Arc::ptr_eq(&cached, &reused));
        }

        #[test]
        fn intra_threads_change_clears_cached_session() {
            let model = ort_identity_model_file();
            let outline = Outline::new(model.path());
            let cached = outline
                .get_or_init_cached_session()
                .expect("should initialize cached session");

            let outline = outline.with_intra_threads(Some(1));
            let rebuilt = outline
                .get_or_init_cached_session()
                .expect("should rebuild cached session after thread change");

            assert!(!Arc::ptr_eq(&cached, &rebuilt));
        }
    }
}
