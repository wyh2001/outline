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
//!
//! # Advanced: ONNX Runtime Strategy
//!
//! By default, `outline-core` enables the `ort-download-binaries` feature so ONNX Runtime is
//! downloaded automatically for supported targets.
//!
//! If your environment needs a different runtime strategy, `outline-core` exposes the supported
//! non-default paths directly. These runtime strategy features are mutually exclusive; enable at
//! most one of `ort-download-binaries`, `ort-load-dynamic`, or `ort-pkg-config`:
//! - `ort-pkg-config`: discover a system ONNX Runtime via `pkg-config`
//! - `ort-load-dynamic`: load a `.dll`/`.so`/`.dylib` at runtime via the
//!   `runtime::init_onnx_runtime_from` helper in the [`runtime`] module
//! - [`runtime::ENV_ORT_DYLIB_PATH`]: choose the shared library used by dynamic loading
//! - [`runtime::ENV_ORT_LIB_LOCATION`]: link against a custom ONNX Runtime build
//! - [`runtime::ENV_ORT_PREFER_DYNAMIC_LINK`]: prefer shared-library linking for a custom build
//!
//! Setting `default-features = false` only disables the download fallback; use it together with
//! one of the non-default setups above.

#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(any(
    all(feature = "ort-download-binaries", feature = "ort-load-dynamic"),
    all(feature = "ort-download-binaries", feature = "ort-pkg-config"),
    all(feature = "ort-load-dynamic", feature = "ort-pkg-config"),
))]
compile_error!(
    "ONNX Runtime strategy features are mutually exclusive; enable at most one of `ort-download-binaries`, `ort-load-dynamic`, or `ort-pkg-config`."
);

mod config;
mod error;
mod foreground;
mod geometry;
mod inference;
mod mask;
mod matte;
pub mod runtime;
mod vectorizer;

#[doc(inline)]
pub use crate::config::{
    DEFAULT_MODEL_PATH, ENV_MODEL_PATH, ErosionBorderMode, InferenceSettings, MaskProcessingOptions,
};
#[doc(inline)]
pub use crate::error::{OutlineError, OutlineResult};
#[doc(inline)]
pub use crate::foreground::ForegroundHandle;
#[doc(inline)]
pub use crate::geometry::{BoundingBox, Padding};
#[doc(inline)]
pub use crate::mask::{MaskAlphaMode, MaskColor, MaskHandle, colorize_mask};
#[doc(inline)]
pub use crate::matte::{InferencedMatte, MatteHandle};
pub use vectorizer::MaskVectorizer;

#[cfg(feature = "vectorizer-vtracer")]
#[cfg_attr(docsrs, doc(cfg(feature = "vectorizer-vtracer")))]
#[doc(inline)]
pub use vectorizer::vtracer::{TraceOptions, VtracerSvgVectorizer, trace_to_svg_string};

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::inference::CachedInferenceSession;
use image::imageops::FilterType;

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
