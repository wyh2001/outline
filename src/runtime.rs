//! ONNX Runtime linkage and loading helpers exposed by `outline-core`.
//!
//! By default, `outline-core` enables the `ort-download-binaries` feature, which mirrors `ort`'s
//! prebuilt runtime download path. Alternative strategies can be selected with crate features or
//! build environment variables.

/// Runtime environment variable used by `ort` to locate a specific ONNX Runtime shared library
/// when `ort-load-dynamic` is enabled.
pub const ENV_ORT_DYLIB_PATH: &str = "ORT_DYLIB_PATH";

/// Build-time environment variable used by `ort-sys` to locate a custom ONNX Runtime build.
pub const ENV_ORT_LIB_LOCATION: &str = "ORT_LIB_LOCATION";

/// Build-time environment variable that prefers dynamic linking when `ENV_ORT_LIB_LOCATION` points
/// at a directory containing shared libraries.
pub const ENV_ORT_PREFER_DYNAMIC_LINK: &str = "ORT_PREFER_DYNAMIC_LINK";

/// Initialize ONNX Runtime from a specific shared library file.
///
/// This is available when the `ort-load-dynamic` feature is enabled and must be called before any
/// other `outline` APIs are used for the loaded runtime to take effect.
///
/// ```no_run
/// # #[cfg(feature = "ort-load-dynamic")]
/// # fn main() -> outline::OutlineResult<()> {
/// let lib_path = std::env::current_exe()?.parent().unwrap().join(std::env::consts::DLL_PREFIX.to_owned() + "onnxruntime" + std::env::consts::DLL_SUFFIX);
/// let committed = outline::runtime::init_onnx_runtime_from(&lib_path)?;
/// assert!(committed);
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "ort-load-dynamic"))]
/// # fn main() {}
/// ```
///
/// Returns `true` if the dynamic runtime configuration was committed, or `false` if ONNX Runtime
/// had already been initialized earlier in the process.
#[cfg(feature = "ort-load-dynamic")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort-load-dynamic")))]
pub fn init_onnx_runtime_from(path: impl AsRef<std::path::Path>) -> crate::OutlineResult<bool> {
    Ok(ort::init_from(path)?.commit())
}
