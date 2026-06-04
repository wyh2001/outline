pub mod tiny_onnx;

pub use tiny_onnx::tiny_matte_model_file;

#[cfg(feature = "ort-load-dynamic")]
pub mod runtime_dynamic;
