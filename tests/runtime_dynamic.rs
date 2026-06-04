#![cfg(feature = "ort-load-dynamic")]

mod support;

use support::runtime_dynamic;

#[ignore = "requires OUTLINE_TEST_ORT_DYLIB=/path/to/libonnxruntime.{dylib,so,dll}"]
#[test]
fn tiny_matte_model_runs_with_runtime_init() -> Result<(), Box<dyn std::error::Error>> {
    let runtime_path = runtime_dynamic::runtime_dylib_from_env()?;
    let model = support::tiny_matte_model_file();

    let committed = outline::runtime::init_onnx_runtime_from(&runtime_path)?;
    assert!(
        committed,
        "expected this test process to initialize ONNX Runtime only once"
    );

    runtime_dynamic::assert_tiny_matte_model_runs(model.path())?;
    Ok(())
}
