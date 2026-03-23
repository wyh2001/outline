#![cfg(feature = "ort-load-dynamic")]

#[path = "support/runtime_dynamic.rs"]
mod support;

#[ignore = "requires OUTLINE_TEST_ORT_DYLIB=/path/to/libonnxruntime.{dylib,so,dll}"]
#[test]
fn identity_model_runs_with_runtime_init() -> Result<(), Box<dyn std::error::Error>> {
    let runtime_path = support::runtime_dylib_from_env()?;
    let (_temp_dir, model_path) = support::write_identity_model()?;

    let committed = outline::runtime::init_onnx_runtime_from(&runtime_path)?;
    assert!(
        committed,
        "expected this test process to initialize ONNX Runtime only once"
    );

    support::assert_identity_model_runs(&model_path)?;
    Ok(())
}
