#![cfg(all(feature = "ort-load-dynamic", not(target_arch = "wasm32")))]

#[test]
fn missing_runtime_returns_load_error() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let runtime_path = temp_dir.path().join("missing-onnxruntime-library");

    let error = outline::runtime::init_onnx_runtime_from(&runtime_path)
        .expect_err("a missing runtime should return a dynamic loading error");

    match error {
        outline::OutlineError::OrtLoadDynamic(ort::LoadDynamicError::Dlopen { path, .. }) => {
            assert_eq!(path, runtime_path);
        }
        other => panic!("expected a dynamic loading error, got {other:?}"),
    }

    Ok(())
}
