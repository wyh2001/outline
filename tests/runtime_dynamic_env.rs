#![cfg(feature = "ort-load-dynamic")]

#[path = "support/runtime_dynamic.rs"]
mod support;

#[ignore = "requires OUTLINE_TEST_ORT_DYLIB and ORT_DYLIB_PATH=/path/to/libonnxruntime.{dylib,so,dll}"]
#[test]
fn identity_model_runs_with_env_runtime() -> Result<(), Box<dyn std::error::Error>> {
    let expected_runtime_path = support::runtime_dylib_from_env()?;
    let configured_runtime_path =
        support::shared_library_from_env(outline::runtime::ENV_ORT_DYLIB_PATH)?;
    let (_temp_dir, model_path) = support::write_identity_model()?;

    assert_eq!(
        configured_runtime_path,
        expected_runtime_path,
        "{} should point to the same library as {} for this integration test",
        outline::runtime::ENV_ORT_DYLIB_PATH,
        support::ENV_TEST_ORT_DYLIB
    );

    support::assert_identity_model_runs(&model_path)?;
    Ok(())
}
