#![cfg(feature = "ort-load-dynamic")]

mod support;

use support::runtime_dynamic;

#[ignore = "requires OUTLINE_TEST_ORT_DYLIB and ORT_DYLIB_PATH=/path/to/libonnxruntime.{dylib,so,dll}"]
#[test]
fn tiny_matte_model_runs_with_env_runtime() -> Result<(), Box<dyn std::error::Error>> {
    let expected_runtime_path = runtime_dynamic::runtime_dylib_from_env()?;
    let configured_runtime_path =
        runtime_dynamic::shared_library_from_env(outline::runtime::ENV_ORT_DYLIB_PATH)?;
    let model = support::tiny_matte_model_file();

    assert_eq!(
        configured_runtime_path,
        expected_runtime_path,
        "{} should point to the same library as {} for this integration test",
        outline::runtime::ENV_ORT_DYLIB_PATH,
        runtime_dynamic::ENV_TEST_ORT_DYLIB
    );

    runtime_dynamic::assert_tiny_matte_model_runs(model.path())?;
    Ok(())
}
