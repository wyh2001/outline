use outline::OutlineError;

pub fn report_error(err: &OutlineError) {
    match err {
        OutlineError::ModelNotFound { path } => {
            eprintln!("Model file not found: {}", path.display());
            eprintln!();
            eprintln!("Please specify the model path:");
            eprintln!("  - Use --model <path>");
            eprintln!(
                "  - Or set environment variable {} to your model path",
                outline::ENV_MODEL_PATH
            );
            #[cfg(feature = "fetch-model")]
            {
                eprintln!();
                eprintln!("Or run `outline fetch-model` to download automatically.");
            }
        }
        _ => {
            eprintln!("{err}");
        }
    }
}
