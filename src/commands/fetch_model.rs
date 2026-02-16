//! Handler for the `fetch-model` command.

use outline::OutlineResult;

use crate::cli::FetchModelCommand;
use crate::model_fetch::{FetchOptions, default_model_cache_path, fetch_model};

/// Run the fetch-model command.
pub fn run(cmd: FetchModelCommand) -> OutlineResult<()> {
    let output = cmd.output.unwrap_or_else(default_model_cache_path);

    let options = FetchOptions::default()
        .with_output(output)
        .with_force(cmd.force);

    fetch_model(&options)?;

    Ok(())
}
