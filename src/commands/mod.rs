mod cut;
#[cfg(feature = "fetch-model")]
mod fetch_model;
mod mask;
mod trace;
mod utils;

use crate::cli::{Cli, Commands, GlobalOptions};
use outline::OutlineResult;

/// The main function to run the command based on CLI input.
pub fn run(cli: Cli) -> OutlineResult<()> {
    let Cli { global, command } = cli;
    dispatch(&global, command)
}

/// Dispatch the command to the appropriate handler.
fn dispatch(global: &GlobalOptions, command: Commands) -> OutlineResult<()> {
    match command {
        Commands::Mask(cmd) => mask::run(global, cmd),
        Commands::Cut(cmd) => cut::run(global, cmd),
        Commands::Trace(cmd) => trace::run(global, cmd),
        #[cfg(feature = "fetch-model")]
        Commands::FetchModel(cmd) => fetch_model::run(cmd),
    }
}
