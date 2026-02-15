mod cli;
mod commands;
#[cfg(feature = "model-fetch")]
mod model_fetch;
mod report;

use clap::Parser;
use std::process::ExitCode;

fn main() -> ExitCode {
    match try_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            report::report_error(&err);
            ExitCode::FAILURE
        }
    }
}

fn try_main() -> outline::OutlineResult<()> {
    let cli = cli::Cli::parse();
    commands::run(cli)
}
