mod cli;
mod commands;

use clap::Parser;
use outline::OutlineResult;

fn main() -> OutlineResult<()> {
    let cli = cli::Cli::parse();
    commands::run(cli)
}
