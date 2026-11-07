//! `sv` — say, listen, devices. Built for people and for agents: `sv say`
//! speaks, `sv listen` returns what was said back as one line of text.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use simply_voice_cli::audio::AudioConfig;
use simply_voice_cli::cli::{self, EXIT_ERROR, EXIT_TIMEOUT, Listened};
use simply_voice_cli::config::{Config, ConfigSource};

#[derive(Parser)]
#[command(name = "sv", version, about = "Speak, listen and pick audio devices")]
struct Cli {
    /// Config file.
    #[arg(long, global = true, default_value_os_t = Config::default_path())]
    config: PathBuf,
    /// Input device: a number from `sv devices`, or part of its name
    /// (case-insensitive); overrides `[audio] input`.
    #[arg(long, global = true)]
    input: Option<String>,
    /// Output device, same matching; overrides `[audio] output`.
    #[arg(long, global = true)]
    output: Option<String>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Speak text in the configured voice; returns when playback ends.
    Say(cli::SayArgs),
    /// Listen for one spoken turn and print it as one line of text.
    ///
    /// Exit codes: 0 heard something (the transcript is on stdout), 1 nobody
    /// started speaking within --timeout (stdout empty), 2 error (device,
    /// model, config).
    Listen(cli::ListenArgs),
    /// List audio devices, numbered for `--input`/`--output`; `*` marks the
    /// system default, `>` the one sv would use.
    Devices {
        /// Choose input and output interactively and save them to the config.
        #[arg(long)]
        pick: bool,
    },
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(code) => code,
        Err(e) => {
            eprintln!("sv: {e:#}");
            ExitCode::from(EXIT_ERROR)
        }
    }
}

async fn run(cli: Cli) -> anyhow::Result<ExitCode> {
    // The config used to live under voice-agent; carry it over on first run.
    if cli.config == Config::default_path()
        && let Err(e) = Config::migrate(&cli.config, &Config::legacy_path())
    {
        eprintln!("sv: {e:#}");
    }
    let source = ConfigSource {
        path: cli.config,
        audio: AudioConfig {
            input: cli.input,
            output: cli.output,
        },
    };
    match cli.command {
        Command::Say(a) => cli::say(source.load()?, a).await?,
        Command::Listen(a) => {
            let events = a.events;
            match cli::listen(source.load()?, a).await? {
                Listened::Heard(text) => println!("{text}"),
                Listened::Nothing if !events => return Ok(ExitCode::from(EXIT_TIMEOUT)),
                Listened::Nothing => {}
            }
        }
        Command::Devices { pick } => cli::devices(&source, pick)?,
    }
    Ok(ExitCode::SUCCESS)
}
