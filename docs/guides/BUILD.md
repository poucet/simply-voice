# Build

A cargo workspace: `crates/simply-voice` is the library, `crates/sv` the `sv`
command line (package `simply-voice-cli`) built on it. `simply-voice` itself
has no git dependencies; `sv` depends on it by path.

    cargo build --workspace
    cargo test --workspace
    cargo build --release -p simply-voice-cli    # target/release/sv

The voice models (Kyutai STT, pocket-tts) download from Hugging Face on first
use into `~/.cache/huggingface`; the Aurora voice is read from
`~/.local/share/noema/models/aurora-voices/aurora.safetensors` (`[tts] voice`
in `~/.config/simply-voice/config.toml`). Dependencies build optimised even in
dev builds (`[profile.dev.package."*"]`), because the models are unusably slow
otherwise — tests load them too. `cargo build -p simply-voice-cli --features
whisper` adds the whisper.cpp STT backend (needs cmake).

## sv

    sv say "text" ["more"…]       speak in the configured voice; returns when done
    sv listen                     wait for one spoken turn, print it as one line
    sv listen --events [--wav f]  every SpeechStart/Partial/Final with timings
    sv devices [--pick]           list / choose audio devices

`--input` / `--output` work on every command and take a number from
`sv devices` or part of a device name. `sv listen` exits 0 with the transcript
on stdout, 1 when nobody started speaking within `--timeout` (default 30 s),
2 on an error.

The first run copies `~/.config/voice-agent/config.toml` to
`~/.config/simply-voice/config.toml` if only the old one exists.
