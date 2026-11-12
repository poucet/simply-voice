# simply-voice

Speech-to-text, text-to-speech and voice activity detection behind three traits. Each provider is a Cargo feature.

The repo also ships `sv`, a small command line on top of the library: an easy way for a person or an agent to speak and listen.

## The shape

```
             ┌─ SttProvider ──── transcribe(chunk) · stream()
simply-voice ├─ TtsProvider ──── synthesize(text, voice) · stream() · voices()
             ├─ RealtimeProvider  connect(...)
             └─ VoiceActivityDetector  energy-based, emits speech start / end
```

* **Callers depend on the traits**, never on a provider.
* **Streaming is a channel pair**: send audio or text in, receive transcriptions or audio chunks out.

## Providers

| feature | provider | STT | TTS | runs |
|---|---|---|---|---|
| `kyutai` | `KyutaiSttProvider` | ✅ | | locally, candle |
| `pocket-tts` | `PocketTtsProvider` | | ✅ | locally, candle |
| `whisper` | `WhisperProvider` | ✅ | | locally, whisper.cpp |
| `voxtral` | `VoxtralProvider` | ✅ | ✅ | remote API |
| `gemini` | `GeminiProvider` | ✅ | ✅ | remote API |
| `elevenlabs` | `ElevenLabsProvider` | ✅ | ✅ | remote API |

* **`PocketTtsProvider::with_voice_dir`** adds custom voices: each `*.safetensors` embedding or `*.wav` clip in the directory is a voice named after its file stem.
* **`metal`** turns on Apple Silicon acceleration for the candle providers.
* **`whisper`** needs cmake to build.
* **No feature is on by default.**
* **`RealtimeProvider` is a trait only.** No provider in this crate implements it yet.

## Use

```toml
[dependencies]
simply-voice = { git = "https://github.com/simply-audio/simply-voice", features = ["kyutai", "pocket-tts", "metal"] }
```

* **Local models** download from Hugging Face on first use, into `~/.cache/huggingface`.

## sv

```
sv say "The build is green."     speak in the configured voice; returns when done
sv listen                        wait for one spoken turn, print it as one line of text
sv devices [--pick]              list audio devices, or choose and save them
```

* **Build**: `cargo build --release -p simply-voice-cli`, giving `target/release/sv`.
* **`sv listen` exits** 0 with the transcript on stdout, 1 when nobody started speaking within `--timeout` (30 s), 2 on an error.
* **Devices**: `--input` / `--output` on every command take a number from `sv devices` or part of a device name.
* **Config**: `~/.config/simply-voice/config.toml`, sections `[audio]`, `[stt]`, `[tts]`; unknown keys are ignored.
* **For agents**: the `voice-say` skill in `skills/voice-say` wraps `sv say` and `sv listen`.

More in [docs/guides/BUILD.md](docs/guides/BUILD.md).

## Tests

* **The Kyutai tests load the real model.** The first run downloads about 2 GB.
  * `cargo test -p simply-voice --features kyutai,metal --test kyutai -- --nocapture`
  * `--nocapture` prints the end-of-speech to final latency per utterance.
* **Fixtures** in `crates/simply-voice/tests/fixtures/` are 16 kHz mono PCM16, shared with the `sv` turn tests.
* `cargo test --workspace` runs everything else, the `sv` turn tests included (they load Kyutai too).

## Where it is used

* [lumina](https://github.com/poucet/lumina): the simply daemon and Aurora.
* `sv`, in this repo.

This crate started inside lumina. Its history came along.
