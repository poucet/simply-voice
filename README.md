# simply-voice

Speech-to-text, text-to-speech and voice activity detection behind three traits. Each provider is a Cargo feature.

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

* **`metal`** turns on Apple Silicon acceleration for the candle providers.
* **`whisper`** needs cmake to build.
* **No feature is on by default.**
* **`RealtimeProvider` is a trait only.** No provider in this crate implements it yet.

## Use

```toml
[dependencies]
simply-voice = { git = "https://github.com/poucet/simply-voice", features = ["kyutai", "pocket-tts", "metal"] }
```

* **Local models** download from Hugging Face on first use, into `~/.cache/huggingface`.

## Tests

* **The Kyutai tests load the real model.** The first run downloads about 2 GB.
  * `cargo test --features kyutai,metal --test kyutai -- --nocapture`
  * `--nocapture` prints the end-of-speech to final latency per utterance.
* **Fixtures** in `tests/fixtures/` are 16 kHz mono PCM16.

## Where it is used

* [lumina](https://github.com/poucet/lumina): the simply daemon and Aurora.
* voice-agent: the `va` voice client.

This crate started inside lumina. Its history came along.
