# S2 audio: model load, end of utterance, time to first audio (2026-09-26)

Measured on an Apple M4 Max, macOS 26.6.2, release build,
`metal` on. Models from the Hugging Face cache (Kyutai `stt-1b-en_fr-candle`,
pocket-tts `b6369a24`), Aurora voice
`~/.local/share/noema/models/aurora-voices/aurora.safetensors`. Tags:
**[V]** measured or read in source · **[I]** inference.

## Method

- **Utterances**: three WAV fixtures in `crates/simply-voice/tests/fixtures/`
  (16 kHz mono PCM16): `run-the-tests.wav` and `two-utterances.wav` (macOS
  `say -v Samantha`; the second is two sentences with 3 s of silence between),
  and `aurora-loopback.wav` (`sv say --out` in the Aurora voice, resampled —
  our TTS read back through our STT). Not a human voice and not the room: the
  mic path was not measured (see the end).
- **End of utterance → Final**: `sv listen --events --wav <f>` feeds the file in 20 ms
  frames on the wall clock, then 4 s of silence. The detector thread stamps
  the last frame whose RMS exceeds 0.01 (the energy VAD's own threshold); the
  number is the time from that frame being fed to the `Final` event. It
  therefore includes trailing breath/decay above the threshold, and measures
  the detector's decision, not the pipeline after it.
- **Push-to-talk**: `sv listen --events --ptt --wav <f>` sends the release the moment
  the file ends.
- **TTS time to first audio**: `sv say "<text>"` — from queueing the sentence
  on the voice worker to its `PlaybackStarted` (first chunk appended to the
  playing rodio sink). `sv say --out` times whole-sentence `synthesize()`.
- **Load**: wall time of `KyutaiSttProvider::new` + opening a stream, and of
  `PocketTtsProvider::new` + `warm_voice`. "Cold" is the first run after a
  build (disk and Metal caches cold), "warm" every later run.

## Results

### Load [V]

| Model | Cold | Warm (runs) |
|---|---|---|
| Kyutai STT 1B | 2998 ms | 307–326 ms (8) |
| pocket-tts + Aurora voice | 8764 ms | 83–100 ms (4) |
| pocket-tts + `alba` | — | 131 ms (1) |

### End of utterance → Final [V]

Kyutai streaming, the backend's own end of turn (default `with_end_of_turn`:
2 s horizon, p > 0.6, 3 steps) plus this adapter's 500 ms hold (below):

| Fixture | Final | ms |
|---|---|---|
| aurora-loopback | can you check why the build is failing? | 1198 |
| run-the-tests | run the tests and tell me what failed. | 1321 |
| two-utterances (1) | open the config file. | 1779 |
| two-utterances (2) | Then run Cargo Clippy. | 1638 |

Tuning, same fixtures, in order:

| `[stt]` | ms |
|---|---|
| `eot_horizon = 1` | 1196 · 1319 · 1218 · 1659 |
| `eot_horizon = 0` | 1204 · 1316 · 1197 · 1261 |
| `eot_horizon = 1`, `eot_consecutive = 2` | 1100 · 1040 · 1140 · 1182 |

Push-to-talk release (forced end: 1 s of silence through the model, stream
closed and reopened): **309 ms** (run-the-tests), **411 ms** (aurora-loopback).

Energy VAD + batch (`stt.backend = "kyutai-batch"`: 500 ms silence cut, then
Kyutai `transcribe`, which pads 1.5 s): 1350 · 1336 · 1228 · 1587 ms. No partials.

### TTS [V]

| Text | Streamed: first audio | Whole sentence: synth time (RTF) |
|---|---|---|
| "Done." | 33 ms | — |
| "Run the tests and tell me what failed." | 48 ms | 1277 ms cold / 461–940 ms warm for 2.1–4.2 s of audio (RTF 0.20–0.25) |
| a 21-word sentence | 46 ms | — |

Streaming (`TtsProvider::stream`, one Mimi frame per chunk, appended as it
arrives) is why the worker streams: first audio in ~40 ms instead of 0.5–1 s.
"Run the tests…" finished playing 2704 ms after it was queued, against
~2640 ms of audio in a separate synthesis of the same text, so generation
keeps ahead of playback at RTF ≈ 0.2 [V on timings; gaps by ear not checked].

## Findings

1. **Kyutai calls end of turn before the last word is out** [V]. Its
   end-of-turn head reads the audio as it arrives; words come out 0.5 s later
   (the ASR delay). Taken at face value, every fixture split in two:
   `run the tests and tell me what` / `failed.`,
   `can you check why the build is` / `failing?`. The adapter now treats
   Kyutai's final as a *signal*, waits 500 ms of further audio, and finalises
   its own word list — so the numbers above include that hold. The right fix
   is in simply-voice: fire `EndOfTurn` only after `asr_delay_in_tokens` more
   steps, collecting the words that arrive meanwhile. Then the adapter's hold
   goes and ~500 ms with it [I].
2. **Kyutai streams share state across sessions** [V]. `KyutaiSttProvider`
   never calls moshi's `asr::State::reset()` between `stream()`s (or
   `transcribe()`s). Fed one after another through one model, the fixtures
   transcribed differently ("cargo clippy" → "cargo" + "clicky", split into
   two turns); a fresh model each time passes. Affects every session in which a
   push-to-talk release reopens the stream. Fix in simply-voice: reset (and
   clear the end-of-turn streak) at the start of each session.
3. **The energy VAD runs on wall-clock time** [V]. `VoiceActivityDetector`
   measures its 200 ms / 500 ms windows with `Instant::now()`, so a WAV must be
   paced in real time, and it has no flush, so push-to-talk keeps its own copy
   of the audio. On 20 ms frames a dip between words sends it back to silence;
   it confirmed speech over a second late on one fixture, and a 400 ms
   pre-roll turned "Open the config file." into "file." (pre-roll now 1.5 s).
   Fix in simply-voice: count samples, not time, and add `flush()`.
4. **Tuning is worth ~150 ms, not more** [V]. `eot_horizon = 1` with
   `eot_consecutive = 2` was fastest (1040–1182 ms) with no false splits on
   these fixtures; they are TTS voices with clean pauses, so a mid-thought
   pause in real dictation is untested. Defaults left as they are.
5. **TTS stream chunks carry no sample rate** [V]. `TtsProvider::stream`
   returns bare PCM16 `AudioChunk`s; the worker is told 24 kHz from
   `PocketTtsProvider::sample_rate()`. A stream-level format in the trait
   would remove that side channel.

## Not measured

- **The mic.** `sv listen` opened "MacBook Pro Microphone" and ran 8 s with no
  events in a quiet room; whether macOS granted the terminal mic access (it
  delivers silence when it has not) was not established. All numbers above
  are WAV-fed.
- Long sessions: Kyutai's transformer is configured with
  `max_seq_len = 16384` steps (≈ 22 min at 12.5 Hz) and `context = 750`
  (60 s); what an always-listening stream does past 22 minutes is unknown [I].
- Whisper (`--features whisper`) was built into neither the binary nor the
  tests.
