//! Kyutai streaming STT on recorded fixtures. These load the real model (from
//! the Hugging Face cache; the first run downloads ~2 GB), so they only build
//! with `--features kyutai`; add `metal` on Apple Silicon.
//!
//! Fixtures (`tests/fixtures/`, 16 kHz mono PCM16, from voice-agent):
//! - `run-the-tests.wav`, `two-utterances.wav` — macOS `say -v Samantha`; the
//!   second is two sentences with 3 s of silence between them.
//! - `aurora-loopback.wav` — Aurora's pocket-tts voice read back.
//!
//! Run with `--nocapture` to see end-of-speech → final latency per utterance:
//! the audio is fed in 20 ms frames on the wall clock, and latency is the time
//! from feeding the last frame whose RMS exceeds 0.01 to the final arriving.
#![cfg(feature = "kyutai")]

use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use simply_voice::{AudioChunk, KyutaiSttProvider, SttProvider};

const SAMPLE_RATE: usize = simply_voice::audio::SAMPLE_RATE as usize;
/// 20 ms of PCM16 bytes.
const FRAME_BYTES: usize = SAMPLE_RATE / 50 * 2;

const RUN_THE_TESTS: (&str, &[&str]) =
    ("run-the-tests.wav", &["run the tests and tell me what failed"]);
const AURORA_LOOPBACK: (&str, &[&str]) =
    ("aurora-loopback.wav", &["can you check why the build is failing"]);
const TWO_UTTERANCES: (&str, &[&str]) =
    ("two-utterances.wav", &["open the config file", "then run cargo clippy"]);

/// A freshly loaded model, and a lock so one test holds the GPU at a time
/// (keeps the wall-clock pacing honest).
fn kyutai() -> (MutexGuard<'static, ()>, KyutaiSttProvider) {
    static LOCK: Mutex<()> = Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    (guard, KyutaiSttProvider::new(None, false).expect("load Kyutai"))
}

/// The PCM16 bytes of a fixture's `data` chunk.
fn fixture(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    let wav = std::fs::read(&path).unwrap();
    let mut at = 12;
    while at + 8 <= wav.len() {
        let len = u32::from_le_bytes(wav[at + 4..at + 8].try_into().unwrap()) as usize;
        if &wav[at..at + 4] == b"data" {
            return wav[at + 8..at + 8 + len].to_vec();
        }
        at += 8 + len + len % 2;
    }
    panic!("{name}: no data chunk");
}

/// Lowercase words without punctuation, for comparing transcripts.
fn norm(text: &str) -> String {
    text.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn loud(frame: &[u8]) -> bool {
    let samples = frame.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f64 / 32768.0);
    let (sum, n) = samples.fold((0.0, 0usize), |(s, n), x| (s + x * x, n + 1));
    n > 0 && (sum / n as f64).sqrt() > 0.01
}

/// One final: its normalised text and, when paced, end-of-speech → final.
struct Final {
    text: String,
    latency: Option<Duration>,
}

/// Stream `pcm` then `tail_ms` of silence through one `stream()` in 20 ms
/// frames (on the wall clock if `paced`), drop the sender, and return every
/// final.
async fn stream(provider: &KyutaiSttProvider, pcm: &[u8], tail_ms: usize, paced: bool) -> Vec<Final> {
    let (tx, mut rx) = provider.stream().await.unwrap();
    let loud_at: Arc<Mutex<Vec<Instant>>> = Arc::default();
    let feeder = {
        let loud_at = Arc::clone(&loud_at);
        let mut audio = pcm.to_vec();
        audio.resize(pcm.len() + SAMPLE_RATE * 2 * tail_ms / 1000, 0);
        tokio::spawn(async move {
            let start = tokio::time::Instant::now();
            for (i, frame) in audio.chunks(FRAME_BYTES).enumerate() {
                if loud(frame) {
                    loud_at.lock().unwrap().push(Instant::now());
                }
                tx.send(AudioChunk::new(frame.to_vec())).await.unwrap();
                if paced {
                    tokio::time::sleep_until(start + Duration::from_millis(20 * (i as u64 + 1))).await;
                }
            }
        })
    };
    let mut finals = Vec::new();
    while let Some(t) = rx.recv().await {
        if t.is_final {
            let now = Instant::now();
            let spoke = loud_at.lock().unwrap().iter().rev().find(|&&t| t <= now).copied();
            finals.push(Final {
                text: norm(&t.text),
                latency: paced.then(|| spoke.map(|s| now - s)).flatten(),
            });
        }
    }
    feeder.await.unwrap();
    finals
}

fn texts(finals: &[Final]) -> Vec<&str> {
    finals.iter().map(|f| f.text.as_str()).collect()
}

/// Streams and batch transcriptions through ONE provider must not leak into
/// each other: each call starts from a clean model. Before the reset, fixtures
/// fed one after another came out as "cargo" + "clicky" in split turns.
#[tokio::test(flavor = "multi_thread")]
async fn one_provider_transcribes_each_call_from_a_clean_state() {
    let (_gpu, kyutai) = kyutai();
    let first = stream(&kyutai, &fixture(RUN_THE_TESTS.0), 3000, false).await;
    assert_eq!(texts(&first), RUN_THE_TESTS.1);

    let batch = kyutai.transcribe(AudioChunk::new(fixture(AURORA_LOOPBACK.0))).await.unwrap();
    assert_eq!(norm(&batch.text), AURORA_LOOPBACK.1[0]);

    for (name, expected) in [TWO_UTTERANCES, AURORA_LOOPBACK] {
        let finals = stream(&kyutai, &fixture(name), 3000, false).await;
        assert_eq!(texts(&finals), expected, "{name}");
    }

    let batch = kyutai.transcribe(AudioChunk::new(fixture(TWO_UTTERANCES.0))).await.unwrap();
    assert_eq!(norm(&batch.text), TWO_UTTERANCES.1.join(" "));

    let again = stream(&kyutai, &fixture(RUN_THE_TESTS.0), 3000, false).await;
    assert_eq!(texts(&again), texts(&first), "the same audio twice, the same transcript");
}

/// End of turn waits out the ASR delay, so an utterance's last word lands in
/// its own final rather than starting the next turn ("…tell me what" /
/// "failed."). Paced on the wall clock; prints end-of-speech → final.
#[tokio::test(flavor = "multi_thread")]
async fn end_of_turn_keeps_the_last_word() {
    let (_gpu, kyutai) = kyutai();
    for (name, expected) in [RUN_THE_TESTS, AURORA_LOOPBACK, TWO_UTTERANCES] {
        let finals = stream(&kyutai, &fixture(name), 3000, true).await;
        for f in &finals {
            eprintln!("{name}: {:?} end of speech → final {:?}", f.text, f.latency);
        }
        assert_eq!(texts(&finals), expected, "{name}");
    }
}

/// Input that stops mid-turn (push-to-talk release, no trailing silence) still
/// yields the whole utterance: closing the stream flushes the ASR delay.
#[tokio::test(flavor = "multi_thread")]
async fn closing_the_stream_flushes_the_last_word() {
    let (_gpu, kyutai) = kyutai();
    let finals = stream(&kyutai, &fixture(RUN_THE_TESTS.0), 0, false).await;
    assert_eq!(texts(&finals), RUN_THE_TESTS.1);
}
