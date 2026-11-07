//! Turn detectors fed recorded WAV fixtures, paced in real time, through the
//! detector thread — the same path `sv listen --wav` takes.
//!
//! Fixtures (the library's `crates/simply-voice/tests/fixtures/`, 16 kHz mono PCM16):
//! - `run-the-tests.wav`, `two-utterances.wav` — macOS `say -v Samantha`;
//!   the second is two sentences with 3 s of silence between them (sox).
//! - `aurora-loopback.wav` — `sv say --out` in the Aurora voice, resampled
//!   to 16 kHz: our own TTS read back through our own STT.
//!
//! These load the real Kyutai model (from the Hugging Face cache; the first
//! run downloads it). One test at a time holds the model.

use std::path::Path;
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use simply_voice::KyutaiSttProvider;
use simply_voice_cli::audio::{self, SAMPLE_RATE};
use simply_voice_cli::turn::{self, KyutaiTurn, TurnDetector, TurnEvent, TurnInput, VadTurn};

/// A freshly loaded Kyutai model, and a lock so one test uses the GPU at a
/// time (keeps real-time pacing honest). Fresh per case because
/// `KyutaiSttProvider` never resets moshi's streaming state between
/// sessions: fed one after another through one model, the fixtures
/// transcribed differently ("cargo clippy" became "cargo" + "clicky").
fn kyutai() -> (std::sync::MutexGuard<'static, ()>, Arc<KyutaiSttProvider>) {
    static LOCK: Mutex<()> = Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    (
        guard,
        Arc::new(KyutaiSttProvider::new(None, false).expect("load Kyutai")),
    )
}

fn fixture(name: &str) -> Vec<i16> {
    audio::read_wav(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../simply-voice/tests/fixtures")
            .join(name),
    )
    .unwrap()
}

/// Lowercase words without punctuation, for comparing transcripts.
fn norm(text: &str) -> String {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Feed `pcm` then `tail_ms` of silence in 20 ms frames on the wall clock
/// (optionally releasing push-to-talk where the audio ends); return every
/// event, normalised.
fn run(detector: Box<dyn TurnDetector>, pcm: &[i16], ptt: bool, tail_ms: u64) -> Vec<TurnEvent> {
    const FRAME: usize = SAMPLE_RATE as usize / 50;
    let (tx, rx) = std_mpsc::channel();
    let mut heard = turn::spawn(detector, rx);
    let silence = vec![0i16; SAMPLE_RATE as usize * tail_ms as usize / 1000];
    let start = Instant::now();
    for (i, frame) in pcm.chunks(FRAME).enumerate() {
        tx.send(TurnInput::Frame(frame.to_vec())).unwrap();
        std::thread::sleep(
            (start + Duration::from_millis(20 * (i as u64 + 1)))
                .saturating_duration_since(Instant::now()),
        );
    }
    if ptt {
        tx.send(TurnInput::EndUtterance).unwrap();
    }
    for frame in silence.chunks(FRAME) {
        tx.send(TurnInput::Frame(frame.to_vec())).unwrap();
        std::thread::sleep(Duration::from_millis(20));
    }
    drop(tx);
    let mut events = Vec::new();
    while let Some(h) = heard.blocking_recv() {
        events.push(match h.event {
            TurnEvent::Partial(p) => TurnEvent::Partial(norm(&p)),
            TurnEvent::Final(f) => TurnEvent::Final(norm(&f)),
            e => e,
        });
    }
    events
}

fn finals(events: &[TurnEvent]) -> Vec<&str> {
    events
        .iter()
        .filter_map(|e| match e {
            TurnEvent::Final(f) => Some(f.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn kyutai_calls_end_of_turn_once_per_utterance_with_the_last_word() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let cases: &[(&str, &[&str])] = &[
        (
            "run-the-tests.wav",
            &["run the tests and tell me what failed"],
        ),
        (
            "aurora-loopback.wav",
            &["can you check why the build is failing"],
        ),
        (
            "two-utterances.wav",
            &["open the config file", "then run cargo clippy"],
        ),
    ];
    for (name, expected) in cases {
        let (_guard, model) = kyutai();
        let detector = KyutaiTurn::new(model, rt.handle().clone()).unwrap();
        let events = run(Box::new(detector), &fixture(name), false, 3000);
        assert_eq!(finals(&events), *expected, "{name}: {events:?}");
        // Each utterance: SpeechStart, then cumulative partials, then Final.
        let starts = events
            .iter()
            .filter(|e| **e == TurnEvent::SpeechStart)
            .count();
        assert_eq!(starts, expected.len(), "{name}: {events:?}");
        assert_eq!(events[0], TurnEvent::SpeechStart, "{name}");
    }
}

#[test]
fn kyutai_push_to_talk_release_forces_the_final() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (_guard, model) = kyutai();
    let detector = KyutaiTurn::new(model, rt.handle().clone()).unwrap();
    // No tail before the release: only the forced end can finish this turn.
    let events = run(Box::new(detector), &fixture("run-the-tests.wav"), true, 0);
    assert_eq!(
        finals(&events),
        ["run the tests and tell me what failed"],
        "{events:?}"
    );
}

#[test]
fn energy_vad_with_a_batch_backend_cuts_each_utterance() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (_guard, model) = kyutai();
    // Kyutai's batch `transcribe` stands in for Whisper/Parakeet.
    let detector = VadTurn::new(model, rt.handle().clone());
    let events = run(
        Box::new(detector),
        &fixture("two-utterances.wav"),
        false,
        1500,
    );
    assert_eq!(
        finals(&events),
        ["open the config file", "then run cargo clippy"],
        "{events:?}"
    );
    assert!(
        !events.iter().any(|e| matches!(e, TurnEvent::Partial(_))),
        "a batch backend has no partials"
    );
}
