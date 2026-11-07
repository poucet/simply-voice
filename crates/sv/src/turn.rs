//! End of utterance — the one thing that differs by STT backend.
//!
//! No new STT trait: backends are `simply_voice::SttProvider`s. What differs
//! is who decides the turn is over, so that is the only abstraction here:
//!
//! - [`KyutaiTurn`] — a streaming backend with its own semantic end-of-turn
//!   (Kyutai's VAD heads); a thin adapter, the backend decides.
//! - [`VadTurn`] — any batch backend (Whisper today, Parakeet later) behind
//!   `simply_voice`'s energy VAD, which cuts the utterance and hands it over
//!   whole.
//!
//! Detectors block (model steps, batch transcription), so they run on a
//! dedicated thread: [`spawn`] feeds one from a channel of [`TurnInput`]s and
//! returns the [`Heard`] events.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use simply_voice::{
    AudioChunk, KyutaiSttProvider, SttProvider, Transcription, VadEvent, VoiceActivityDetector,
};
use tokio::runtime::Handle;
use tokio::sync::mpsc;

use crate::audio::{self, SAMPLE_RATE, VOICED_RMS};

/// What a detector reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnEvent {
    SpeechStart,
    /// The utterance so far (cumulative, not one word).
    Partial(String),
    /// The finished utterance.
    Final(String),
}

/// Decides where an utterance ends.
pub trait TurnDetector: Send {
    /// Feed one frame of mono PCM16 at [`SAMPLE_RATE`]. An empty frame is a
    /// poll: it surfaces events a backend produced asynchronously without
    /// advancing any timing. At most one event per call; call again with an
    /// empty frame until `None` to drain.
    fn feed(&mut self, frame: &[i16]) -> Option<TurnEvent>;

    /// End the utterance now — a push-to-talk release. Returns the `Final`
    /// of what was heard since the last one, if anything was.
    fn end_utterance(&mut self) -> Option<TurnEvent>;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// `[stt]` config section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SttConfig {
    pub backend: SttBackend,
    /// Hugging Face repo for Kyutai; unset = `kyutai/stt-1b-en_fr-candle`.
    pub kyutai_repo: Option<String>,
    /// Kyutai end-of-turn tuning, passed to `with_end_of_turn`; unset keeps
    /// simply-voice's defaults (horizon 2 = 2 s window, threshold 0.6, 3 steps).
    pub eot_horizon: Option<usize>,
    pub eot_threshold: Option<f32>,
    pub eot_consecutive: Option<u32>,
    /// ggml model for `backend = "whisper"` (needs the `whisper` feature).
    pub whisper_model: Option<String>,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            backend: SttBackend::Kyutai,
            kyutai_repo: None,
            eot_horizon: None,
            eot_threshold: None,
            eot_consecutive: None,
            whisper_model: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum SttBackend {
    /// Kyutai streaming STT; Kyutai's semantic end-of-turn decides.
    Kyutai,
    /// Kyutai run as a batch backend behind the energy VAD (for A/B).
    KyutaiBatch,
    /// whisper.cpp behind the energy VAD.
    Whisper,
}

/// Load the configured backend and wrap it in its detector. Blocking (model
/// load); call from a blocking context. `rt` hosts the providers' tasks.
pub fn load(cfg: &SttConfig, rt: Handle) -> Result<Box<dyn TurnDetector>> {
    Ok(match cfg.backend {
        SttBackend::Kyutai => Box::new(KyutaiTurn::new(load_kyutai(cfg)?, rt)?),
        SttBackend::KyutaiBatch => Box::new(VadTurn::new(load_kyutai(cfg)?, rt)),
        SttBackend::Whisper => Box::new(VadTurn::new(load_whisper(cfg)?, rt)),
    })
}

fn load_kyutai(cfg: &SttConfig) -> Result<Arc<KyutaiSttProvider>> {
    let provider = KyutaiSttProvider::new(cfg.kyutai_repo.clone(), false)
        .context("loading Kyutai STT")?
        .with_end_of_turn(cfg.eot_horizon, cfg.eot_threshold, cfg.eot_consecutive);
    Ok(Arc::new(provider))
}

#[cfg(feature = "whisper")]
fn load_whisper(cfg: &SttConfig) -> Result<Arc<dyn SttProvider>> {
    let Some(path) = &cfg.whisper_model else {
        bail!("stt.backend = \"whisper\" needs stt.whisper_model")
    };
    Ok(Arc::new(simply_voice::WhisperProvider::new(
        crate::config::expand_home(path),
    )?))
}

#[cfg(not(feature = "whisper"))]
fn load_whisper(_: &SttConfig) -> Result<Arc<dyn SttProvider>> {
    bail!("stt.backend = \"whisper\" needs sv built with `--features whisper`")
}

// ---------------------------------------------------------------------------
// Kyutai: streaming, the backend decides
// ---------------------------------------------------------------------------

/// Silence pushed through Kyutai on a forced end of utterance: its 0.5 s ASR
/// delay plus margin, so words still held back by the delay come out.
const KYUTAI_FLUSH: Duration = Duration::from_millis(1000);

/// How much audio to wait, after Kyutai calls end of turn, before the turn is
/// final. Kyutai's end-of-turn head reads the audio *now*, but its words come
/// out 0.5 s late (the ASR delay), so the last word of an utterance routinely
/// arrives just after the end of turn that should have included it.
/// Measured on the fixtures: without this hold "…tell me what failed" came
/// out as two turns, "…tell me what" and "failed."
const KYUTAI_HOLD: Duration = Duration::from_millis(500);

/// Adapter over `KyutaiSttProvider::stream()`. Kyutai emits one word at a
/// time (`is_final = false`) and, when its end-of-turn head fires, the words
/// since its last end of turn (`is_final = true`). This keeps its own word
/// list and uses Kyutai's finals only as the end-of-turn *signal*: the turn
/// becomes `Final` [`KYUTAI_HOLD`] of audio later, with any straggling words
/// included. A forced end (push-to-talk) flushes the stream with silence,
/// closes it, and reopens a fresh one — the model stays loaded.
pub struct KyutaiTurn {
    provider: Arc<KyutaiSttProvider>,
    rt: Handle,
    audio_tx: Option<mpsc::Sender<AudioChunk>>,
    text_rx: mpsc::Receiver<Transcription>,
    /// Words of the utterance in progress.
    words: Vec<String>,
    /// Samples fed so far — the clock for the hold.
    fed: u64,
    /// End of turn was called; final once `fed` reaches this.
    hold_until: Option<u64>,
    pending: VecDeque<TurnEvent>,
}

impl KyutaiTurn {
    pub fn new(provider: Arc<KyutaiSttProvider>, rt: Handle) -> Result<Self> {
        let (audio_tx, text_rx) = rt.block_on(provider.stream())?;
        Ok(Self {
            provider,
            rt,
            audio_tx: Some(audio_tx),
            text_rx,
            words: Vec::new(),
            fed: 0,
            hold_until: None,
            pending: VecDeque::new(),
        })
    }

    fn reopen(&mut self) -> Result<()> {
        let (audio_tx, text_rx) = self.rt.block_on(self.provider.stream())?;
        self.audio_tx = Some(audio_tx);
        self.text_rx = text_rx;
        Ok(())
    }

    fn send(&mut self, frame: &[i16]) {
        let sent = self.audio_tx.as_ref().is_some_and(|tx| {
            tx.blocking_send(AudioChunk::new(audio::pcm16_bytes(frame)))
                .is_ok()
        });
        if !sent {
            tracing::error!("kyutai stream ended; reopening");
            if let Err(e) = self.reopen() {
                tracing::error!("kyutai reopen failed: {e:#}");
            }
        }
        self.fed += frame.len() as u64;
    }

    fn absorb(&mut self, t: Transcription) {
        if !t.is_final {
            if self.words.is_empty() {
                self.pending.push_back(TurnEvent::SpeechStart);
            }
            self.words.push(t.text);
            self.pending
                .push_back(TurnEvent::Partial(self.words.join(" ")));
        } else if !self.words.is_empty() && self.hold_until.is_none() {
            // Kyutai's final text is ignored: after a hold it can be words we
            // already finalised.
            let hold = SAMPLE_RATE as u64 * KYUTAI_HOLD.as_millis() as u64 / 1000;
            self.hold_until = Some(self.fed + hold);
        }
    }

    fn finalise(&mut self) {
        self.hold_until = None;
        if !self.words.is_empty() {
            self.pending
                .push_back(TurnEvent::Final(self.words.join(" ")));
            self.words.clear();
        }
    }
}

impl TurnDetector for KyutaiTurn {
    fn feed(&mut self, frame: &[i16]) -> Option<TurnEvent> {
        if !frame.is_empty() {
            self.send(frame);
        }
        while let Ok(t) = self.text_rx.try_recv() {
            self.absorb(t);
        }
        if self.hold_until.is_some_and(|until| self.fed >= until) {
            self.finalise();
        }
        self.pending.pop_front()
    }

    fn end_utterance(&mut self) -> Option<TurnEvent> {
        if let Some(tx) = self.audio_tx.take() {
            let silence =
                vec![0i16; (SAMPLE_RATE as u128 * KYUTAI_FLUSH.as_millis() / 1000) as usize];
            let _ = tx.blocking_send(AudioChunk::new(audio::pcm16_bytes(&silence)));
            // Dropping the sender ends the stream after these samples.
        }
        while let Some(t) = self.text_rx.blocking_recv() {
            self.absorb(t);
        }
        self.finalise();
        if let Err(e) = self.reopen() {
            tracing::error!("kyutai reopen failed: {e:#}");
        }
        // One push-to-talk hold is one utterance: fold every Final not yet
        // delivered into one.
        let text: Vec<String> = self
            .pending
            .drain(..)
            .filter_map(|e| match e {
                TurnEvent::Final(t) => Some(t),
                _ => None,
            })
            .collect();
        (!text.is_empty()).then(|| TurnEvent::Final(text.join(" ")))
    }
}

// ---------------------------------------------------------------------------
// Energy VAD + batch STT
// ---------------------------------------------------------------------------

/// Audio kept from before the VAD confirms speech, so the start of the
/// utterance reaches the transcriber. The VAD wants 200 ms of unbroken
/// energy, and a dip between words sends it back to silence, so on 20 ms
/// frames it can confirm a second or more late: on the two-utterance fixture
/// a 400 ms pre-roll turned "Open the config file." into "file.".
const PREROLL: Duration = Duration::from_millis(1500);

/// Adapter for batch backends: `simply_voice`'s energy VAD cuts utterances
/// (500 ms of silence ends one) and the whole utterance is transcribed at
/// once. No partials — a batch backend has none.
///
/// The VAD measures its durations in wall-clock time, so frames must arrive
/// at the rate they were captured (a WAV must be paced, not blasted).
pub struct VadTurn {
    vad: VoiceActivityDetector,
    stt: Arc<dyn SttProvider>,
    rt: Handle,
    preroll: VecDeque<i16>,
    utterance: Vec<i16>,
    in_speech: bool,
}

impl VadTurn {
    pub fn new(stt: Arc<dyn SttProvider>, rt: Handle) -> Self {
        Self {
            vad: VoiceActivityDetector::new(),
            stt,
            rt,
            preroll: VecDeque::new(),
            utterance: Vec::new(),
            in_speech: false,
        }
    }

    fn finish(&mut self) -> Option<TurnEvent> {
        self.in_speech = false;
        let pcm = std::mem::take(&mut self.utterance);
        let chunk = AudioChunk::new(audio::pcm16_bytes(&pcm));
        match self.rt.block_on(self.stt.transcribe(chunk)) {
            Ok(t) if !t.text.trim().is_empty() => Some(TurnEvent::Final(t.text.trim().to_string())),
            Ok(_) => None,
            Err(e) => {
                tracing::error!("batch transcription failed: {e:#}");
                None
            }
        }
    }
}

impl TurnDetector for VadTurn {
    fn feed(&mut self, frame: &[i16]) -> Option<TurnEvent> {
        if frame.is_empty() {
            return None;
        }
        let event = self.vad.process(frame);
        if self.in_speech {
            self.utterance.extend_from_slice(frame);
        } else {
            self.preroll.extend(frame);
            let keep = (SAMPLE_RATE as u128 * PREROLL.as_millis() / 1000) as usize;
            let excess = self.preroll.len().saturating_sub(keep);
            self.preroll.drain(..excess);
        }
        match event {
            Some(VadEvent::SpeechStart) => {
                self.in_speech = true;
                self.utterance = self.preroll.drain(..).collect();
                Some(TurnEvent::SpeechStart)
            }
            Some(VadEvent::SpeechEnd(_)) => self.finish(),
            _ => None,
        }
    }

    fn end_utterance(&mut self) -> Option<TurnEvent> {
        if !self.in_speech {
            return None;
        }
        self.vad = VoiceActivityDetector::new();
        self.finish()
    }
}

// ---------------------------------------------------------------------------
// The detector thread
// ---------------------------------------------------------------------------

/// Input to a running detector.
#[derive(Debug)]
pub enum TurnInput {
    Frame(Vec<i16>),
    /// Push-to-talk release: end the utterance now.
    EndUtterance,
}

/// A detector event with its timing.
#[derive(Debug, Clone)]
pub struct Heard {
    pub event: TurnEvent,
    /// For a `Final`: time from the last voiced frame fed (RMS above
    /// [`VOICED_RMS`]) to the `Final` — the end-of-utterance latency.
    pub since_voice: Option<Duration>,
}

/// How often an idle detector is polled, so events a backend produces after
/// the mic went quiet (or closed) still surface.
const POLL: Duration = Duration::from_millis(20);

/// Run `detector` on its own thread. It ends when `input`'s senders are all
/// dropped or the returned receiver is.
pub fn spawn(
    mut detector: Box<dyn TurnDetector>,
    input: std_mpsc::Receiver<TurnInput>,
) -> mpsc::UnboundedReceiver<Heard> {
    let (tx, rx) = mpsc::unbounded_channel();
    std::thread::Builder::new()
        .name("turn-detector".into())
        .spawn(move || {
            let mut last_voice: Option<Instant> = None;
            loop {
                let first = match input.recv_timeout(POLL) {
                    Ok(TurnInput::Frame(frame)) => {
                        if audio::rms(&frame) > VOICED_RMS {
                            last_voice = Some(Instant::now());
                        }
                        detector.feed(&frame)
                    }
                    Ok(TurnInput::EndUtterance) => detector.end_utterance(),
                    Err(std_mpsc::RecvTimeoutError::Timeout) => detector.feed(&[]),
                    Err(std_mpsc::RecvTimeoutError::Disconnected) => return,
                };
                let events = std::iter::successors(first, |_| detector.feed(&[]));
                for event in events {
                    let since_voice = match event {
                        TurnEvent::Final(_) => last_voice.map(|t| t.elapsed()),
                        _ => None,
                    };
                    if tx.send(Heard { event, since_voice }).is_err() {
                        return;
                    }
                }
            }
        })
        .expect("spawn turn-detector thread");
    rx
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    /// A batch STT that reports how much audio it was handed.
    struct CountingStt;

    #[async_trait]
    impl SttProvider for CountingStt {
        async fn transcribe(&self, audio: AudioChunk) -> Result<Transcription> {
            Ok(Transcription {
                text: format!("{} samples", audio.data.len() / 2),
                is_final: true,
            })
        }
        async fn stream(
            &self,
        ) -> Result<(mpsc::Sender<AudioChunk>, mpsc::Receiver<Transcription>)> {
            unimplemented!()
        }
    }

    fn tone(ms: u64) -> Vec<i16> {
        (0..SAMPLE_RATE as u64 * ms / 1000)
            .map(|i| ((i as f32 * 0.2).sin() * 8000.0) as i16)
            .collect()
    }

    #[test]
    fn vad_turn_force_final_on_release_and_ignores_polls() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut d = VadTurn::new(Arc::new(CountingStt), rt.handle().clone());
        assert_eq!(d.end_utterance(), None, "nothing heard, nothing final");
        let mut events = Vec::new();
        for frame in tone(400).chunks(320) {
            events.extend(d.feed(frame));
            assert_eq!(d.feed(&[]), None, "a poll never advances the VAD");
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(events, vec![TurnEvent::SpeechStart]);
        match d.end_utterance() {
            Some(TurnEvent::Final(t)) => {
                let n: usize = t.split(' ').next().unwrap().parse().unwrap();
                assert!(
                    n >= 6000,
                    "pre-roll and speech reach the transcriber, got {n}"
                );
            }
            other => panic!("expected a Final, got {other:?}"),
        }
        assert_eq!(d.end_utterance(), None);
    }
}
