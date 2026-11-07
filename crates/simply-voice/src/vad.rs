//! Voice activity detection — energy-based state machine.
//!
//! Determines when the user is speaking by tracking audio energy levels.
//! Emits state transitions that drive the voice pipeline and client UI.

use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// VAD state transitions.
#[derive(Debug, Clone)]
pub enum VadEvent {
    /// Speech started — user began talking.
    SpeechStart,
    /// Intermediate audio chunk during active speech.
    SpeechChunk(Vec<i16>),
    /// Speech ended — complete utterance ready for processing.
    SpeechEnd(Vec<i16>),
}

#[derive(Debug, Clone, PartialEq)]
enum State {
    Silence,
    PossibleSpeech,
    Speech,
    PossibleSilence,
}

#[derive(Clone)]
pub struct VoiceActivityDetector {
    energy_threshold: f32,
    silence_duration_ms: u64,
    speech_duration_ms: u64,
    /// Quiet this long ends a `PossibleSpeech` onset. Shorter dips (the gap
    /// between two words) keep it, so the first word is not thrown away.
    onset_dip_ms: u64,
    /// Audio kept from before an onset, so a soft start (an "h" or "s"
    /// below the threshold) reaches STT.
    pre_roll_ms: u64,
    state: State,
    state_start: Instant,
    /// Last chunk above the threshold, for `onset_dip_ms`.
    last_speech: Instant,
    accumulated: Vec<i16>,
    /// Recent chunks while silent, oldest first, each with its arrival time.
    pre_roll: VecDeque<(Instant, Vec<i16>)>,
}

impl VoiceActivityDetector {
    pub fn new() -> Self {
        Self {
            energy_threshold: 0.01,
            silence_duration_ms: 500,
            speech_duration_ms: 200,
            onset_dip_ms: 150,
            pre_roll_ms: 300,
            state: State::Silence,
            state_start: Instant::now(),
            last_speech: Instant::now(),
            accumulated: Vec::new(),
            pre_roll: VecDeque::new(),
        }
    }

    /// Process a chunk of PCM16 samples. Returns a VAD event if a state transition occurred.
    pub fn process(&mut self, samples: &[i16]) -> Option<VadEvent> {
        let energy = Self::energy(samples);
        let is_speech = energy > self.energy_threshold;
        let now = Instant::now();
        let elapsed = now.duration_since(self.state_start);

        match self.state {
            State::Silence => {
                self.push_pre_roll(now, samples.to_vec());
                if is_speech {
                    debug!("VAD: Silence -> PossibleSpeech (energy: {energy:.4})");
                    self.transition(State::PossibleSpeech, now);
                    self.last_speech = now;
                    // The onset starts with the pre-roll, this chunk included.
                    self.accumulated = self.pre_roll.drain(..).flat_map(|(_, c)| c).collect();
                }
                None
            }
            State::PossibleSpeech => {
                self.accumulated.extend_from_slice(samples);
                if is_speech {
                    self.last_speech = now;
                }

                if is_speech && elapsed.as_millis() >= self.speech_duration_ms as u128 {
                    info!("VAD: PossibleSpeech -> Speech");
                    self.transition(State::Speech, now);
                    Some(VadEvent::SpeechStart)
                } else if !is_speech
                    && now.duration_since(self.last_speech).as_millis() >= self.onset_dip_ms as u128
                {
                    debug!("VAD: PossibleSpeech -> Silence (false positive)");
                    self.transition(State::Silence, now);
                    // Keep the abandoned onset as pre-roll: if speech resumes
                    // within `pre_roll_ms`, it is the start of the utterance.
                    let onset = std::mem::take(&mut self.accumulated);
                    self.push_pre_roll(now, onset);
                    None
                } else {
                    None
                }
            }
            State::Speech => {
                self.accumulated.extend_from_slice(samples);

                if !is_speech {
                    debug!("VAD: Speech -> PossibleSilence");
                    self.transition(State::PossibleSilence, now);
                }
                Some(VadEvent::SpeechChunk(samples.to_vec()))
            }
            State::PossibleSilence => {
                self.accumulated.extend_from_slice(samples);

                if is_speech {
                    debug!("VAD: PossibleSilence -> Speech (resumed)");
                    self.transition(State::Speech, now);
                    Some(VadEvent::SpeechChunk(samples.to_vec()))
                } else if elapsed.as_millis() >= self.silence_duration_ms as u128 {
                    let audio = std::mem::take(&mut self.accumulated);
                    self.transition(State::Silence, now);
                    self.pre_roll.clear();
                    info!("VAD: Speech ended ({} samples)", audio.len());
                    Some(VadEvent::SpeechEnd(audio))
                } else {
                    None
                }
            }
        }
    }

    fn energy(samples: &[i16]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_squares: f64 = samples
            .iter()
            .map(|&s| {
                let f = s as f64 / 32768.0;
                f * f
            })
            .sum();
        (sum_squares / samples.len() as f64).sqrt() as f32
    }

    /// Append a chunk to the pre-roll and drop chunks older than `pre_roll_ms`.
    fn push_pre_roll(&mut self, now: Instant, chunk: Vec<i16>) {
        self.pre_roll.push_back((now, chunk));
        let horizon = Duration::from_millis(self.pre_roll_ms);
        while self
            .pre_roll
            .front()
            .is_some_and(|(t, _)| now.duration_since(*t) > horizon)
        {
            self.pre_roll.pop_front();
        }
    }

    fn transition(&mut self, new_state: State, timestamp: Instant) {
        self.state = new_state;
        self.state_start = timestamp;
    }
}

impl Default for VoiceActivityDetector {
    fn default() -> Self {
        Self::new()
    }
}
