use serde::{Deserialize, Serialize};

/// Voice session events emitted to the client.
///
/// These drive UI state — the client needs to know whether the pipeline
/// is idle, listening, transcribing, or buffering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoiceEvent {
    /// VAD detected speech — user is speaking.
    Listening,
    /// Speech ended, audio is being transcribed (pipeline mode only).
    Transcribing,
    /// User's speech was transcribed into text.
    UserTranscript(String),
    /// Model's text response (pipeline mode) or transcript of model speech (realtime mode).
    ModelTranscript(String),
    /// Audio output from the model (realtime mode or TTS).
    Audio(crate::AudioChunk),
    /// The model finished its response turn.
    TurnEnd,
    /// An error occurred in the pipeline.
    Error(String),
}

/// Voice session state, trackable by the client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceState {
    /// Waiting for speech.
    Idle,
    /// VAD detected speech, user is talking.
    Listening,
    /// Processing audio (STT or realtime).
    Processing,
    /// Model is speaking (TTS playing or realtime audio out).
    Speaking,
    /// Buffering results while the downstream consumer is busy.
    Buffering,
}
