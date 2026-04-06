use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::audio::AudioChunk;

/// Event emitted by a realtime voice session.
#[derive(Debug, Clone)]
pub enum RealtimeEvent {
    /// Audio output from the model.
    Audio(AudioChunk),
    /// Text transcript of what the model said.
    ModelTranscript(String),
    /// Text transcript of what the user said (if supported).
    UserTranscript(String),
    /// The model has finished its current response turn.
    TurnEnd,
}

/// Input to a realtime voice session.
#[derive(Debug, Clone)]
pub enum RealtimeInput {
    /// Audio from the user's microphone.
    Audio(AudioChunk),
    /// Text instruction injected into the conversation (e.g., system prompt update).
    Text(String),
    /// Signal an interruption — cancel the model's current response.
    Interrupt,
}

/// Realtime voice provider: bidirectional audio with a built-in LLM.
///
/// Unlike STT + TTS composed with your own LLM, a realtime provider handles
/// the full loop — audio in, reasoning, audio out — as a single streaming session.
/// Examples: Gemini Realtime API.
#[async_trait]
pub trait RealtimeProvider: Send + Sync {
    /// Start a realtime voice session.
    ///
    /// Returns a sender for inputs (audio, text, interrupts) and a receiver for events.
    /// The session stays open until the sender is dropped.
    async fn connect(
        &self,
        config: RealtimeConfig,
    ) -> Result<(mpsc::Sender<RealtimeInput>, mpsc::Receiver<RealtimeEvent>)>;
}

/// Configuration for a realtime voice session.
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// System prompt / instructions for the model.
    pub system_prompt: Option<String>,
    /// Voice to use for audio output.
    pub voice: Option<String>,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            system_prompt: None,
            voice: None,
        }
    }
}
