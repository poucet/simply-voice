use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::audio::AudioChunk;

/// Text-to-speech provider: takes text, produces audio.
#[async_trait]
pub trait TtsProvider: Send + Sync {
    /// Synthesize a complete text string into audio (non-streaming).
    /// `voice` is a provider-specific voice ID. Empty string uses the provider default.
    async fn synthesize(&self, text: &str, voice: &str) -> Result<AudioChunk>;

    /// Start a streaming synthesis session.
    ///
    /// Returns a sender for text fragments and a receiver for audio chunks.
    /// Send text as it arrives (e.g., token-by-token from an LLM); receive audio chunks.
    /// Drop the sender to signal end of input.
    async fn stream(&self) -> Result<(mpsc::Sender<String>, mpsc::Receiver<AudioChunk>)>;

    /// List available voices for this provider.
    async fn voices(&self) -> Result<Vec<Voice>>;
}

/// A voice available from a TTS provider.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Voice {
    /// Provider-specific voice ID.
    pub id: String,
    /// Human-readable name.
    pub name: String,
}
