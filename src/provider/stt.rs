use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::audio::AudioChunk;

/// A transcription result from an STT provider.
#[derive(Debug, Clone)]
pub struct Transcription {
    /// The transcribed text.
    pub text: String,
    /// Whether this is a final or interim result.
    pub is_final: bool,
}

/// Speech-to-text provider: streams audio in, produces text transcriptions.
#[async_trait]
pub trait SttProvider: Send + Sync {
    /// Transcribe a complete audio buffer (non-streaming).
    async fn transcribe(&self, audio: AudioChunk) -> Result<Transcription>;

    /// Start a streaming transcription session.
    ///
    /// Returns a sender for audio chunks and a receiver for transcription results.
    /// Send audio chunks as they arrive; receive transcriptions (interim and final).
    /// Drop the sender to end the session.
    async fn stream(&self) -> Result<(mpsc::Sender<AudioChunk>, mpsc::Receiver<Transcription>)>;
}
