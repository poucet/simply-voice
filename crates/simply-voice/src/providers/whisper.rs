//! Whisper STT provider — local speech-to-text via whisper.cpp.

use anyhow::Result;
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Once;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::audio::AudioChunk;
use crate::provider::{SttProvider, Transcription};

static INIT_LOGGING: Once = Once::new();

pub struct WhisperProvider {
    model_path: PathBuf,
}

impl WhisperProvider {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        INIT_LOGGING.call_once(|| {
            whisper_rs::install_logging_hooks();
        });

        let model_path = model_path.as_ref().to_path_buf();

        // Validate the model loads
        let params = whisper_rs::WhisperContextParameters::default();
        let _ = whisper_rs::WhisperContext::new_with_params(
            model_path.to_str().unwrap(),
            params,
        )?;

        Ok(Self { model_path })
    }

    fn create_context(&self) -> Result<whisper_rs::WhisperContext> {
        let params = whisper_rs::WhisperContextParameters::default();
        Ok(whisper_rs::WhisperContext::new_with_params(
            self.model_path.to_str().unwrap(),
            params,
        )?)
    }

    /// Convert PCM16 LE bytes to f32 samples for whisper.
    fn pcm16_to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32 / 32768.0
            })
            .collect()
    }

    fn transcribe_samples(context: &whisper_rs::WhisperContext, samples: &[f32]) -> Result<String> {
        let mut state = context.create_state()?;
        let mut params =
            whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });

        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);

        state.full(params, samples)?;

        let num_segments = state.full_n_segments();
        let mut result = String::new();

        for i in 0..num_segments {
            if let Some(segment) = state.get_segment(i) {
                result.push_str(&segment.to_string());
                result.push(' ');
            }
        }

        Ok(result.trim().to_string())
    }
}

#[async_trait]
impl SttProvider for WhisperProvider {
    async fn transcribe(&self, audio: AudioChunk) -> Result<Transcription> {
        let samples = Self::pcm16_to_f32(&audio.data);
        let context = self.create_context()?;

        let text = tokio::task::spawn_blocking(move || {
            Self::transcribe_samples(&context, &samples)
        })
        .await??;

        Ok(Transcription {
            text,
            is_final: true,
        })
    }

    async fn stream(&self) -> Result<(mpsc::Sender<AudioChunk>, mpsc::Receiver<Transcription>)> {
        let (audio_tx, mut audio_rx) = mpsc::channel::<AudioChunk>(32);
        let (text_tx, text_rx) = mpsc::channel::<Transcription>(32);

        let context = self.create_context()?;

        tokio::task::spawn_blocking(move || {
            // Accumulate audio until the sender is dropped, then transcribe.
            // Whisper works best on complete utterances, not tiny chunks.
            let mut all_samples: Vec<f32> = Vec::new();

            while let Some(chunk) = audio_rx.blocking_recv() {
                let samples = Self::pcm16_to_f32(&chunk.data);
                all_samples.extend_from_slice(&samples);
            }

            if all_samples.is_empty() {
                return;
            }

            match Self::transcribe_samples(&context, &all_samples) {
                Ok(text) if !text.is_empty() => {
                    let _ = text_tx.blocking_send(Transcription {
                        text,
                        is_final: true,
                    });
                }
                Ok(_) => debug!("Whisper: empty transcription"),
                Err(e) => error!("Whisper transcription failed: {e}"),
            }
        });

        Ok((audio_tx, text_rx))
    }
}
