//! ElevenLabs provider — STT + TTS.
//!
//! TTS: POST /v1/text-to-speech/{voice_id} → raw PCM audio
//! STT: POST /v1/speech-to-text (multipart form, WAV file)

use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde_json::Value;
use tokio::sync::mpsc;
use tracing;

use crate::audio::{Audio, AudioChunk};
use crate::provider::{SttProvider, TtsProvider, Transcription, Voice};

const BASE_URL: &str = "https://api.elevenlabs.io";

pub struct ElevenLabsProvider {
    client: Client,
    api_key: String,
    tts_model: String,
    stt_model: String,
}

impl ElevenLabsProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            tts_model: "eleven_multilingual_v2".to_string(),
            stt_model: "scribe_v2".to_string(),
        }
    }

    pub fn with_tts_model(mut self, model: impl Into<String>) -> Self {
        self.tts_model = model.into();
        self
    }

    /// Build a WAV file from PCM16 LE bytes for upload.
    fn pcm16_to_wav(data: &[u8], sample_rate: u32) -> Vec<u8> {
        let channels: u16 = 1;
        let bits_per_sample: u16 = 16;
        let byte_rate = sample_rate * channels as u32 * bits_per_sample as u32 / 8;
        let block_align = channels * bits_per_sample / 8;
        let data_len = data.len() as u32;

        let mut wav = Vec::with_capacity(44 + data.len());
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&block_align.to_le_bytes());
        wav.extend_from_slice(&bits_per_sample.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.extend_from_slice(data);
        wav
    }
}

// ---------------------------------------------------------------------------
// STT
// ---------------------------------------------------------------------------

#[async_trait]
impl SttProvider for ElevenLabsProvider {
    async fn transcribe(&self, audio: AudioChunk) -> Result<Transcription> {
        let wav = Self::pcm16_to_wav(&audio.data, 16_000);

        let part = reqwest::multipart::Part::bytes(wav)
            .file_name("audio.wav")
            .mime_str("audio/wav")?;

        let form = reqwest::multipart::Form::new()
            .text("model_id", self.stt_model.clone())
            .part("file", part);

        tracing::debug!(audio_bytes = audio.data.len(), "elevenlabs STT: transcribing");

        let resp = self.client
            .post(format!("{BASE_URL}/v1/speech-to-text"))
            .header("xi-api-key", &self.api_key)
            .multipart(form)
            .send()
            .await?;

        let status = resp.status();
        let json: Value = resp.json().await?;

        if !status.is_success() {
            let err = json["detail"].as_str()
                .or(json["message"].as_str())
                .unwrap_or("unknown error");
            anyhow::bail!("ElevenLabs STT error ({status}): {err}");
        }

        let text = json["text"].as_str().unwrap_or("").trim().to_string();
        tracing::debug!(text = %text, "elevenlabs STT: result");

        Ok(Transcription {
            text,
            is_final: true,
        })
    }

    async fn stream(&self) -> Result<(mpsc::Sender<AudioChunk>, mpsc::Receiver<Transcription>)> {
        // Non-streaming: accumulate audio, transcribe on close
        let (audio_tx, mut audio_rx) = mpsc::channel::<AudioChunk>(32);
        let (text_tx, text_rx) = mpsc::channel::<Transcription>(32);

        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let stt_model = self.stt_model.clone();

        tokio::spawn(async move {
            let mut all_data: Vec<u8> = Vec::new();
            while let Some(chunk) = audio_rx.recv().await {
                all_data.extend_from_slice(&chunk.data);
            }

            if all_data.is_empty() {
                return;
            }

            let wav = Self::pcm16_to_wav(&all_data, 16_000);
            let part = match reqwest::multipart::Part::bytes(wav)
                .file_name("audio.wav")
                .mime_str("audio/wav") {
                Ok(p) => p,
                Err(_) => return,
            };

            let form = reqwest::multipart::Form::new()
                .text("model_id", stt_model)
                .part("file", part);

            match client
                .post(format!("{BASE_URL}/v1/speech-to-text"))
                .header("xi-api-key", &api_key)
                .multipart(form)
                .send()
                .await
            {
                Ok(resp) => {
                    if let Ok(json) = resp.json::<Value>().await {
                        let text = json["text"].as_str().unwrap_or("").trim().to_string();
                        if !text.is_empty() {
                            let _ = text_tx.send(Transcription { text, is_final: true }).await;
                        }
                    }
                }
                Err(e) => tracing::error!("elevenlabs STT failed: {e}"),
            }
        });

        Ok((audio_tx, text_rx))
    }
}

// ---------------------------------------------------------------------------
// TTS
// ---------------------------------------------------------------------------

#[async_trait]
impl TtsProvider for ElevenLabsProvider {
    async fn synthesize(&self, text: &str, voice: &str) -> Result<Audio> {
        let voice_id = if voice.is_empty() {
            "21m00Tcm4TlvDq8ikWAM".to_string() // Rachel (default)
        } else {
            voice.to_string()
        };

        let body = serde_json::json!({
            "text": text,
            "model_id": self.tts_model,
        });

        let url = format!(
            "{BASE_URL}/v1/text-to-speech/{voice_id}?output_format=pcm_24000"
        );

        let resp = self.client
            .post(&url)
            .header("xi-api-key", &self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let err = resp.text().await.unwrap_or_default();
            anyhow::bail!("ElevenLabs TTS error ({status}): {err}");
        }

        let bytes = resp.bytes().await?;
        tracing::debug!(bytes = bytes.len(), "elevenlabs TTS: got audio");

        Ok(Audio::from_pcm16(bytes.to_vec(), 24_000))
    }

    async fn stream(&self) -> Result<(mpsc::Sender<String>, mpsc::Receiver<AudioChunk>)> {
        anyhow::bail!("ElevenLabs streaming TTS not implemented")
    }

    async fn voices(&self) -> Result<Vec<Voice>> {
        let resp = self.client
            .get(format!("{BASE_URL}/v1/voices"))
            .header("xi-api-key", &self.api_key)
            .send()
            .await?;

        let json: Value = resp.json().await?;
        let voices = json["voices"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| {
                        Some(Voice {
                            id: v["voice_id"].as_str()?.to_string(),
                            name: v["name"].as_str()?.to_string(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(voices)
    }
}
