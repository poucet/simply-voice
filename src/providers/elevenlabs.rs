//! ElevenLabs TTS provider.
//!
//! TTS only — ElevenLabs doesn't offer STT.
//! API: POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}
//! Returns raw audio bytes (PCM, MP3, etc.)

use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::audio::{Audio, AudioChunk};
use crate::provider::{TtsProvider, Voice};

const BASE_URL: &str = "https://api.elevenlabs.io";

pub struct ElevenLabsProvider {
    client: Client,
    api_key: String,
    model_id: String,
}

impl ElevenLabsProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model_id: "eleven_multilingual_v2".to_string(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model_id = model.into();
        self
    }
}

#[async_trait]
impl TtsProvider for ElevenLabsProvider {
    async fn synthesize(&self, text: &str, voice: &str) -> Result<Audio> {
        let voice_id = if voice.is_empty() {
            // Default ElevenLabs voice (Rachel)
            "21m00Tcm4TlvDq8ikWAM".to_string()
        } else {
            voice.to_string()
        };

        let body = json!({
            "text": text,
            "model_id": self.model_id,
        });

        // Request PCM 24kHz for consistency with other providers
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
            let err_text = resp.text().await.unwrap_or_default();
            anyhow::bail!("ElevenLabs TTS error ({status}): {err_text}");
        }

        // Response is raw PCM16 LE bytes at 24kHz
        let bytes = resp.bytes().await?;
        Ok(Audio::from_pcm16(bytes.to_vec(), 24_000))
    }

    async fn stream(&self) -> Result<(mpsc::Sender<String>, mpsc::Receiver<AudioChunk>)> {
        // ElevenLabs has streaming but we don't implement it yet
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
