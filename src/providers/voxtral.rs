//! Voxtral STT provider — speech-to-text via Mistral's chat completions API.
//!
//! Voxtral is an audio-understanding model. Audio is sent as base64-encoded
//! data in the `audio_url` content block of a chat completion request.
//! The model returns transcribed text.

use anyhow::Result;
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine};
use reqwest::Client;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::{debug, error};

use crate::audio::AudioChunk;
use crate::provider::{SttProvider, Transcription};

pub struct VoxtralProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl VoxtralProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model: "mistral-small-latest".to_string(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Encode PCM16 LE audio as a base64 WAV for the API.
    fn pcm16_to_wav_base64(audio: &AudioChunk) -> String {
        let data = &audio.data;
        let sample_rate = audio.sample_rate;
        let num_channels: u16 = 1;
        let bits_per_sample: u16 = 16;
        let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
        let block_align = num_channels * bits_per_sample / 8;
        let data_len = data.len() as u32;

        let mut wav = Vec::with_capacity(44 + data.len());
        // RIFF header
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        // fmt chunk
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&num_channels.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&block_align.to_le_bytes());
        wav.extend_from_slice(&bits_per_sample.to_le_bytes());
        // data chunk
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.extend_from_slice(data);

        STANDARD.encode(&wav)
    }
}

#[async_trait]
impl SttProvider for VoxtralProvider {
    async fn transcribe(&self, audio: AudioChunk) -> Result<Transcription> {
        let wav_b64 = Self::pcm16_to_wav_base64(&audio);
        let data_url = format!("data:audio/wav;base64,{wav_b64}");

        let body = json!({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": { "url": data_url }
                    },
                    {
                        "type": "text",
                        "text": "Transcribe this audio exactly. Return only the transcribed text, nothing else."
                    }
                ]
            }]
        });

        let resp = self.client
            .post("https://api.mistral.ai/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        let json: Value = resp.json().await?;

        if !status.is_success() {
            let err = json["message"].as_str()
                .or(json["error"]["message"].as_str())
                .unwrap_or("unknown error");
            anyhow::bail!("Mistral API error ({status}): {err}");
        }

        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string();

        debug!("Voxtral transcription: {text:?}");

        Ok(Transcription {
            text,
            is_final: true,
        })
    }

    async fn stream(&self) -> Result<(mpsc::Sender<AudioChunk>, mpsc::Receiver<Transcription>)> {
        // Voxtral doesn't support streaming STT — accumulate then transcribe.
        let (audio_tx, mut audio_rx) = mpsc::channel::<AudioChunk>(32);
        let (text_tx, text_rx) = mpsc::channel::<Transcription>(32);

        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let model = self.model.clone();

        tokio::spawn(async move {
            let mut all_data: Vec<u8> = Vec::new();
            let mut sample_rate = crate::audio::SAMPLE_RATE;

            while let Some(chunk) = audio_rx.recv().await {
                sample_rate = chunk.sample_rate;
                all_data.extend_from_slice(&chunk.data);
            }

            if all_data.is_empty() {
                return;
            }

            let audio = AudioChunk::with_sample_rate(all_data, sample_rate);
            let wav_b64 = Self::pcm16_to_wav_base64(&audio);
            let data_url = format!("data:audio/wav;base64,{wav_b64}");

            let body = json!({
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        { "type": "audio_url", "audio_url": { "url": data_url } },
                        { "type": "text", "text": "Transcribe this audio exactly. Return only the transcribed text, nothing else." }
                    ]
                }]
            });

            match client
                .post("https://api.mistral.ai/v1/chat/completions")
                .bearer_auth(&api_key)
                .json(&body)
                .send()
                .await
            {
                Ok(resp) => {
                    if let Ok(json) = resp.json::<Value>().await {
                        let text = json["choices"][0]["message"]["content"]
                            .as_str()
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        if !text.is_empty() {
                            let _ = text_tx.send(Transcription { text, is_final: true }).await;
                        }
                    }
                }
                Err(e) => error!("Voxtral streaming transcription failed: {e}"),
            }
        });

        Ok((audio_tx, text_rx))
    }
}
