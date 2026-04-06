//! Voxtral provider — STT via realtime transcription WebSocket,
//! TTS via speech generation REST API.
//!
//! Works with both Mistral's hosted API and local vLLM serving.
//! Set `base_url` to point at your local vLLM instance.

use anyhow::Result;
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine};
use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, warn};

use crate::audio::AudioChunk;
use crate::provider::{SttProvider, TtsProvider, Transcription, Voice};

const DEFAULT_BASE_URL: &str = "https://api.mistral.ai";
const DEFAULT_STT_MODEL: &str = "voxtral-mini-transcribe-realtime-2602";
const DEFAULT_TTS_MODEL: &str = "voxtral-mini-tts-2603";

pub struct VoxtralProvider {
    client: Client,
    api_key: String,
    base_url: String,
    stt_model: String,
    tts_model: String,
}

impl VoxtralProvider {
    /// Connect to Mistral's hosted API.
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
            stt_model: DEFAULT_STT_MODEL.to_string(),
            tts_model: DEFAULT_TTS_MODEL.to_string(),
        }
    }

    /// Connect to a local vLLM instance (or any compatible server).
    pub fn local(base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: String::new(),
            base_url: base_url.into(),
            stt_model: "mistralai/Voxtral-Mini-4B-Realtime-2602".to_string(),
            tts_model: "mistralai/Voxtral-4B-TTS-2603".to_string(),
        }
    }

    pub fn with_stt_model(mut self, model: impl Into<String>) -> Self {
        self.stt_model = model.into();
        self
    }

    pub fn with_tts_model(mut self, model: impl Into<String>) -> Self {
        self.tts_model = model.into();
        self
    }

    /// HTTP base URL (e.g. "https://api.mistral.ai" or "http://localhost:8000").
    fn http_url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url.trim_end_matches('/'))
    }

    /// WebSocket base URL — derive from http base URL.
    fn ws_url(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        let ws_base = if base.starts_with("https://") {
            base.replacen("https://", "wss://", 1)
        } else {
            base.replacen("http://", "ws://", 1)
        };
        if self.api_key.is_empty() {
            format!("{ws_base}{path}")
        } else {
            format!("{ws_base}{path}?api_key={}", self.api_key)
        }
    }
}

// ---------------------------------------------------------------------------
// STT — Realtime transcription via WebSocket
// ---------------------------------------------------------------------------

#[async_trait]
impl SttProvider for VoxtralProvider {
    async fn transcribe(&self, audio: AudioChunk) -> Result<Transcription> {
        let (tx, mut rx) = SttProvider::stream(self).await?;
        tx.send(audio).await.map_err(|_| anyhow::anyhow!("send failed"))?;
        drop(tx);

        let mut full_text = String::new();
        while let Some(t) = rx.recv().await {
            full_text.push_str(&t.text);
        }

        Ok(Transcription {
            text: full_text.trim().to_string(),
            is_final: true,
        })
    }

    async fn stream(&self) -> Result<(mpsc::Sender<AudioChunk>, mpsc::Receiver<Transcription>)> {
        let url = self.ws_url("/v1/audio/transcriptions/stream");

        let mut request = tokio_tungstenite::tungstenite::http::Request::builder()
            .uri(&url)
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header("Sec-WebSocket-Version", "13")
            .header("Sec-WebSocket-Key", tokio_tungstenite::tungstenite::handshake::client::generate_key());

        if !self.api_key.is_empty() {
            request = request.header("Authorization", format!("Bearer {}", self.api_key));
        }

        let request = request.body(())?;
        let (ws_stream, _) = connect_async(request).await?;
        let (mut ws_tx, mut ws_rx) = ws_stream.split();

        // Send setup/config message
        let setup = json!({
            "model": self.stt_model,
            "encoding": "pcm_s16le",
            "sample_rate": 16000,
            "language": "en",
        });
        ws_tx.send(Message::Text(setup.to_string().into())).await?;

        let (audio_tx, mut audio_rx) = mpsc::channel::<AudioChunk>(64);
        let (text_tx, text_rx) = mpsc::channel::<Transcription>(64);

        // Sender: forward audio chunks as binary frames
        tokio::spawn(async move {
            while let Some(chunk) = audio_rx.recv().await {
                if let Err(e) = ws_tx.send(Message::Binary(chunk.data.to_vec().into())).await {
                    warn!("Voxtral STT send error: {e}");
                    break;
                }
            }
            let _ = ws_tx.close().await;
            debug!("Voxtral STT sender exited");
        });

        // Receiver: parse transcription events
        tokio::spawn(async move {
            while let Some(result) = ws_rx.next().await {
                match result {
                    Ok(Message::Text(text)) => {
                        let msg: Value = match serde_json::from_str(&text) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };

                        if let Some(text) = msg.get("text").and_then(|t| t.as_str()) {
                            if !text.is_empty() {
                                let is_final = msg.get("type")
                                    .and_then(|t| t.as_str())
                                    .map(|t| t == "TranscriptionStreamDone")
                                    .unwrap_or(false);

                                let _ = text_tx.send(Transcription {
                                    text: text.to_string(),
                                    is_final,
                                }).await;
                            }
                        }

                        if msg.get("type").and_then(|t| t.as_str()) == Some("TranscriptionStreamDone") {
                            break;
                        }
                    }
                    Ok(Message::Close(_)) => break,
                    Err(e) => {
                        error!("Voxtral STT ws error: {e}");
                        break;
                    }
                    _ => {}
                }
            }
            debug!("Voxtral STT receiver exited");
        });

        Ok((audio_tx, text_rx))
    }
}

// ---------------------------------------------------------------------------
// TTS — Speech generation via REST
// ---------------------------------------------------------------------------

#[async_trait]
impl TtsProvider for VoxtralProvider {
    async fn synthesize(&self, text: &str) -> Result<AudioChunk> {
        let body = json!({
            "model": self.tts_model,
            "input": text,
            "response_format": "pcm",
        });

        let mut req = self.client
            .post(self.http_url("/v1/audio/speech"))
            .json(&body);

        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }

        let resp = req.send().await?;
        let status = resp.status();
        let json: Value = resp.json().await?;

        if !status.is_success() {
            let err = json["message"].as_str()
                .or(json["error"]["message"].as_str())
                .unwrap_or("unknown error");
            anyhow::bail!("Voxtral TTS error ({status}): {err}");
        }

        let audio_b64 = json["audio_data"].as_str()
            .ok_or_else(|| anyhow::anyhow!("no audio_data in response"))?;

        // PCM format returns raw float32 LE samples — convert to PCM16 LE
        let raw_bytes = STANDARD.decode(audio_b64)?;
        let pcm16: Vec<u8> = raw_bytes
            .chunks_exact(4)
            .flat_map(|chunk| {
                let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let i = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                i.to_le_bytes()
            })
            .collect();

        Ok(AudioChunk::with_sample_rate(pcm16, 24_000))
    }

    async fn stream(&self) -> Result<(mpsc::Sender<String>, mpsc::Receiver<AudioChunk>)> {
        // Voxtral TTS is not streaming — accumulate text, synthesize once.
        let (text_tx, mut text_rx) = mpsc::channel::<String>(32);
        let (audio_tx, audio_rx) = mpsc::channel::<AudioChunk>(32);

        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let url = self.http_url("/v1/audio/speech");
        let model = self.tts_model.clone();

        tokio::spawn(async move {
            let mut full_text = String::new();
            while let Some(fragment) = text_rx.recv().await {
                full_text.push_str(&fragment);
            }

            if full_text.trim().is_empty() {
                return;
            }

            let body = json!({
                "model": model,
                "input": full_text,
                "response_format": "pcm",
            });

            let mut req = client.post(&url).json(&body);
            if !api_key.is_empty() {
                req = req.bearer_auth(&api_key);
            }

            match req.send().await {
                Ok(resp) => {
                    if let Ok(json) = resp.json::<Value>().await {
                        if let Some(b64) = json["audio_data"].as_str() {
                            if let Ok(raw_bytes) = STANDARD.decode(b64) {
                                let pcm16: Vec<u8> = raw_bytes
                                    .chunks_exact(4)
                                    .flat_map(|chunk| {
                                        let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                                        let i = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                                        i.to_le_bytes()
                                    })
                                    .collect();
                                let _ = audio_tx.send(AudioChunk::with_sample_rate(pcm16, 24_000)).await;
                            }
                        }
                    }
                }
                Err(e) => error!("Voxtral TTS failed: {e}"),
            }
        });

        Ok((text_tx, audio_rx))
    }

    async fn voices(&self) -> Result<Vec<Voice>> {
        let mut req = self.client.get(self.http_url("/v1/audio/voices"));
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }

        let resp = req.send().await?;
        let json: Value = resp.json().await?;
        let voices = json["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| {
                        Some(Voice {
                            id: v["id"].as_str()?.to_string(),
                            name: v["name"].as_str()?.to_string(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(voices)
    }
}
