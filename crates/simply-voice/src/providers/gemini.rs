//! Gemini voice provider — STT + TTS via `generateContent`.
//!
//! - **STT**: send the audio buffer as `inline_data` (audio/wav) to a
//!   multimodal Gemini model (default `gemini-2.5-flash`) with a terse
//!   "transcribe this" prompt.
//! - **TTS**: call a TTS-capable Gemini model (default
//!   `gemini-2.5-flash-preview-tts`) with `responseModalities: ["AUDIO"]`
//!   and a prebuilt voice.
//!
//! Replaces the old `GeminiRealtimeProvider` (Multimodal Live API over
//! WebSocket) — the new shape fits the `SttProvider` / `TtsProvider`
//! traits so it slots into the same STT/TTS pipeline as Voxtral and
//! ElevenLabs.

use anyhow::Result;
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine};
use reqwest::Client;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::{debug, error, warn};

use crate::audio::{Audio, AudioChunk};
use crate::provider::{SttProvider, TtsProvider, Transcription, Voice};

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_STT_MODEL: &str = "models/gemini-2.5-flash";
const DEFAULT_TTS_MODEL: &str = "models/gemini-2.5-flash-preview-tts";
const STT_PROMPT: &str = "Transcribe the user's speech verbatim. Return only the transcription — no commentary, no timestamps, no punctuation fixes beyond what was spoken.";

/// Gemini's prebuilt TTS voices. These are documented names the API accepts
/// as `speechConfig.voiceConfig.prebuiltVoiceConfig.voiceName`.
const PREBUILT_VOICES: &[(&str, &str)] = &[
    ("Zephyr", "Zephyr (bright)"),
    ("Puck", "Puck (upbeat)"),
    ("Charon", "Charon (informative)"),
    ("Kore", "Kore (firm)"),
    ("Fenrir", "Fenrir (excitable)"),
    ("Leda", "Leda (youthful)"),
    ("Orus", "Orus (firm)"),
    ("Aoede", "Aoede (breezy)"),
    ("Callirrhoe", "Callirrhoe (easygoing)"),
    ("Autonoe", "Autonoe (bright)"),
    ("Enceladus", "Enceladus (breathy)"),
    ("Iapetus", "Iapetus (clear)"),
    ("Umbriel", "Umbriel (easygoing)"),
    ("Algieba", "Algieba (smooth)"),
    ("Despina", "Despina (smooth)"),
    ("Erinome", "Erinome (clear)"),
    ("Algenib", "Algenib (gravelly)"),
    ("Rasalgethi", "Rasalgethi (informative)"),
    ("Laomedeia", "Laomedeia (upbeat)"),
    ("Achernar", "Achernar (soft)"),
    ("Alnilam", "Alnilam (firm)"),
    ("Schedar", "Schedar (even)"),
    ("Gacrux", "Gacrux (mature)"),
    ("Pulcherrima", "Pulcherrima (forward)"),
    ("Achird", "Achird (friendly)"),
    ("Zubenelgenubi", "Zubenelgenubi (casual)"),
    ("Vindemiatrix", "Vindemiatrix (gentle)"),
    ("Sadachbia", "Sadachbia (lively)"),
    ("Sadaltager", "Sadaltager (knowledgeable)"),
    ("Sulafat", "Sulafat (warm)"),
];

pub struct GeminiProvider {
    client: Client,
    api_key: String,
    base_url: String,
    stt_model: String,
    tts_model: String,
}

impl GeminiProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
            stt_model: DEFAULT_STT_MODEL.to_string(),
            tts_model: DEFAULT_TTS_MODEL.to_string(),
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

    fn generate_url(&self, model: &str) -> String {
        format!("{}/{model}:generateContent?key={}", self.base_url.trim_end_matches('/'), self.api_key)
    }

    /// Build a WAV file from 16kHz mono PCM16 LE bytes so Gemini accepts it
    /// via `inline_data` with `audio/wav`.
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
        wav.extend_from_slice(&1u16.to_le_bytes());
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
impl SttProvider for GeminiProvider {
    async fn transcribe(&self, audio: AudioChunk) -> Result<Transcription> {
        let wav = Self::pcm16_to_wav(&audio.data, crate::audio::SAMPLE_RATE);
        let b64 = STANDARD.encode(&wav);

        let body = json!({
            "contents": [{
                "parts": [
                    { "inline_data": { "mime_type": "audio/wav", "data": b64 } },
                    { "text": STT_PROMPT },
                ]
            }]
        });

        debug!(audio_bytes = audio.data.len(), model = %self.stt_model, "gemini STT: transcribing");

        let resp = self.client
            .post(self.generate_url(&self.stt_model))
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        let json: Value = resp.json().await?;

        if !status.is_success() {
            let err = json["error"]["message"].as_str().unwrap_or("unknown error");
            anyhow::bail!("Gemini STT error ({status}): {err}");
        }

        let text = extract_first_text(&json).unwrap_or_default();
        Ok(Transcription { text: text.trim().to_string(), is_final: true })
    }

    async fn stream(&self) -> Result<(mpsc::Sender<AudioChunk>, mpsc::Receiver<Transcription>)> {
        // Gemini's text-generation transcription isn't a streaming endpoint —
        // we gather all incoming chunks, then transcribe once at end-of-stream.
        let (audio_tx, mut audio_rx) = mpsc::channel::<AudioChunk>(32);
        let (text_tx, text_rx) = mpsc::channel::<Transcription>(8);

        let client = self.client.clone();
        let url = self.generate_url(&self.stt_model);
        let model = self.stt_model.clone();

        tokio::spawn(async move {
            let mut buffer: Vec<u8> = Vec::new();
            while let Some(chunk) = audio_rx.recv().await {
                buffer.extend_from_slice(&chunk.data);
            }
            if buffer.is_empty() {
                return;
            }

            let wav = Self::pcm16_to_wav(&buffer, crate::audio::SAMPLE_RATE);
            let b64 = STANDARD.encode(&wav);
            let body = json!({
                "contents": [{
                    "parts": [
                        { "inline_data": { "mime_type": "audio/wav", "data": b64 } },
                        { "text": STT_PROMPT },
                    ]
                }]
            });

            match client.post(url).json(&body).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    match resp.json::<Value>().await {
                        Ok(v) if status.is_success() => {
                            if let Some(text) = extract_first_text(&v) {
                                let _ = text_tx.send(Transcription {
                                    text: text.trim().to_string(),
                                    is_final: true,
                                }).await;
                            }
                        }
                        Ok(v) => error!(model = %model, status = %status, error = %v["error"]["message"], "gemini STT error"),
                        Err(e) => error!(error = %e, "gemini STT parse error"),
                    }
                }
                Err(e) => error!(error = %e, "gemini STT request failed"),
            }
        });

        Ok((audio_tx, text_rx))
    }
}

// ---------------------------------------------------------------------------
// TTS
// ---------------------------------------------------------------------------

#[async_trait]
impl TtsProvider for GeminiProvider {
    async fn synthesize(&self, text: &str, voice: &str) -> Result<Audio> {
        let voice_name = if voice.is_empty() {
            PREBUILT_VOICES.first().map(|(id, _)| *id).unwrap_or("Aoede")
        } else {
            voice
        };

        let body = json!({
            "contents": [{ "parts": [{ "text": text }] }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": { "voiceName": voice_name }
                    }
                }
            }
        });

        let resp = self.client
            .post(self.generate_url(&self.tts_model))
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        let json: Value = resp.json().await?;

        if !status.is_success() {
            let err = json["error"]["message"].as_str().unwrap_or("unknown error");
            anyhow::bail!("Gemini TTS error ({status}): {err}");
        }

        let (audio_b64, mime) = extract_first_inline_audio(&json)
            .ok_or_else(|| anyhow::anyhow!("no inline_data audio in Gemini TTS response"))?;

        let raw_bytes = STANDARD.decode(audio_b64.as_bytes())?;
        let sample_rate = parse_sample_rate_from_mime(&mime).unwrap_or(24_000);

        // Gemini TTS returns signed 16-bit LE PCM mono.
        Ok(Audio::from_pcm16(raw_bytes, sample_rate))
    }

    async fn stream(&self) -> Result<(mpsc::Sender<String>, mpsc::Receiver<AudioChunk>)> {
        // No native streaming TTS — accumulate text, synthesize once.
        let (text_tx, mut text_rx) = mpsc::channel::<String>(32);
        let (audio_tx, audio_rx) = mpsc::channel::<AudioChunk>(8);

        let client = self.client.clone();
        let url = self.generate_url(&self.tts_model);
        let default_voice = PREBUILT_VOICES.first().map(|(id, _)| id.to_string()).unwrap_or_else(|| "Aoede".to_string());

        tokio::spawn(async move {
            let mut full_text = String::new();
            while let Some(fragment) = text_rx.recv().await {
                full_text.push_str(&fragment);
            }
            if full_text.trim().is_empty() {
                return;
            }

            let body = json!({
                "contents": [{ "parts": [{ "text": full_text }] }],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": { "prebuiltVoiceConfig": { "voiceName": default_voice } }
                    }
                }
            });

            match client.post(url).json(&body).send().await {
                Ok(resp) => {
                    match resp.json::<Value>().await {
                        Ok(v) => {
                            if let Some((b64, _mime)) = extract_first_inline_audio(&v) {
                                if let Ok(raw) = STANDARD.decode(b64.as_bytes()) {
                                    let _ = audio_tx.send(AudioChunk::new(raw)).await;
                                }
                            } else {
                                warn!(body = %v, "gemini TTS: no audio in response");
                            }
                        }
                        Err(e) => error!(error = %e, "gemini TTS parse error"),
                    }
                }
                Err(e) => error!(error = %e, "gemini TTS request failed"),
            }
        });

        Ok((text_tx, audio_rx))
    }

    async fn voices(&self) -> Result<Vec<Voice>> {
        Ok(PREBUILT_VOICES.iter()
            .map(|(id, name)| Voice { id: id.to_string(), name: name.to_string() })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Response-parsing helpers
// ---------------------------------------------------------------------------

/// Extract the first `text` part from the first candidate's content.
fn extract_first_text(body: &Value) -> Option<String> {
    body.get("candidates")?
        .as_array()?
        .iter()
        .filter_map(|c| c.get("content")?.get("parts")?.as_array())
        .flatten()
        .find_map(|p| p.get("text").and_then(|t| t.as_str()).map(|s| s.to_string()))
}

/// Extract the first `inline_data` audio part from the first candidate's
/// content. Returns (base64_data, mime_type).
fn extract_first_inline_audio(body: &Value) -> Option<(String, String)> {
    let parts = body.get("candidates")?
        .as_array()?
        .iter()
        .filter_map(|c| c.get("content")?.get("parts")?.as_array())
        .flatten();
    for p in parts {
        // Gemini may serialize the field as inline_data OR inlineData.
        let inline = p.get("inline_data").or_else(|| p.get("inlineData"))?;
        let data = inline.get("data")?.as_str()?.to_string();
        let mime = inline.get("mime_type").or_else(|| inline.get("mimeType"))
            .and_then(|m| m.as_str())
            .unwrap_or("audio/pcm")
            .to_string();
        return Some((data, mime));
    }
    None
}

/// Parse a sample rate hint from a MIME type like `audio/pcm;rate=24000`.
fn parse_sample_rate_from_mime(mime: &str) -> Option<u32> {
    mime.split(';')
        .find_map(|kv| kv.trim().strip_prefix("rate="))
        .and_then(|v| v.parse().ok())
}
