//! Gemini Realtime provider — bidirectional audio via WebSocket.
//!
//! Uses the Gemini Multimodal Live API for real-time voice conversation.
//! Audio in: 16kHz 16-bit PCM mono (base64 in JSON).
//! Audio out: 24kHz 16-bit PCM mono (base64 in JSON).
//! The model has built-in VAD and handles interruptions automatically.

use anyhow::Result;
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

use crate::audio::AudioChunk;
use crate::provider::{RealtimeConfig, RealtimeEvent, RealtimeInput, RealtimeProvider};

const WS_ENDPOINT: &str = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";
const OUTPUT_SAMPLE_RATE: u32 = 24_000;

pub struct GeminiRealtimeProvider {
    api_key: String,
    model: String,
}

impl GeminiRealtimeProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: "models/gemini-2.0-flash-live-001".to_string(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    fn build_setup_message(&self, config: &RealtimeConfig) -> Value {
        let mut generation_config = json!({
            "responseModalities": ["AUDIO", "TEXT"],
        });

        if let Some(ref voice) = config.voice {
            generation_config["speechConfig"] = json!({
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice
                    }
                }
            });
        }

        let mut setup = json!({
            "model": self.model,
            "generationConfig": generation_config,
        });

        if let Some(ref prompt) = config.system_prompt {
            setup["systemInstruction"] = json!({
                "parts": [{ "text": prompt }]
            });
        }

        json!({ "setup": setup })
    }
}

#[async_trait]
impl RealtimeProvider for GeminiRealtimeProvider {
    async fn connect(
        &self,
        config: RealtimeConfig,
    ) -> Result<(mpsc::Sender<RealtimeInput>, mpsc::Receiver<RealtimeEvent>)> {
        let url = format!("{WS_ENDPOINT}?key={}", self.api_key);
        let (ws_stream, _) = connect_async(&url).await?;
        let (mut ws_tx, mut ws_rx) = ws_stream.split();

        // Send setup message
        let setup = self.build_setup_message(&config);
        ws_tx.send(Message::Text(setup.to_string().into())).await?;

        // Wait for setupComplete
        loop {
            match ws_rx.next().await {
                Some(Ok(Message::Text(text))) => {
                    let msg: Value = serde_json::from_str(&text)?;
                    if msg.get("setupComplete").is_some() {
                        info!("Gemini realtime session established");
                        break;
                    }
                }
                Some(Err(e)) => anyhow::bail!("WebSocket error during setup: {e}"),
                None => anyhow::bail!("WebSocket closed during setup"),
                _ => continue,
            }
        }

        let (input_tx, mut input_rx) = mpsc::channel::<RealtimeInput>(64);
        let (event_tx, event_rx) = mpsc::channel::<RealtimeEvent>(64);

        // Spawn sender task: reads from input_rx, writes to WebSocket
        let event_tx_clone = event_tx.clone();
        tokio::spawn(async move {
            while let Some(input) = input_rx.recv().await {
                let msg = match input {
                    RealtimeInput::Audio(chunk) => {
                        let b64 = STANDARD.encode(&chunk.data);
                        json!({
                            "realtimeInput": {
                                "mediaChunks": [{
                                    "mimeType": format!("audio/pcm;rate={}", crate::audio::SAMPLE_RATE),
                                    "data": b64
                                }]
                            }
                        })
                    }
                    RealtimeInput::Text(text) => {
                        json!({
                            "clientContent": {
                                "turns": [{
                                    "role": "user",
                                    "parts": [{ "text": text }]
                                }],
                                "turnComplete": true
                            }
                        })
                    }
                    RealtimeInput::Interrupt => {
                        // Gemini handles interruptions via VAD automatically.
                        // Client can't explicitly interrupt, so we skip.
                        continue;
                    }
                };

                if let Err(e) = ws_tx.send(Message::Text(msg.to_string().into())).await {
                    warn!("Failed to send to Gemini WS: {e}");
                    break;
                }
            }

            // Input channel closed — close WebSocket
            let _ = ws_tx.close().await;
            debug!("Gemini sender task exited");
        });

        // Spawn receiver task: reads from WebSocket, writes to event_tx
        tokio::spawn(async move {
            while let Some(result) = ws_rx.next().await {
                match result {
                    Ok(Message::Text(text)) => {
                        let msg: Value = match serde_json::from_str(&text) {
                            Ok(v) => v,
                            Err(e) => {
                                warn!("Failed to parse Gemini message: {e}");
                                continue;
                            }
                        };

                        if let Some(server_content) = msg.get("serverContent") {
                            // Check for interruption
                            if server_content.get("interrupted") == Some(&json!(true)) {
                                debug!("Gemini: model interrupted");
                                // Client should flush playback buffer
                                continue;
                            }

                            // Check for turn complete
                            if server_content.get("turnComplete") == Some(&json!(true)) {
                                let _ = event_tx.send(RealtimeEvent::TurnEnd).await;
                                continue;
                            }

                            // Process model turn parts
                            if let Some(parts) = server_content
                                .get("modelTurn")
                                .and_then(|t| t.get("parts"))
                                .and_then(|p| p.as_array())
                            {
                                for part in parts {
                                    // Audio output
                                    if let Some(inline) = part.get("inlineData") {
                                        if let Some(b64) = inline.get("data").and_then(|d| d.as_str()) {
                                            match STANDARD.decode(b64) {
                                                Ok(bytes) => {
                                                    let chunk = AudioChunk::new(
                                                        bytes
                                                    );
                                                    let _ = event_tx
                                                        .send(RealtimeEvent::Audio(chunk))
                                                        .await;
                                                }
                                                Err(e) => warn!("Failed to decode audio: {e}"),
                                            }
                                        }
                                    }

                                    // Text output
                                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                        let _ = event_tx
                                            .send(RealtimeEvent::ModelTranscript(
                                                text.to_string(),
                                            ))
                                            .await;
                                    }
                                }
                            }
                        }

                        // Tool calls (future use)
                        if msg.get("toolCall").is_some() {
                            debug!("Gemini: tool call received (not yet handled)");
                        }
                    }
                    Ok(Message::Close(_)) => {
                        info!("Gemini WebSocket closed by server");
                        break;
                    }
                    Err(e) => {
                        error!("Gemini WebSocket error: {e}");
                        break;
                    }
                    _ => {}
                }
            }
            debug!("Gemini receiver task exited");
        });

        Ok((input_tx, event_rx))
    }
}
