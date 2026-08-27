//! Pocket TTS provider — fully local text-to-speech via Kyutai's pocket-tts
//! (candle port). Runs on CPU, ~2x faster with the `metal` feature on Apple
//! Silicon. Model weights auto-download from Hugging Face on first load.
//
// API assumptions compile-verified against pocket-tts 0.6.2: TTSModel::load,
// model.sample_rate, get_voice_state, generate/generate_stream returning
// candle Tensors (flattened to f32 via flatten_all/to_vec1), and the Send
// bounds required by spawn_blocking. Runtime behavior still unverified:
// HF auto-download on first load, and get_voice_state accepting a bundled
// voice name ("alba"), a .wav path, or a .safetensors embedding path.

use anyhow::{Context, Result};
use bytes::Bytes;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tracing::{debug, error, info};

use crate::audio::{Audio, AudioChunk};
use crate::provider::{TtsProvider, Voice};

/// Default model variant published by kyutai (see pocket-tts README).
const DEFAULT_VARIANT: &str = "b6369a24";

/// Voices bundled with the released weights. `--voice` also accepts a path to
/// a .wav (cloned on the fly, slow) or a precomputed .safetensors embedding
/// (fast) — pocket-tts resolves all three forms.
const BUNDLED_VOICES: &[&str] = &[
    "alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma",
];

struct Inner {
    model: pocket_tts::TTSModel,
    /// Voice states are expensive to compute from .wav; cache per voice key.
    voice_states: HashMap<String, pocket_tts::ModelState>,
}

impl Inner {
    /// Ensure the voice state for `voice` is cached. Callers then split-borrow
    /// `model` and `voice_states` so generation can take the model mutably.
    fn ensure_voice_state(&mut self, voice: &str) -> Result<()> {
        if !self.voice_states.contains_key(voice) {
            info!("pocket-tts: preparing voice state for {voice}");
            let state = self
                .model
                .get_voice_state(voice)
                .with_context(|| format!("pocket-tts: failed to load voice '{voice}'"))?;
            self.voice_states.insert(voice.to_string(), state);
        }
        Ok(())
    }
}

/// Fully local TTS on Kyutai's pocket-tts model.
///
/// The model lives behind one mutex: **only one `synthesize()` or `stream()`
/// can run at a time**, and an open stream holds the lock until its text
/// sender is dropped.
pub struct PocketTtsProvider {
    inner: Arc<Mutex<Inner>>,
    default_voice: String,
    sample_rate: u32,
}

impl PocketTtsProvider {
    /// Load the default model variant. `default_voice` is a bundled voice
    /// name, a .wav path, or a .safetensors voice-embedding path; used when
    /// callers pass an empty voice.
    pub fn new(default_voice: impl Into<String>) -> Result<Self> {
        Self::with_variant(DEFAULT_VARIANT, default_voice)
    }

    pub fn with_variant(variant: &str, default_voice: impl Into<String>) -> Result<Self> {
        let default_voice = default_voice.into();
        info!("pocket-tts: loading model variant {variant}");
        let model = pocket_tts::TTSModel::load(variant)
            .with_context(|| format!("pocket-tts: failed to load model variant '{variant}'"))?;
        let sample_rate = model.sample_rate as u32;
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                model,
                voice_states: HashMap::new(),
            })),
            default_voice,
            sample_rate,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn resolve_voice<'a>(&'a self, voice: &'a str) -> &'a str {
        if voice.is_empty() {
            &self.default_voice
        } else {
            voice
        }
    }

    fn f32_to_pcm16_bytes(samples: &[f32]) -> Bytes {
        let mut out = Vec::with_capacity(samples.len() * 2);
        for s in samples {
            let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
            out.extend_from_slice(&v.to_le_bytes());
        }
        Bytes::from(out)
    }

    fn f32_bytes(samples: &[f32]) -> Bytes {
        let mut out = Vec::with_capacity(samples.len() * 4);
        for s in samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        Bytes::from(out)
    }
}

#[async_trait::async_trait]
impl TtsProvider for PocketTtsProvider {
    async fn synthesize(&self, text: &str, voice: &str) -> Result<Audio> {
        let voice = self.resolve_voice(voice).to_string();
        let text = text.to_string();
        let inner = Arc::clone(&self.inner);
        let sample_rate = self.sample_rate;

        let samples: Vec<f32> = tokio::task::spawn_blocking(move || -> Result<Vec<f32>> {
            let mut inner = inner.lock().expect("pocket-tts mutex poisoned");
            inner.ensure_voice_state(&voice)?;
            // Split borrows: state from the cache, model taken separately.
            let Inner { model, voice_states } = &mut *inner;
            let state = &voice_states[&voice];
            // generate() returns a candle tensor of f32 samples.
            let audio = model.generate(&text, state)?;
            Ok(audio.flatten_all()?.to_vec1::<f32>()?)
        })
        .await??;

        Ok(Audio::from_f32_bytes(Self::f32_bytes(&samples), sample_rate))
    }

    async fn stream(&self) -> Result<(mpsc::Sender<String>, mpsc::Receiver<AudioChunk>)> {
        let (text_tx, mut text_rx) = mpsc::channel::<String>(32);
        let (audio_tx, audio_rx) = mpsc::channel::<AudioChunk>(32);

        let inner = Arc::clone(&self.inner);
        let voice = self.default_voice.clone();

        tokio::task::spawn_blocking(move || {
            // The lock is held for the life of the stream — see the type-level
            // doc comment about single-session use.
            let mut inner = inner.lock().expect("pocket-tts mutex poisoned");
            if let Err(e) = inner.ensure_voice_state(&voice) {
                error!("pocket-tts stream: {e:#}");
                return;
            }
            // Split borrows: state from the cache, model taken separately.
            let Inner { model, voice_states } = &mut *inner;
            let state = &voice_states[&voice];

            while let Some(text) = text_rx.blocking_recv() {
                if text.trim().is_empty() {
                    continue;
                }
                let chunks = model.generate_stream(&text, state);
                for chunk in chunks {
                    match chunk {
                        Ok(samples) => {
                            // Stream chunks are candle tensors of f32 samples.
                            let samples =
                                match samples.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                                    Ok(s) => s,
                                    Err(e) => {
                                        error!("pocket-tts: bad audio tensor: {e:#}");
                                        break;
                                    }
                                };
                            let data = Self::f32_to_pcm16_bytes(&samples);
                            if audio_tx.blocking_send(AudioChunk { data }).is_err() {
                                debug!("pocket-tts stream: receiver dropped");
                                return;
                            }
                        }
                        Err(e) => {
                            error!("pocket-tts stream chunk failed: {e:#}");
                            break;
                        }
                    }
                }
            }
        });

        Ok((text_tx, audio_rx))
    }

    async fn voices(&self) -> Result<Vec<Voice>> {
        Ok(BUNDLED_VOICES
            .iter()
            .map(|name| Voice {
                id: (*name).to_string(),
                name: (*name).to_string(),
            })
            .collect())
    }
}
