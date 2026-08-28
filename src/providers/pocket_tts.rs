//! Pocket TTS provider — fully local text-to-speech via Kyutai's pocket-tts
//! (candle port). Runs on CPU, ~2x faster with the `metal` feature on Apple
//! Silicon. Model weights auto-download from Hugging Face on first load.
//!
//! API verified against the pocket-tts 0.6.2 sources:
//! - `TTSModel::load_with_params_device(variant, ...)` builds the model on a
//!   chosen device (plain `load()` is CPU-only); weights auto-download from
//!   HF (`HF_TOKEN` env respected). `model.sample_rate` is `usize`.
//! - Voice resolution is a CLI-level concern upstream, not a library one:
//!   `get_voice_state(path)` reads a .wav ONLY;
//!   `get_voice_state_from_prompt_file(path)` reads a .safetensors embedding
//!   (single "audio_prompt" tensor); stock voice names map to embeddings in
//!   the kyutai/pocket-tts-without-voice-cloning HF repo. [`resolve_voice_state`]
//!   below mirrors the upstream CLI's resolution.
//! - `generate(&self, text, &ModelState) -> Result<Tensor>` ([C, T] f32);
//!   `generate_stream(&self, ...) -> Box<dyn Iterator<Item = Result<Tensor>>>`
//!   with [B, C, T] chunks.

use anyhow::{Context, Result};
use bytes::Bytes;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tracing::{debug, error, info};

use crate::audio::{Audio, AudioChunk};
use crate::provider::{TtsProvider, Voice};

/// Default model variant published by kyutai.
const DEFAULT_VARIANT: &str = pocket_tts::config::defaults::DEFAULT_VARIANT;

/// Stock voices published alongside the released weights (embeddings in the
/// [`STOCK_VOICE_REPO`] HF repo). A voice spec may also be a path to a .wav
/// (cloned on the fly, slow), a precomputed .safetensors embedding (fast), or
/// an `hf://owner/repo/file` URL — see [`resolve_voice_state`].
pub const BUNDLED_VOICES: &[&str] = &[
    "alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma",
];

/// HF repo holding the stock voice embeddings (same one the upstream CLI uses).
const STOCK_VOICE_REPO: &str = "kyutai/pocket-tts-without-voice-cloning";

/// Weight files `TTSModel::load_with_params_device` resolves for the default
/// variant, mirrored verbatim (repo, filename, pinned revision) from the
/// crate's bundled `config/b6369a24.yaml`: `weights_path` and
/// `flow_lm.lookup_table.tokenizer_path`. Pre-fetching them through a tokened
/// hf-hub API (below) leaves the crate's own `download_if_necessary` a
/// guaranteed cache hit, so its token handling never matters.
const DEFAULT_VARIANT_FILES: &[&str] = &[
    "hf://kyutai/pocket-tts/tts_b6369a24.safetensors@427e3d61b276ed69fdd03de0d185fa8a8d97fc5b",
    "hf://kyutai/pocket-tts-without-voice-cloning/tokenizer.model@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3",
];

/// A Hugging Face hub client that authenticates like the rest of the
/// ecosystem: `ApiBuilder::new()` reads the huggingface-cli token file
/// (`~/.cache/huggingface/token`, via `Cache::default()`) — no env var
/// needed. `$HF_TOKEN_PATH` is honored as an explicit override. The
/// pocket-tts crate itself builds its API with `.with_token(env HF_TOKEN)`,
/// which *clears* the token-file token whenever the env var is unset — hence
/// the pre-fetching in this module. Uses `Cache::default()` (not `from_env`)
/// deliberately: it must be the same cache directory pocket-tts hardcodes.
fn hf_api() -> Result<hf_hub::api::sync::Api> {
    let mut builder = hf_hub::api::sync::ApiBuilder::new();
    if let Ok(token_path) = std::env::var("HF_TOKEN_PATH") {
        if let Ok(contents) = std::fs::read_to_string(&token_path) {
            let token = contents.trim();
            if !token.is_empty() {
                builder = builder.with_token(Some(token.to_string()));
            }
        }
    }
    builder
        .build()
        .context("pocket-tts: failed to build HF hub client")
}

/// Download one `hf://owner/repo/file[@revision]` spec into the shared hf-hub
/// cache, parsing the spec exactly like pocket-tts's
/// `weights::download_if_necessary` so both resolve to the same cached path.
fn fetch_hf_file(spec: &str) -> Result<std::path::PathBuf> {
    use hf_hub::{Repo, RepoType};
    let path = spec.trim_start_matches("hf://");
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() < 3 {
        anyhow::bail!("pocket-tts: invalid hf:// spec '{spec}'");
    }
    let repo_id = format!("{}/{}", parts[0], parts[1]);
    let filename_with_revision = parts[2..].join("/");
    let (filename, revision) = match filename_with_revision.rfind('@') {
        Some(at) => {
            let (f, r) = filename_with_revision.split_at(at);
            (f.to_string(), Some(r[1..].to_string()))
        }
        None => (filename_with_revision, None),
    };
    let repo = match revision {
        Some(rev) => Repo::with_revision(repo_id, RepoType::Model, rev),
        None => Repo::model(repo_id),
    };
    Ok(hf_api()?.repo(repo).get(&filename)?)
}

/// Pre-fetch the default variant's weight files (instant cache hits once
/// downloaded). Failures are logged, not fatal: the crate then retries the
/// download itself, and the load error carries the actionable message.
fn prefetch_default_variant_weights() {
    for spec in DEFAULT_VARIANT_FILES {
        if let Err(e) = fetch_hf_file(spec) {
            tracing::warn!("pocket-tts: pre-fetch of {spec} failed: {e:#}");
        }
    }
}

/// Resolve a voice spec to a `ModelState`, mirroring the upstream pocket-tts
/// CLI: a bundled name downloads its stock embedding from HF; an `hf://` URL
/// downloads then dispatches on extension; otherwise the spec is a local
/// .safetensors embedding or .wav reference clip.
pub fn resolve_voice_state(
    model: &pocket_tts::TTSModel,
    spec: &str,
) -> Result<pocket_tts::ModelState> {
    let spec = spec.trim();
    if BUNDLED_VOICES.contains(&spec) {
        let url = format!("hf://{STOCK_VOICE_REPO}/embeddings/{spec}.safetensors");
        let path = fetch_hf_file(&url)
            .with_context(|| format!("pocket-tts: failed to download stock voice '{spec}'"))?;
        return model
            .get_voice_state_from_prompt_file(&path)
            .with_context(|| format!("pocket-tts: failed to load stock voice '{spec}'"));
    }
    if spec.starts_with("hf://") {
        let path = fetch_hf_file(spec)
            .with_context(|| format!("pocket-tts: failed to download voice '{spec}'"))?;
        return voice_state_from_file(model, &path);
    }
    voice_state_from_file(model, Path::new(spec))
}

/// Load a voice from a local file: .safetensors embedding (fast path) or .wav
/// reference clip (encoded through Mimi, slow).
fn voice_state_from_file(
    model: &pocket_tts::TTSModel,
    path: &Path,
) -> Result<pocket_tts::ModelState> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "safetensors" => model
            .get_voice_state_from_prompt_file(path)
            .with_context(|| format!("pocket-tts: failed to load embedding {path:?}")),
        "wav" | "wave" => model
            .get_voice_state(path)
            .with_context(|| format!("pocket-tts: failed to encode reference clip {path:?}")),
        _ => anyhow::bail!(
            "pocket-tts: voice '{}' is not a bundled name ({}) and not a \
             .wav/.safetensors path",
            path.display(),
            BUNDLED_VOICES.join(", ")
        ),
    }
}

/// Flatten a generated audio tensor ([C, T] or [B, C, T], mono, f32) into raw
/// samples.
fn tensor_to_f32_samples(audio: &candle::Tensor) -> Result<Vec<f32>> {
    Ok(audio.flatten_all()?.to_vec1::<f32>()?)
}

/// Best available device: Metal when the feature is enabled and the device
/// initializes, CPU otherwise.
fn best_device() -> candle::Device {
    #[cfg(feature = "metal")]
    {
        match candle::Device::new_metal(0) {
            Ok(d) => return d,
            Err(e) => tracing::warn!("pocket-tts: Metal unavailable ({e}); falling back to CPU"),
        }
    }
    candle::Device::Cpu
}

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
            let state = resolve_voice_state(&self.model, voice)?;
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
        // Pre-fetch through the token-file-authenticated client; the crate's
        // internal downloads then hit the cache. Other variants have no
        // bundled config upstream, so there is nothing to mirror for them.
        if variant == "b6369a24" {
            prefetch_default_variant_weights();
        }
        use pocket_tts::config::defaults;
        let model = pocket_tts::TTSModel::load_with_params_device(
            variant,
            defaults::TEMPERATURE,
            defaults::LSD_DECODE_STEPS,
            defaults::EOS_THRESHOLD,
            defaults::NOISE_CLAMP,
            &best_device(),
        )
        .with_context(|| {
            format!(
                "pocket-tts: failed to load model variant '{variant}' — if this is a \
                 401, accept the terms at huggingface.co/kyutai/pocket-tts, log in \
                 with `huggingface-cli login`, or pre-download with `hf download \
                 kyutai/pocket-tts --revision 427e3d61b276ed69fdd03de0d185fa8a8d97fc5b`"
            )
        })?;
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

    /// Pre-compute and cache the voice state for `voice` (empty = the default
    /// voice) so the first `synthesize()`/`stream()` doesn't pay it — a .wav
    /// reference clip in particular costs seconds to encode. Blocking (takes
    /// the model mutex); call from `spawn_blocking`.
    pub fn warm_voice(&self, voice: &str) -> Result<()> {
        let voice = self.resolve_voice(voice).to_string();
        let mut inner = self.inner.lock().expect("pocket-tts mutex poisoned");
        inner.ensure_voice_state(&voice)
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
            // [C, T] mono f32 tensor → raw samples.
            let audio = model.generate(&text, state)?;
            tensor_to_f32_samples(&audio)
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
                    // Each chunk is a [B, C, T] mono f32 tensor (one Mimi frame).
                    match chunk.and_then(|t| tensor_to_f32_samples(&t)) {
                        Ok(samples) => {
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
