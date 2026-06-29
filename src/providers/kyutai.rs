//! Kyutai STT provider — fully local streaming speech-to-text via the moshi
//! candle port (kyutai-labs delayed-streams-modeling). Model weights
//! auto-download from Hugging Face on first load.
//
// VERIFY (offline assumptions, mirrors kyutai stt-rs example verbatim):
// - moshi 0.6.1 API: asr::State::new(1, delay, 0., mimi, lm), state.step_pcm(pcm,
//   None, &().into(), |_, _, _| ()) -> Vec<AsrMsg>, AsrMsg::{Word{tokens,..},
//   EndWord{..}, Step{prs,..}}, lm::Config/ExtraHeadsConfig fields as in the
//   upstream stt-rs example pinned to the same versions.
// - moshi::asr::State, mimi, LmModel and sentencepiece::SentencePieceProcessor
//   are Send (required for Arc<Mutex<Inner>> to cross into spawn_blocking).
// - step_pcm accepts a final partial (<1920 sample) frame, as upstream does.
// - prs semantics: P(no voice activity) at horizons [0.5s, 1s, 2s, 3s]; only
//   valid because we always load with the VAD extra heads enabled.

use anyhow::{Context, Result};
use async_trait::async_trait;
use candle::{Device, Tensor};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tracing::{debug, error, info};

use crate::audio::AudioChunk;
use crate::provider::{SttProvider, Transcription};

/// Default HF repo for the 1B English/French streaming STT model.
const DEFAULT_HF_REPO: &str = "kyutai/stt-1b-en_fr-candle";
/// Weights file within the repo.
const MODEL_FILE: &str = "model.safetensors";
/// The model consumes 24 kHz mono f32 PCM.
const MODEL_SAMPLE_RATE: u32 = 24_000;
/// The model steps in 80 ms frames (12.5 Hz token rate).
const STEP_SAMPLES: usize = 1920;
/// Token frame rate, used to convert the ASR delay from seconds to tokens.
const FRAMES_PER_SECOND: f64 = 12.5;
/// End-of-turn detection uses the VAD extra head at this horizon index.
/// `prs` = P(no voice activity) at horizons 0.5s, 1s, 2s, 3s — index 2 = 2s.
const EOT_HORIZON: usize = 2;
/// P(no voice activity) above this marks end of turn.
const EOT_THRESHOLD: f32 = 0.5;

/// Model config as stored in the HF repo's config.json
/// (mirrors kyutai's stt-rs example).
#[derive(Debug, serde::Deserialize)]
struct SttConfig {
    audio_silence_prefix_seconds: f64,
    audio_delay_seconds: f64,
}

#[derive(Debug, serde::Deserialize)]
struct Config {
    mimi_name: String,
    tokenizer_name: String,
    card: usize,
    text_card: usize,
    dim: usize,
    n_q: usize,
    context: usize,
    max_period: f64,
    num_heads: usize,
    num_layers: usize,
    causal: bool,
    stt_config: SttConfig,
}

impl Config {
    fn model_config(&self, vad: bool) -> moshi::lm::Config {
        let lm_cfg = moshi::transformer::Config {
            d_model: self.dim,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            dim_feedforward: self.dim * 4,
            causal: self.causal,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: self.context,
            max_period: self.max_period as usize,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: moshi::NormType::RmsNorm,
            positional_embedding: moshi::transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096 * 4,
            shared_cross_attn: false,
        };
        let extra_heads = if vad {
            Some(moshi::lm::ExtraHeadsConfig {
                num_heads: 4,
                dim: 6,
            })
        } else {
            None
        };
        moshi::lm::Config {
            transformer: lm_cfg,
            depformer: None,
            audio_vocab_size: self.card + 1,
            text_in_vocab_size: self.text_card + 1,
            text_out_vocab_size: self.text_card,
            audio_codebooks: self.n_q,
            conditioners: Default::default(),
            extra_heads,
        }
    }
}

/// Events distilled from one model step, decoupled from moshi types so the
/// stream loop stays simple.
enum SttEvent {
    /// A recognized word (already decoded).
    Word(String),
    /// The VAD head crossed the end-of-turn threshold.
    EndOfTurn,
}

/// The stateful model. `moshi::asr::State` owns the weights and its streaming
/// KV/conv caches, so everything lives behind one mutex.
struct Inner {
    state: moshi::asr::State,
    text_tokenizer: sentencepiece::SentencePieceProcessor,
    config: Config,
    dev: Device,
    /// Set once end-of-turn fired, cleared on the next word — avoids emitting
    /// EndOfTurn on every step while silence continues.
    eot_fired: bool,
}

impl Inner {
    /// Step the model over one frame of 24 kHz f32 PCM (normally
    /// `STEP_SAMPLES` long; the final frame of a batch may be shorter).
    fn step(&mut self, frame: &[f32]) -> Result<Vec<SttEvent>> {
        let pcm = Tensor::new(frame, &self.dev)?.reshape((1, 1, ()))?;
        let asr_msgs = self.state.step_pcm(pcm, None, &().into(), |_, _, _| ())?;
        let mut events = Vec::new();
        for asr_msg in asr_msgs.iter() {
            match asr_msg {
                moshi::asr::AsrMsg::Step { prs, .. } => {
                    if prs[EOT_HORIZON][0] > EOT_THRESHOLD && !self.eot_fired {
                        self.eot_fired = true;
                        debug!(pr = prs[EOT_HORIZON][0], "kyutai: end of turn");
                        events.push(SttEvent::EndOfTurn);
                    }
                }
                moshi::asr::AsrMsg::EndWord { .. } => {
                    self.eot_fired = false;
                }
                moshi::asr::AsrMsg::Word { tokens, .. } => {
                    self.eot_fired = false;
                    let word = self
                        .text_tokenizer
                        .decode_piece_ids(tokens)
                        .unwrap_or_else(|_| String::new());
                    if !word.is_empty() {
                        events.push(SttEvent::Word(word));
                    }
                }
            }
        }
        Ok(events)
    }
}

/// Streaming local STT on Kyutai's delayed-streams model.
///
/// The moshi ASR state is stateful and owns the model weights, so the provider
/// serializes all use behind a mutex: **only one `stream()` or `transcribe()`
/// call can run at a time**, and the model's streaming state (KV caches, VAD)
/// **persists across calls** — a batch `transcribe()` between streams will
/// leave its audio in the model's context.
pub struct KyutaiSttProvider {
    inner: Arc<Mutex<Inner>>,
}

impl KyutaiSttProvider {
    /// Load the model. `hf_repo` defaults to `kyutai/stt-1b-en_fr-candle`;
    /// `cpu` forces CPU (otherwise cuda -> metal -> cpu). Weights download
    /// from the Hugging Face hub on first run (HF_TOKEN env if needed).
    /// The VAD extra heads are always enabled — end-of-turn detection
    /// depends on them.
    pub fn new(hf_repo: Option<String>, cpu: bool) -> Result<Self> {
        let hf_repo = hf_repo.unwrap_or_else(|| DEFAULT_HF_REPO.to_string());
        let dev = Self::device(cpu)?;
        info!(repo = %hf_repo, device = ?dev, "kyutai: loading STT model");

        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(hf_repo.clone());
        let config_file = repo.get("config.json")?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_file)?)
            .with_context(|| format!("kyutai: bad config.json in '{hf_repo}'"))?;
        let tokenizer_file = repo.get(&config.tokenizer_name)?;
        let model_file = repo.get(MODEL_FILE)?;
        let mimi_file = repo.get(&config.mimi_name)?;
        let is_quantized = model_file.to_str().unwrap().ends_with(".gguf");

        let text_tokenizer = sentencepiece::SentencePieceProcessor::open(&tokenizer_file)?;

        let lm = if is_quantized {
            let vb_lm = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &model_file,
                &dev,
            )?;
            moshi::lm::LmModel::new(
                &config.model_config(true),
                moshi::nn::MaybeQuantizedVarBuilder::Quantized(vb_lm),
            )?
        } else {
            let dtype = dev.bf16_default_to_f32();
            let vb_lm = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&[&model_file], dtype, &dev)?
            };
            moshi::lm::LmModel::new(
                &config.model_config(true),
                moshi::nn::MaybeQuantizedVarBuilder::Real(vb_lm),
            )?
        };

        let audio_tokenizer = moshi::mimi::load(mimi_file.to_str().unwrap(), Some(32), &dev)?;
        let asr_delay_in_tokens =
            (config.stt_config.audio_delay_seconds * FRAMES_PER_SECOND) as usize;
        let state = moshi::asr::State::new(1, asr_delay_in_tokens, 0., audio_tokenizer, lm)?;
        info!("kyutai: STT model loaded");

        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                state,
                text_tokenizer,
                config,
                dev,
                eot_fired: false,
            })),
        })
    }

    fn device(cpu: bool) -> Result<Device> {
        if cpu {
            Ok(Device::Cpu)
        } else if candle::utils::cuda_is_available() {
            Ok(Device::new_cuda(0)?)
        } else if candle::utils::metal_is_available() {
            Ok(Device::new_metal(0)?)
        } else {
            Ok(Device::Cpu)
        }
    }

    /// Convert PCM16 LE bytes to f32 samples.
    fn pcm16_to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32 / 32768.0
            })
            .collect()
    }

    /// Linear interpolation resampler — fine for speech, avoids pulling in
    /// rubato/kaudio. Used to lift pipeline audio (16 kHz) to the model's
    /// 24 kHz input rate.
    fn resample_linear(src: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
        if src.is_empty() || src_rate == dst_rate {
            return src.to_vec();
        }
        let ratio = src_rate as f64 / dst_rate as f64;
        let out_len = (src.len() as f64 / ratio) as usize;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let pos = i as f64 * ratio;
            let idx = pos as usize;
            let frac = (pos - idx as f64) as f32;
            let a = src.get(idx).copied().unwrap_or(0.0);
            let b = src.get(idx + 1).copied().unwrap_or(a);
            out.push(a + (b - a) * frac);
        }
        out
    }

    /// Pipeline PCM16 @ 16 kHz -> model f32 @ 24 kHz.
    fn to_model_pcm(data: &[u8]) -> Vec<f32> {
        let samples = Self::pcm16_to_f32(data);
        Self::resample_linear(&samples, crate::audio::SAMPLE_RATE, MODEL_SAMPLE_RATE)
    }
}

#[async_trait]
impl SttProvider for KyutaiSttProvider {
    async fn transcribe(&self, audio: AudioChunk) -> Result<Transcription> {
        let inner = Arc::clone(&self.inner);
        let mut pcm = Self::to_model_pcm(&audio.data);

        let text = tokio::task::spawn_blocking(move || -> Result<String> {
            let mut inner = inner.lock().expect("kyutai mutex poisoned");

            // Pad like the upstream example: leading silence to warm up the
            // model, trailing silence to flush words held back by the ASR
            // delay (plus a second of margin).
            let prefix =
                (inner.config.stt_config.audio_silence_prefix_seconds * MODEL_SAMPLE_RATE as f64)
                    as usize;
            if prefix > 0 {
                pcm.splice(0..0, vec![0.0; prefix]);
            }
            let suffix = (inner.config.stt_config.audio_delay_seconds * MODEL_SAMPLE_RATE as f64)
                as usize;
            pcm.resize(pcm.len() + suffix + MODEL_SAMPLE_RATE as usize, 0.0);

            let mut words = Vec::new();
            for frame in pcm.chunks(STEP_SAMPLES) {
                for event in inner.step(frame)? {
                    if let SttEvent::Word(word) = event {
                        words.push(word);
                    }
                }
            }
            Ok(words.join(" "))
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

        let inner = Arc::clone(&self.inner);

        tokio::task::spawn_blocking(move || {
            // The lock is held for the life of the stream — see the type-level
            // doc comment about single-session use.
            let mut inner = inner.lock().expect("kyutai mutex poisoned");

            // Resampled 24 kHz samples waiting to fill a full model frame.
            let mut pending: Vec<f32> = Vec::new();
            // Words of the utterance in progress, flushed at end of turn.
            let mut utterance: Vec<String> = Vec::new();

            while let Some(chunk) = audio_rx.blocking_recv() {
                pending.extend_from_slice(&Self::to_model_pcm(&chunk.data));

                while pending.len() >= STEP_SAMPLES {
                    let frame: Vec<f32> = pending.drain(..STEP_SAMPLES).collect();
                    let events = match inner.step(&frame) {
                        Ok(events) => events,
                        Err(e) => {
                            error!("kyutai stream step failed: {e:#}");
                            return;
                        }
                    };
                    for event in events {
                        let msg = match event {
                            // Interim result: one word at a time.
                            SttEvent::Word(word) => {
                                utterance.push(word.clone());
                                Transcription {
                                    text: word,
                                    is_final: false,
                                }
                            }
                            // End of turn: flush the accumulated utterance.
                            SttEvent::EndOfTurn => {
                                if utterance.is_empty() {
                                    continue;
                                }
                                let text = utterance.join(" ");
                                utterance.clear();
                                Transcription {
                                    text,
                                    is_final: true,
                                }
                            }
                        };
                        if text_tx.blocking_send(msg).is_err() {
                            debug!("kyutai stream: receiver dropped");
                            return;
                        }
                    }
                }
            }

            // Sender dropped: flush whatever remains as a final result.
            if !utterance.is_empty() {
                let _ = text_tx.blocking_send(Transcription {
                    text: utterance.join(" "),
                    is_final: true,
                });
            }
        });

        Ok((audio_tx, text_rx))
    }
}
