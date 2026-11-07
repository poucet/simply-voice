//! Voice out: the configured pocket-tts voice, sentences streamed to the
//! speaker one after another.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use simply_voice::{PocketTtsProvider, TtsProvider};
use tokio::sync::mpsc;

/// How often [`speak`] checks whether the speaker has gone quiet.
const TICK: Duration = Duration::from_millis(40);

/// `[tts]` config section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TtsConfig {
    /// pocket-tts voice: a `.safetensors` embedding, a `.wav` clip, or a stock
    /// name. `~` expands.
    pub voice: String,
    /// Stock voice used when `voice` is a path that does not exist.
    pub fallback_voice: String,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            voice: "~/.local/share/noema/models/aurora-voices/aurora.safetensors".into(),
            fallback_voice: "alba".into(),
        }
    }
}

impl TtsConfig {
    /// The voice spec to load: `voice` with `~` expanded, or the fallback
    /// when it names a file that is not there.
    pub fn resolved_voice(&self) -> String {
        let voice = crate::config::expand_home(&self.voice);
        let looks_like_path =
            voice.contains('/') || voice.ends_with(".safetensors") || voice.ends_with(".wav");
        if looks_like_path && !Path::new(&voice).exists() {
            tracing::warn!(voice, fallback = %self.fallback_voice, "voice file missing; using the fallback");
            return self.fallback_voice.clone();
        }
        voice
    }
}

/// Load pocket-tts with the configured voice, and prepare the voice so the
/// first sentence doesn't pay for it. Blocking; call from a blocking context.
pub fn load(cfg: &TtsConfig) -> Result<Arc<PocketTtsProvider>> {
    let voice = cfg.resolved_voice();
    let tts = PocketTtsProvider::new(voice.clone()).context("loading pocket-tts")?;
    tts.warm_voice("")
        .with_context(|| format!("loading voice {voice}"))?;
    Ok(Arc::new(tts))
}

/// How a [`speak`] call went.
#[derive(Debug, Default)]
pub struct Spoken {
    /// From the call to the first audio reaching the sink.
    pub first_audio: Option<Duration>,
    /// Sentences that failed to synthesise, with why; the others still played.
    pub failed: Vec<String>,
}

/// Speak `sentences` in order into `sink` and return once the speaker has gone
/// quiet. Synthesis streams: each PCM16 chunk at `sample_rate` is appended as
/// it arrives, so audio starts after the first chunk rather than the whole
/// sentence, and the next sentence synthesises while this one plays.
pub async fn speak(
    tts: &dyn TtsProvider,
    sample_rate: u32,
    sink: &rodio::Sink,
    sentences: &[String],
) -> Spoken {
    let t0 = Instant::now();
    let mut spoken = Spoken::default();
    for text in sentences.iter().filter(|s| !s.trim().is_empty()) {
        let mut chunks = match stream(tts, text).await {
            Ok(rx) => rx,
            Err(e) => {
                spoken.failed.push(format!("{text:?}: {e:#}"));
                continue;
            }
        };
        while let Some(chunk) = chunks.recv().await {
            let pcm = chunk
                .data
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0);
            sink.append(rodio::buffer::SamplesBuffer::new(
                1,
                sample_rate,
                pcm.collect::<Vec<_>>(),
            ));
            spoken.first_audio.get_or_insert_with(|| t0.elapsed());
        }
    }
    while !sink.empty() {
        tokio::time::sleep(TICK).await;
    }
    spoken
}

/// One sentence through the provider's streaming synthesis.
async fn stream(
    tts: &dyn TtsProvider,
    text: &str,
) -> Result<mpsc::Receiver<simply_voice::AudioChunk>> {
    let (text_tx, chunks) = tts.stream().await?;
    text_tx
        .send(text.to_string())
        .await
        .context("tts stream closed")?;
    Ok(chunks) // `text_tx` drops here: end of input
}

#[cfg(test)]
mod tests {
    use super::*;
    use simply_voice::{Audio, AudioChunk, Voice};
    use std::sync::Mutex;

    /// Streams 100 ms of silence per sentence and records what it was asked
    /// for; the first `fail` streams cannot be opened.
    #[derive(Clone, Default)]
    struct FakeTts {
        said: Arc<Mutex<Vec<String>>>,
        fail: Arc<Mutex<usize>>,
    }

    #[async_trait::async_trait]
    impl TtsProvider for FakeTts {
        async fn synthesize(&self, _: &str, _: &str) -> Result<Audio> {
            unimplemented!("speak streams")
        }
        async fn stream(&self) -> Result<(mpsc::Sender<String>, mpsc::Receiver<AudioChunk>)> {
            {
                let mut fail = self.fail.lock().unwrap();
                if *fail > 0 {
                    *fail -= 1;
                    anyhow::bail!("no voice");
                }
            }
            let (text_tx, mut text_rx) = mpsc::channel::<String>(4);
            let (audio_tx, audio_rx) = mpsc::channel(4);
            let said = Arc::clone(&self.said);
            tokio::spawn(async move {
                while let Some(text) = text_rx.recv().await {
                    said.lock().unwrap().push(text);
                    let _ = audio_tx.send(AudioChunk::new(vec![0u8; 2400 * 2])).await;
                }
            });
            Ok((text_tx, audio_rx))
        }
        async fn voices(&self) -> Result<Vec<Voice>> {
            Ok(vec![])
        }
    }

    /// Stand-in for a device: pull the sink's output at about real time.
    fn drain_in_real_time(mut output: impl Iterator + Send + 'static) {
        std::thread::spawn(move || {
            loop {
                for _ in 0..240 {
                    output.next();
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        });
    }

    #[tokio::test]
    async fn speaks_in_order_and_returns_once_the_speaker_is_quiet() {
        let tts = FakeTts::default();
        let (sink, output) = rodio::Sink::new_idle();
        drain_in_real_time(output);
        let sentences = ["one", "", "two"].map(String::from);
        let spoken = speak(&tts, 24_000, &sink, &sentences).await;
        assert_eq!(
            *tts.said.lock().unwrap(),
            ["one", "two"],
            "blank lines are skipped"
        );
        assert!(spoken.first_audio.is_some());
        assert!(spoken.failed.is_empty());
        assert!(sink.empty(), "returns after playback ends");
    }

    #[tokio::test]
    async fn a_failed_sentence_is_reported_and_the_rest_still_play() {
        let tts = FakeTts {
            fail: Arc::new(Mutex::new(1)),
            ..FakeTts::default()
        };
        let (sink, output) = rodio::Sink::new_idle();
        drain_in_real_time(output);
        let sentences = ["bad", "good"].map(String::from);
        let spoken = speak(&tts, 24_000, &sink, &sentences).await;
        assert_eq!(*tts.said.lock().unwrap(), ["good"]);
        assert_eq!(spoken.failed.len(), 1);
        assert!(spoken.failed[0].contains("bad"), "{:?}", spoken.failed);
    }
}
