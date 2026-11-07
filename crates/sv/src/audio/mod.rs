//! Native audio I/O: cpal capture, rodio playback, WAV files.
//!
//! Copied and trimmed from lumina's `aurora/src/audio.rs` (aurora is not a
//! dependency). Capture: cpal's `Stream` is `!Send` on macOS, so the stream
//! is built and owned on a dedicated OS thread that parks on a stop channel —
//! dropping the stream on that thread stops capture. Every cpal sample format
//! is downmixed to mono and linear-resampled to [`SAMPLE_RATE`] PCM16, and
//! handed to a caller-supplied callback that must not block.
//!
//! Playback mirrors the pattern: rodio's `OutputStream` is `!Send`, so it
//! lives on its own thread for the life of the [`Playback`], while the `Sink`
//! (which IS `Send`) is handed back to the caller.

pub mod devices;

use std::path::Path;
use std::thread;

use anyhow::{Context, Result, anyhow};
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use serde::{Deserialize, Serialize};
use simply_voice::Audio;

/// Everything upstream of the speaker runs on mono PCM16 at this rate.
pub const SAMPLE_RATE: u32 = simply_voice::audio::SAMPLE_RATE;

/// RMS level (0..1) above which a frame counts as voiced. The same threshold
/// `simply_voice::vad` uses.
pub const VOICED_RMS: f32 = 0.01;

/// `[audio]` config section.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    /// Input device name (case-insensitive, exact then substring); unset =
    /// the system default.
    pub input: Option<String>,
    /// Output device name, same matching; unset = the system default.
    pub output: Option<String>,
}

// ---------------------------------------------------------------------------
// Microphone capture
// ---------------------------------------------------------------------------

/// Handle to an active mic capture. Dropping it (or [`MicCapture::stop`])
/// signals the cpal thread to drop its stream, halting capture.
pub struct MicCapture {
    name: String,
    stream_stop: std::sync::mpsc::Sender<()>,
}

impl MicCapture {
    /// Start capturing from the named input device (`None` = system default).
    /// `on_frame` receives mono PCM16 at [`SAMPLE_RATE`] on the realtime audio
    /// thread, so it must not block — send it into a channel.
    pub fn start(
        device: Option<&str>,
        on_frame: impl FnMut(Vec<i16>) + Send + 'static,
    ) -> Result<Self> {
        let device = devices::resolve_input(device)?;
        let name = devices::display_name(&device);
        let config = device
            .default_input_config()
            .context("default input config")?;
        let sample_format = config.sample_format();
        let rate = config.sample_rate().0;
        let channels = config.channels();
        let stream_config: StreamConfig = config.into();
        tracing::info!(device = %name, rate, channels, "mic capture starting");

        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();
        thread::spawn(move || {
            let mut on_frame = on_frame;
            let err_fn = |e| tracing::error!(error = %e, "cpal input stream error");
            let built = match sample_format {
                SampleFormat::F32 => device.build_input_stream(
                    &stream_config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        on_frame(to_pcm16(data.iter().copied(), channels, rate))
                    },
                    err_fn,
                    None,
                ),
                SampleFormat::I16 => device.build_input_stream(
                    &stream_config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        on_frame(to_pcm16(
                            data.iter().map(|&s| s as f32 / 32768.0),
                            channels,
                            rate,
                        ))
                    },
                    err_fn,
                    None,
                ),
                SampleFormat::U16 => device.build_input_stream(
                    &stream_config,
                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                        on_frame(to_pcm16(
                            data.iter().map(|&s| (s as i32 - 32768) as f32 / 32768.0),
                            channels,
                            rate,
                        ))
                    },
                    err_fn,
                    None,
                ),
                other => {
                    let _ = ready_tx.send(Err(anyhow!("unsupported sample format {other:?}")));
                    return;
                }
            };
            let stream = match built.map_err(|e| anyhow!("build input stream: {e}")) {
                Ok(s) => s,
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            if let Err(e) = stream.play() {
                let _ = ready_tx.send(Err(anyhow!("start input stream: {e}")));
                return;
            }
            let _ = ready_tx.send(Ok(()));
            // Park until asked to stop; dropping `stream` halts capture.
            let _ = stop_rx.recv();
            drop(stream);
        });
        ready_rx
            .recv()
            .context("mic thread died before starting")??;
        Ok(Self {
            name,
            stream_stop: stop_tx,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    /// Stop capturing. Also happens on drop.
    pub fn stop(&self) {
        let _ = self.stream_stop.send(());
    }
}

impl Drop for MicCapture {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Interleaved samples in -1..1 at `rate` → mono PCM16 at [`SAMPLE_RATE`].
fn to_pcm16(data: impl Iterator<Item = f32>, channels: u16, rate: u32) -> Vec<i16> {
    let data: Vec<f32> = data.collect();
    let mono: Vec<f32> = if channels <= 1 {
        data
    } else {
        let ch = channels as usize;
        data.chunks(ch)
            .map(|f| f.iter().sum::<f32>() / ch as f32)
            .collect()
    };
    f32_to_pcm16(&resample_linear(&mono, rate, SAMPLE_RATE))
}

/// Clamp and quantise f32 samples to PCM16.
pub fn f32_to_pcm16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect()
}

/// Linear interpolation resampler — fine for speech, avoids pulling rubato.
pub fn resample_linear(src: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src.is_empty() || src_rate == dst_rate {
        return src.to_vec();
    }
    let ratio = src_rate as f64 / dst_rate as f64;
    let out_len = (src.len() as f64 / ratio) as usize;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos as usize;
            let frac = (pos - idx as f64) as f32;
            let a = src.get(idx).copied().unwrap_or(0.0);
            let b = src.get(idx + 1).copied().unwrap_or(a);
            a + (b - a) * frac
        })
        .collect()
}

/// RMS level of a PCM16 frame, 0..1.
pub fn rms(frame: &[i16]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum: f64 = frame.iter().map(|&s| (s as f64 / 32768.0).powi(2)).sum();
    (sum / frame.len() as f64).sqrt() as f32
}

/// PCM16 samples → little-endian bytes, the `simply_voice::AudioChunk` wire form.
pub fn pcm16_bytes(frame: &[i16]) -> Vec<u8> {
    frame.iter().flat_map(|s| s.to_le_bytes()).collect()
}

// ---------------------------------------------------------------------------
// WAV files
// ---------------------------------------------------------------------------

/// Read a WAV file as mono PCM16 at [`SAMPLE_RATE`], whatever its format.
pub fn read_wav(path: &Path) -> Result<Vec<i16>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|s| s as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    Ok(to_pcm16(
        samples.into_iter(),
        spec.channels,
        spec.sample_rate,
    ))
}

/// Write synthesized audio as a mono PCM16 WAV at its own sample rate.
pub fn write_wav(path: &Path, audio: &Audio) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio.format.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create {}", path.display()))?;
    for s in f32_to_pcm16(&audio.to_f32_samples()) {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Speaker playback
// ---------------------------------------------------------------------------

/// Speaker output. The `!Send` rodio `OutputStream` is owned by a dedicated
/// thread; the sink is handed back.
pub struct Playback {
    sink: rodio::Sink,
    stream_stop: std::sync::mpsc::Sender<()>,
}

impl Playback {
    /// Open the named output device (`None` = system default). An unmatched
    /// name falls back to the default with a warning so playback keeps working.
    pub fn new(device: Option<&str>) -> Result<Self> {
        let device = device.and_then(|q| {
            devices::resolve_output(Some(q))
                .inspect_err(|e| tracing::warn!(error = %e, "falling back to the default output"))
                .ok()
        });
        let name = match &device {
            Some(d) => devices::display_name(d),
            None => devices::resolve_output(None)
                .map(|d| devices::display_name(&d))
                .unwrap_or_else(|_| "<unknown>".into()),
        };
        tracing::info!(device = %name, "playback opening");

        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let (sink_tx, sink_rx) = std::sync::mpsc::channel::<Result<rodio::Sink>>();
        thread::spawn(move || {
            let pair = match &device {
                Some(d) => rodio::OutputStream::try_from_device(d),
                None => rodio::OutputStream::try_default(),
            };
            let (stream, handle) = match pair {
                Ok(pair) => pair,
                Err(e) => {
                    let _ = sink_tx.send(Err(anyhow!("open output device: {e}")));
                    return;
                }
            };
            match rodio::Sink::try_new(&handle) {
                Ok(s) => {
                    let _ = sink_tx.send(Ok(s));
                }
                Err(e) => {
                    let _ = sink_tx.send(Err(anyhow!("create sink: {e}")));
                    return;
                }
            }
            let _ = stop_rx.recv();
            drop(stream);
        });
        let sink = sink_rx
            .recv()
            .context("rodio thread died before handing back a sink")??;
        Ok(Self {
            sink,
            stream_stop: stop_tx,
        })
    }

    /// The shared sink [`crate::voice::speak`] appends to.
    pub fn sink(&self) -> &rodio::Sink {
        &self.sink
    }
}

impl Drop for Playback {
    fn drop(&mut self) {
        self.sink.stop();
        let _ = self.stream_stop.send(());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stereo_48k_downmixes_and_resamples_to_16k_mono() {
        // 30 ms of stereo 48 kHz: left 0.5, right -0.5 → mono 0.
        let data: Vec<f32> = (0..1440).flat_map(|_| [0.5, -0.5]).collect();
        let out = to_pcm16(data.into_iter(), 2, 48_000);
        assert_eq!(out.len(), 480);
        assert!(out.iter().all(|&s| s == 0));
    }

    #[test]
    fn rms_of_silence_and_full_scale() {
        assert_eq!(rms(&[]), 0.0);
        assert_eq!(rms(&[0; 160]), 0.0);
        assert!((rms(&[i16::MAX; 160]) - 1.0).abs() < 1e-3);
    }
}
