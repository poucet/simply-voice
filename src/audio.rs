use bytes::Bytes;

/// Standard audio format used across the voice pipeline.
///
/// All audio flowing through simply-voice is normalized to this format.
/// Providers handle conversion to/from their native formats internally.
pub const SAMPLE_RATE: u32 = 16_000;
pub const CHANNELS: u16 = 1;

/// A chunk of PCM audio data.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Raw PCM samples as 16-bit signed integers, little-endian.
    /// 16kHz mono.
    pub data: Bytes,
    /// Sample rate in Hz. Always 16000 for the standard format.
    pub sample_rate: u32,
}

impl AudioChunk {
    /// Create a new audio chunk from raw PCM16 LE bytes at the standard sample rate.
    pub fn new(data: impl Into<Bytes>) -> Self {
        Self {
            data: data.into(),
            sample_rate: SAMPLE_RATE,
        }
    }

    /// Create an audio chunk with a custom sample rate (for provider I/O before normalization).
    pub fn with_sample_rate(data: impl Into<Bytes>, sample_rate: u32) -> Self {
        Self {
            data: data.into(),
            sample_rate,
        }
    }

    /// Number of samples in this chunk.
    pub fn sample_count(&self) -> usize {
        self.data.len() / 2 // 16-bit = 2 bytes per sample
    }

    /// Duration in milliseconds.
    pub fn duration_ms(&self) -> f64 {
        (self.sample_count() as f64 / self.sample_rate as f64) * 1000.0
    }

    /// Returns true if this chunk contains no audio data.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
