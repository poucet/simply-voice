use bytes::Bytes;
use serde::{Deserialize, Serialize};

/// Standard audio format used across the voice pipeline.
pub const SAMPLE_RATE: u32 = 16_000;
pub const CHANNELS: u16 = 1;

/// Describes the encoding of audio data in a stream or response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioEncoding {
    /// 16-bit signed integers, little-endian (2 bytes per sample).
    Pcm16,
    /// 32-bit IEEE float, little-endian (4 bytes per sample).
    F32,
}

/// Describes the format of an audio stream.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AudioFormat {
    pub encoding: AudioEncoding,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioFormat {
    pub fn pcm16(sample_rate: u32) -> Self {
        Self { encoding: AudioEncoding::Pcm16, sample_rate, channels: 1 }
    }

    pub fn f32(sample_rate: u32) -> Self {
        Self { encoding: AudioEncoding::F32, sample_rate, channels: 1 }
    }

    /// Bytes per sample for this encoding.
    pub fn bytes_per_sample(&self) -> usize {
        match self.encoding {
            AudioEncoding::Pcm16 => 2,
            AudioEncoding::F32 => 4,
        }
    }
}

/// A chunk of raw audio data. The format is described at the stream level,
/// not per-chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioChunk {
    /// Raw audio bytes in the stream's format.
    #[serde(with = "bytes_serde")]
    pub data: Bytes,
}

/// Audio data with its format — used for self-describing responses
/// like TTS synthesis where there's no persistent stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Audio {
    pub format: AudioFormat,
    #[serde(with = "bytes_serde")]
    pub data: Bytes,
}

impl AudioChunk {
    pub fn new(data: impl Into<Bytes>) -> Self {
        Self { data: data.into() }
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Audio {
    /// Create audio from raw f32 LE bytes.
    pub fn from_f32_bytes(data: impl Into<Bytes>, sample_rate: u32) -> Self {
        Self {
            format: AudioFormat::f32(sample_rate),
            data: data.into(),
        }
    }

    /// Create audio from PCM16 LE bytes.
    pub fn from_pcm16(data: impl Into<Bytes>, sample_rate: u32) -> Self {
        Self {
            format: AudioFormat::pcm16(sample_rate),
            data: data.into(),
        }
    }

    /// Convert to f32 samples regardless of encoding.
    pub fn to_f32_samples(&self) -> Vec<f32> {
        match self.format.encoding {
            AudioEncoding::F32 => {
                self.data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            AudioEncoding::Pcm16 => {
                self.data.chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                    .collect()
            }
        }
    }

    /// Number of samples.
    pub fn sample_count(&self) -> usize {
        self.data.len() / self.format.bytes_per_sample()
    }

    /// Duration in milliseconds.
    pub fn duration_ms(&self) -> f64 {
        self.sample_count() as f64 / self.format.sample_rate as f64 * 1000.0
    }
}

mod bytes_serde {
    use bytes::Bytes;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(data: &Bytes, serializer: S) -> Result<S::Ok, S::Error> {
        data.as_ref().serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Bytes, D::Error> {
        let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
        Ok(Bytes::from(bytes))
    }
}
