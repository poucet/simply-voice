pub mod audio;
pub mod provider;
pub mod providers;
pub mod session;
pub mod vad;

pub use audio::{Audio, AudioChunk, AudioEncoding, AudioFormat};
pub use provider::{
    SttProvider, TtsProvider, RealtimeProvider,
    Transcription, Voice,
    RealtimeConfig, RealtimeEvent, RealtimeInput,
};
pub use session::{VoiceEvent, VoiceInput, VoiceState};
pub use vad::{VoiceActivityDetector, VadEvent};

#[cfg(feature = "whisper")]
pub use providers::WhisperProvider;

#[cfg(feature = "voxtral")]
pub use providers::VoxtralProvider;

#[cfg(feature = "gemini")]
pub use providers::GeminiRealtimeProvider;

#[cfg(feature = "elevenlabs")]
pub use providers::ElevenLabsProvider;
