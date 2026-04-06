#[cfg(feature = "whisper")]
pub mod whisper;

#[cfg(feature = "voxtral")]
pub mod voxtral;

#[cfg(feature = "gemini")]
pub mod gemini;

#[cfg(feature = "elevenlabs")]
pub mod elevenlabs;

#[cfg(feature = "whisper")]
pub use whisper::WhisperProvider;

#[cfg(feature = "voxtral")]
pub use voxtral::VoxtralProvider;

#[cfg(feature = "gemini")]
pub use gemini::GeminiRealtimeProvider;

#[cfg(feature = "elevenlabs")]
pub use elevenlabs::ElevenLabsProvider;
