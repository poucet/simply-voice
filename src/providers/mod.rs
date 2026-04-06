#[cfg(feature = "whisper")]
pub mod whisper;

#[cfg(feature = "whisper")]
pub use whisper::WhisperProvider;
