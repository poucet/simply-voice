//! `~/.config/simply-voice/config.toml`. Every field has a default and a
//! missing file is the all-defaults config. Unknown sections and keys are
//! ignored, so a config written for another tool (or an older `va`) loads.
//!
//! ```toml
//! [audio]
//! input = "MacBook Pro Microphone"   # case-insensitive, exact then substring
//! output = "headphones"
//! [stt]
//! backend = "kyutai"                  # or "kyutai-batch", "whisper"
//! [tts]
//! voice = "~/voices/aurora.safetensors"
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::audio::AudioConfig;
use crate::turn::SttConfig;
use crate::voice::TtsConfig;

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct Config {
    pub audio: AudioConfig,
    pub stt: SttConfig,
    pub tts: TtsConfig,
}

/// Where the config comes from, plus command-line overrides applied on load —
/// the one place every subcommand gets its `Config` from.
#[derive(Debug, Clone)]
pub struct ConfigSource {
    pub path: PathBuf,
    /// `--input` / `--output`; a set field replaces the file's `[audio]` value.
    pub audio: AudioConfig,
}

impl ConfigSource {
    pub fn load(&self) -> Result<Config> {
        let mut config = Config::load(&self.path)?;
        if let Some(input) = &self.audio.input {
            config.audio.input = Some(input.clone());
        }
        if let Some(output) = &self.audio.output {
            config.audio.output = Some(output.clone());
        }
        Ok(config)
    }
}

impl Config {
    /// `~/.config/simply-voice/config.toml`.
    pub fn default_path() -> PathBuf {
        home().join(".config/simply-voice/config.toml")
    }

    /// Where the config lived when the client was `va` in voice-agent.
    pub fn legacy_path() -> PathBuf {
        home().join(".config/voice-agent/config.toml")
    }

    /// When `path` does not exist but `legacy` does, copy `legacy` to `path`
    /// (the old file stays) and say so on stderr. Returns whether it copied.
    pub fn migrate(path: &Path, legacy: &Path) -> Result<bool> {
        if path.exists() || !legacy.is_file() {
            return Ok(false);
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::copy(legacy, path)
            .with_context(|| format!("copying {} to {}", legacy.display(), path.display()))?;
        eprintln!(
            "sv: copied your config from {} to {}",
            legacy.display(),
            path.display()
        );
        Ok(true)
    }

    /// Load `path`; a missing file is the default config.
    pub fn load(path: &Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }

    /// Set one key (`section.key`) in the file at `path`, keeping everything
    /// else in it — comments, formatting, and sections this build does not
    /// know.
    pub fn persist(
        path: &Path,
        section: &str,
        key: &str,
        value: impl Into<toml_edit::Value>,
    ) -> Result<()> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        let mut doc: toml_edit::DocumentMut = text
            .parse()
            .with_context(|| format!("parsing {}", path.display()))?;
        let table = doc
            .entry(section)
            .or_insert_with(toml_edit::table)
            .as_table_like_mut()
            .ok_or_else(|| anyhow!("[{section}] in {} is not a table", path.display()))?;
        table.insert(key, toml_edit::value(value));
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, doc.to_string())?;
        Ok(())
    }
}

/// `$HOME`, or `/` when unset.
pub fn home() -> PathBuf {
    std::env::var_os("HOME").map_or_else(|| PathBuf::from("/"), PathBuf::from)
}

/// Expand a leading `~/` to [`home`].
pub fn expand_home(path: &str) -> String {
    match path.strip_prefix("~/") {
        Some(rest) => home().join(rest).to_string_lossy().into_owned(),
        None => path.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrate_copies_the_legacy_file_once_and_keeps_it() {
        let dir = std::env::temp_dir().join(format!("sv-migrate-{}", std::process::id()));
        let legacy = dir.join("voice-agent/config.toml");
        let path = dir.join("simply-voice/config.toml");
        std::fs::create_dir_all(legacy.parent().unwrap()).unwrap();
        std::fs::write(&legacy, "[pipeline]\nvoice_out = false\n").unwrap();

        assert!(Config::migrate(&path, &legacy).unwrap());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "[pipeline]\nvoice_out = false\n"
        );
        assert!(legacy.is_file(), "the old file stays");

        std::fs::write(&path, "").unwrap();
        assert!(
            !Config::migrate(&path, &legacy).unwrap(),
            "an existing config wins"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migrate_without_a_legacy_file_does_nothing() {
        let dir = std::env::temp_dir().join(format!("sv-migrate-none-{}", std::process::id()));
        let path = dir.join("simply-voice/config.toml");
        assert!(!Config::migrate(&path, &dir.join("absent.toml")).unwrap());
        assert!(!path.exists());
    }

    #[test]
    fn missing_file_is_defaults() {
        let config = Config::load(Path::new("/nonexistent/simply-voice.toml")).unwrap();
        assert_eq!(config, Config::default());
        assert_eq!(config.stt.backend, crate::turn::SttBackend::Kyutai);
    }

    #[test]
    fn sections_parse_and_unknown_keys_are_ignored() {
        // A full config from the old voice client: agent, pipeline and keys
        // sections, and keys this build does not know inside known sections.
        let config: Config = toml::from_str(
            r#"
            [agent]
            default = "mine"
            [agent.mine]
            command = "claude-agent-acp"
            [pipeline]
            voice_out = false
            [audio]
            input = "MacBook"
            echo = "strict"
            [stt]
            backend = "kyutai-batch"
            eot_horizon = 1
            [tts]
            voice = "alba"
            [keys]
            stop = "q"
            "#,
        )
        .unwrap();
        assert_eq!(config.audio.input.as_deref(), Some("MacBook"));
        assert_eq!(config.stt.backend, crate::turn::SttBackend::KyutaiBatch);
        assert_eq!(config.stt.eot_horizon, Some(1));
        assert_eq!(config.tts.voice, "alba");
    }

    #[test]
    fn persist_sets_one_key_and_keeps_the_rest() {
        let path = std::env::temp_dir().join(format!("sv-config-{}.toml", std::process::id()));
        std::fs::write(
            &path,
            "# mine\n[audio]\ninput = \"mic\"\n[pipeline]\nlisten_mode = \"push-to-talk\"\n",
        )
        .unwrap();
        Config::persist(&path, "audio", "output", "speakers").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            text.contains("# mine") && text.contains("listen_mode = \"push-to-talk\""),
            "{text}"
        );
        let config = Config::load(&path).unwrap();
        assert_eq!(config.audio.input.as_deref(), Some("mic"));
        assert_eq!(config.audio.output.as_deref(), Some("speakers"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn source_overrides_audio_devices_only_when_set() {
        let dir = std::env::temp_dir().join(format!("sv-src-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            "[audio]\ninput = \"file-in\"\noutput = \"file-out\"\n",
        )
        .unwrap();
        let source = ConfigSource {
            path,
            audio: AudioConfig {
                input: Some("flag-in".into()),
                output: None,
            },
        };
        let config = source.load().unwrap();
        assert_eq!(config.audio.input.as_deref(), Some("flag-in"));
        assert_eq!(config.audio.output.as_deref(), Some("file-out"));
    }
}
