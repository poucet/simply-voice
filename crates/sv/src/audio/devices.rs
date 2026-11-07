//! Audio device enumeration and name-based selection. Copied from lumina's
//! `aurora/src/devices.rs` (aurora is not a dependency).
//!
//! Matching rule for user-supplied names: a number is the 1-based position in
//! `sv devices`; otherwise case-insensitive exact match first, then
//! case-insensitive substring; the first enumerated match wins.

use anyhow::{Result, anyhow, ensure};
use cpal::traits::{DeviceTrait, HostTrait};

/// One enumerated device, with the system-default flag.
pub struct DeviceInfo {
    pub name: String,
    pub is_default: bool,
}

/// A device's name for display/logging (cpal name lookups can fail).
pub fn display_name(device: &cpal::Device) -> String {
    device.name().unwrap_or_else(|_| "<unknown>".into())
}

/// All input devices, marking the system default.
pub fn list_inputs() -> Result<Vec<DeviceInfo>> {
    let host = cpal::default_host();
    let default = host.default_input_device().map(|d| display_name(&d));
    let devices = host
        .input_devices()
        .map_err(|e| anyhow!("enumerate input devices: {e}"))?;
    Ok(to_infos(devices, default))
}

/// All output devices, marking the system default.
pub fn list_outputs() -> Result<Vec<DeviceInfo>> {
    let host = cpal::default_host();
    let default = host.default_output_device().map(|d| display_name(&d));
    let devices = host
        .output_devices()
        .map_err(|e| anyhow!("enumerate output devices: {e}"))?;
    Ok(to_infos(devices, default))
}

fn to_infos(
    devices: impl Iterator<Item = cpal::Device>,
    default: Option<String>,
) -> Vec<DeviceInfo> {
    devices
        .map(|d| {
            let name = display_name(&d);
            let is_default = default.as_deref() == Some(name.as_str());
            DeviceInfo { name, is_default }
        })
        .collect()
}

/// Resolve an input device by name (`None` = system default).
pub fn resolve_input(name: Option<&str>) -> Result<cpal::Device> {
    let host = cpal::default_host();
    match name {
        None => host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device")),
        Some(query) => {
            let devices: Vec<_> = host
                .input_devices()
                .map_err(|e| anyhow!("enumerate input devices: {e}"))?
                .collect();
            match_by_name(devices, query, "input")
        }
    }
}

/// Resolve an output device by name (`None` = system default).
pub fn resolve_output(name: Option<&str>) -> Result<cpal::Device> {
    let host = cpal::default_host();
    match name {
        None => host
            .default_output_device()
            .ok_or_else(|| anyhow!("no default output device")),
        Some(query) => {
            let devices: Vec<_> = host
                .output_devices()
                .map_err(|e| anyhow!("enumerate output devices: {e}"))?
                .collect();
            match_by_name(devices, query, "output")
        }
    }
}

/// The index in `names` that `query` selects: a 1-based number (as `sv
/// devices` prints), else case-insensitive exact, else substring.
pub fn match_index(names: &[String], query: &str) -> Option<usize> {
    let q = query.trim().to_lowercase();
    if let Ok(n) = q.parse::<usize>() {
        return (1..=names.len()).contains(&n).then(|| n - 1);
    }
    let lower: Vec<String> = names.iter().map(|n| n.to_lowercase()).collect();
    lower
        .iter()
        .position(|n| *n == q)
        .or_else(|| lower.iter().position(|n| n.contains(&q)))
}

/// [`match_index`] over devices; the error lists every available name so the
/// user can correct the query.
fn match_by_name(devices: Vec<cpal::Device>, query: &str, kind: &str) -> Result<cpal::Device> {
    ensure!(!query.trim().is_empty(), "empty {kind} device name");
    let names: Vec<String> = devices.iter().map(display_name).collect();
    let idx = match_index(&names, query);
    match idx {
        Some(i) => Ok(devices.into_iter().nth(i).expect("matched index in range")),
        None => Err(anyhow!(
            "no {kind} device matching '{query}' — available: {}",
            if names.is_empty() {
                "(none)".to_string()
            } else {
                names.join(", ")
            }
        )),
    }
}

#[cfg(test)]
mod match_tests {
    use super::match_index;

    #[test]
    fn number_then_exact_then_substring() {
        let names: Vec<String> = ["BlackHole 2ch", "MacBook Pro Microphone", "Mic"]
            .map(String::from)
            .into();
        assert_eq!(match_index(&names, "2"), Some(1));
        assert_eq!(match_index(&names, "4"), None);
        assert_eq!(match_index(&names, "0"), None);
        assert_eq!(match_index(&names, "mic"), Some(2));
        assert_eq!(match_index(&names, "macbook"), Some(1));
        assert_eq!(match_index(&names, "nope"), None);
    }
}
