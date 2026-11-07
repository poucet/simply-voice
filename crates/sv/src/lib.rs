//! `sv`: speak, listen and pick audio devices from the command line, on the
//! `simply-voice` providers. The library half is what the binary is built
//! from: audio I/O, turn detection, voice out and the config.
pub mod audio;
pub mod cli;
pub mod config;
pub mod turn;
pub mod voice;
