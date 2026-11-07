//! The three commands: `sv say`, `sv listen` and `sv devices`. Headless —
//! they use the mic and speaker but open no window.

use std::io::Write;
use std::path::PathBuf;
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use simply_voice::TtsProvider;
use tokio::sync::mpsc;

use crate::audio::{self, MicCapture, Playback, SAMPLE_RATE, devices};
use crate::config::{Config, ConfigSource};
use crate::turn::{self, Heard, SttBackend, TurnEvent, TurnInput};
use crate::voice;

/// Logs go to stderr so stdout stays the events; `RUST_LOG` overrides.
pub fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "warn,simply_voice_cli=info,simply_voice=info".into());
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

fn ms(d: Duration) -> u128 {
    d.as_millis()
}

// ---------------------------------------------------------------------------
// sv listen
// ---------------------------------------------------------------------------

/// How long `sv listen` waits for someone to start speaking before it gives
/// up with [`EXIT_TIMEOUT`].
pub const DEFAULT_LISTEN_TIMEOUT_SECS: f32 = 30.0;

/// `sv listen` exit code: nobody started speaking within `--timeout`.
pub const EXIT_TIMEOUT: u8 = 1;
/// Exit code for any failure: device, model, config.
pub const EXIT_ERROR: u8 = 2;

#[derive(clap::Args)]
pub struct ListenArgs {
    /// Seconds to wait for speech to start; on timeout exit 1 and print nothing.
    #[arg(long, default_value_t = DEFAULT_LISTEN_TIMEOUT_SECS)]
    pub timeout: f32,
    /// Instead of one turn: print every SpeechStart/Partial/Final with
    /// timings until Ctrl-C (or the end of --wav), for measuring.
    #[arg(long)]
    pub events: bool,
    /// Feed this WAV, paced in real time, instead of the mic.
    #[arg(long)]
    pub wav: Option<PathBuf>,
    /// Override `stt.backend`.
    #[arg(long, value_enum)]
    pub backend: Option<SttBackend>,
    /// With --wav: release push-to-talk the moment the file ends, to measure
    /// the forced end of utterance.
    #[arg(long)]
    pub ptt: bool,
    /// With --wav: seconds of silence fed after the file, so the backend can
    /// call end of turn on its own.
    #[arg(long, default_value_t = 4.0)]
    pub tail: f32,
}

/// How a one-turn `sv listen` ended.
pub enum Listened {
    /// The transcript of the first complete turn.
    Heard(String),
    /// Nobody started speaking within the timeout, or the input ended first.
    Nothing,
}

/// One turn by default; with `--events`, the timing stream.
pub async fn listen(mut config: Config, args: ListenArgs) -> Result<Listened> {
    init_tracing();
    if let Some(b) = args.backend {
        config.stt.backend = b;
    }
    let t_load = Instant::now();
    let stt = config.stt.clone();
    let rt = tokio::runtime::Handle::current();
    let detector = tokio::task::spawn_blocking(move || turn::load(&stt, rt)).await??;
    eprintln!(
        "model load ({:?}): {} ms",
        config.stt.backend,
        ms(t_load.elapsed())
    );

    let (tx, rx) = std_mpsc::channel::<TurnInput>();
    let mut heard = turn::spawn(detector, rx);
    let t0 = Instant::now();

    // Keep the capture alive for the whole run; a WAV feeder owns `tx` and
    // drops it at the end of the file, which ends the run.
    let _mic = match &args.wav {
        Some(path) => {
            let pcm = audio::read_wav(path)?;
            let secs = pcm.len() as f32 / SAMPLE_RATE as f32;
            eprintln!("feeding {} ({secs:.2} s) in real time", path.display());
            let _ = feed_wav(pcm, args.ptt, args.tail, tx);
            None
        }
        None => {
            let mic = MicCapture::start(config.audio.input.as_deref(), move |frame| {
                let _ = tx.send(TurnInput::Frame(frame));
            })?;
            eprintln!("listening on {}", mic.device_name());
            Some(mic)
        }
    };

    if !args.events {
        return one_turn(&mut heard, Duration::from_secs_f32(args.timeout)).await;
    }
    loop {
        let h = tokio::select! {
            h = heard.recv() => match h { Some(h) => h, None => break },
            _ = tokio::signal::ctrl_c() => break,
        };
        let t = t0.elapsed().as_secs_f32();
        match h.event {
            TurnEvent::SpeechStart => println!("{t:8.3}s  SpeechStart"),
            TurnEvent::Partial(p) => println!("{t:8.3}s  Partial  {p}"),
            TurnEvent::Final(f) => match h.since_voice {
                Some(d) => println!(
                    "{t:8.3}s  Final    {f}    [end of voice → Final {} ms]",
                    ms(d)
                ),
                None => println!("{t:8.3}s  Final    {f}"),
            },
        }
    }
    Ok(Listened::Nothing)
}

/// Wait for the first `Final`. The timeout only runs until speech starts:
/// once someone is talking, the turn ends when the detector says so.
async fn one_turn(
    heard: &mut mpsc::UnboundedReceiver<Heard>,
    timeout: Duration,
) -> Result<Listened> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut started = false;
    loop {
        let next = if started {
            heard.recv().await
        } else {
            match tokio::time::timeout_at(deadline, heard.recv()).await {
                Ok(h) => h,
                Err(_) => return Ok(Listened::Nothing),
            }
        };
        let Some(h) = next else {
            return Ok(Listened::Nothing);
        };
        match h.event {
            TurnEvent::SpeechStart | TurnEvent::Partial(_) => started = true,
            TurnEvent::Final(text) => return Ok(Listened::Heard(text)),
        }
    }
}

/// Feed `pcm` in 20 ms frames on the wall clock, then `tail` seconds of
/// silence; `ptt` sends a push-to-talk release where the file ends.
fn feed_wav(
    pcm: Vec<i16>,
    ptt: bool,
    tail: f32,
    tx: std_mpsc::Sender<TurnInput>,
) -> std::thread::JoinHandle<()> {
    const FRAME: usize = SAMPLE_RATE as usize / 50;
    std::thread::spawn(move || {
        let start = Instant::now();
        let silence = vec![0i16; (tail * SAMPLE_RATE as f32) as usize];
        let file_frames = pcm.len().div_ceil(FRAME);
        let frames = pcm.chunks(FRAME).chain(silence.chunks(FRAME));
        for (i, frame) in frames.enumerate() {
            if ptt && i == file_frames && tx.send(TurnInput::EndUtterance).is_err() {
                return;
            }
            if tx.send(TurnInput::Frame(frame.to_vec())).is_err() {
                return;
            }
            let due = start + Duration::from_millis(20 * (i as u64 + 1));
            std::thread::sleep(due.saturating_duration_since(Instant::now()));
        }
    })
}

// ---------------------------------------------------------------------------
// sv say
// ---------------------------------------------------------------------------

#[derive(clap::Args)]
pub struct SayArgs {
    /// Sentences to speak, in order.
    #[arg(required = true)]
    pub text: Vec<String>,
    /// Override `tts.voice` (a .safetensors/.wav path or a stock name).
    #[arg(long)]
    pub voice: Option<String>,
    /// Write the synthesised audio to this WAV instead of playing it.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

/// Speak, and return once playback has ended. Fails if any sentence could
/// not be synthesised (the others are still spoken).
pub async fn say(mut config: Config, args: SayArgs) -> Result<()> {
    init_tracing();
    if let Some(v) = args.voice {
        config.tts.voice = v;
    }
    let t_load = Instant::now();
    let tts_cfg = config.tts.clone();
    let tts = tokio::task::spawn_blocking(move || voice::load(&tts_cfg)).await??;
    println!(
        "model + voice load ({}): {} ms",
        config.tts.resolved_voice(),
        ms(t_load.elapsed())
    );

    if let Some(out) = args.out {
        let text = args.text.join(" ");
        let t = Instant::now();
        let audio = tts.synthesize(&text, "").await?;
        let took = t.elapsed();
        println!(
            "synthesised {:.0} ms of audio in {} ms (RTF {:.2})",
            audio.duration_ms(),
            ms(took),
            took.as_secs_f64() * 1000.0 / audio.duration_ms()
        );
        audio::write_wav(&out, &audio).with_context(|| format!("writing {}", out.display()))?;
        println!("wrote {}", out.display());
        return Ok(());
    }

    let playback = Playback::new(config.audio.output.as_deref())?;
    let t = Instant::now();
    let spoken = voice::speak(tts.as_ref(), tts.sample_rate(), playback.sink(), &args.text).await;
    if let Some(first) = spoken.first_audio {
        println!("time to first audio: {} ms", ms(first));
    }
    println!("finished after {} ms", ms(t.elapsed()));
    if !spoken.failed.is_empty() {
        bail!("synthesis failed: {}", spoken.failed.join("; "));
    }
    Ok(())
}

/// `sv devices [--pick]`: list both directions; with `--pick`, ask for each
/// and save the chosen *name* (numbers shift as devices come and go).
pub fn devices(source: &ConfigSource, pick: bool) -> Result<()> {
    let config = source.load()?;
    let sections = [
        (
            "input",
            devices::list_inputs()?,
            devices::resolve_input(config.audio.input.as_deref()),
        ),
        (
            "output",
            devices::list_outputs()?,
            devices::resolve_output(config.audio.output.as_deref()),
        ),
    ];
    for (kind, list, chosen) in sections {
        println!("{}:", kind[..1].to_uppercase() + &kind[1..]);
        let chosen = chosen.map(|d| devices::display_name(&d));
        for (i, d) in list.iter().enumerate() {
            let mark = if chosen.as_deref().ok() == Some(d.name.as_str()) {
                '>'
            } else {
                ' '
            };
            let default = if d.is_default { '*' } else { ' ' };
            println!("  {mark}{default} {:>2}. {}", i + 1, d.name);
        }
        if let Err(e) = &chosen {
            println!("  ! {e}");
        }
        if pick {
            let names: Vec<String> = list.into_iter().map(|d| d.name).collect();
            if let Some(name) = ask_device(kind, &names)? {
                Config::persist(&source.path, "audio", kind, name.as_str())?;
                println!("  saved {kind} = {name:?} to {}", source.path.display());
            }
        }
    }
    Ok(())
}

/// Prompt until the answer selects a device; empty keeps the current one.
fn ask_device(kind: &str, names: &[String]) -> Result<Option<String>> {
    loop {
        print!("  {kind} (number or name, Enter to keep): ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line)? == 0 || line.trim().is_empty() {
            return Ok(None);
        }
        match devices::match_index(names, &line) {
            Some(i) => return Ok(Some(names[i].clone())),
            None => println!("  no {kind} device matches {:?}", line.trim()),
        }
    }
}
