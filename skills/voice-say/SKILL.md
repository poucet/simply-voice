---
name: voice-say
description: "Talk to the user out loud through simply-voice's `sv`: speak a short message in the Aurora voice (`say`), or ask a question and get the spoken answer back as text (`listen`). Use when the user asked to be told or asked out loud, or when long-running work finishes, fails, or is blocked on them and they may not be watching the screen."
---

# voice-say

Talk to the user out loud with simply-voice's `sv`. Run the bundled scripts
from this skill's own directory:

    <skill-dir>/scripts/say "The build is green."
    <skill-dir>/scripts/listen

They find simply-voice themselves. If `sv` isn't built yet, the first call
builds it, which is slow.

If your session already gives you a `say` tool, use that instead.

## When

- The user asked you to tell them something out loud, or to ask them by voice.
- Long work just finished, failed, or needs the user (a decision, a
  permission, a login) and they may be away from the screen.
- Once per event. Never narrate progress step by step.

## What to say

- One or two short sentences, the gist only: *"Tests pass. Ready for review."*
  *"The migration failed on the users table. I need you."*
- Plain spoken English. No code, paths, URLs, markdown, or long identifiers:
  they are read out literally. Say *"the config file"*, not
  `~/.config/simply-voice/config.toml`.
- Put the details in your normal text reply; the voice is only the headline.

## Asking: `listen`

- Ask with `say` first, then `listen`: `scripts/say "Deploy to staging or
  production?"` then `scripts/listen`.
- `listen` opens the mic, waits for the user to speak, and returns when they
  finish their turn (a pause the speech model reads as the end of what they
  meant, about 1–2 s after they stop). It prints what they said as one line
  of plain text on stdout.
- Exit codes: **0** heard (the line is the answer), **1** nobody started
  speaking within the timeout (30 s; `--timeout <secs>` to change), **2**
  error (mic, model). On 1, ask once more or fall back to text; on 2, carry
  on in text.
- The transcript is speech-to-text: read it for intent, expect small word
  errors, and confirm anything destructive before acting on it.
- Ask one question at a time, answerable in a few words.

## Mechanics

- Each `say` argument is a sentence, spoken in order: `scripts/say "Done."
  "Two tests fail."`. It blocks until playback ends (about a second per
  sentence; the first call after a reboot loads the voice model, about 9 s).
  `say` prints nothing; the exit code says whether it spoke.
- Both use the user's configured devices (`[audio]` in
  `~/.config/simply-voice/config.toml`); pass `--input <name>` /
  `--output <name>` only if asked.
- If `say` exits non-zero, carry on in text and mention that speech failed
  once; do not retry in a loop.
