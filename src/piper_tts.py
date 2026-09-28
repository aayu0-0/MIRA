"""
A5 — piper_tts.py

Synthesises text to a local WAV file via the `piper` CLI and (optionally)
plays it back using the platform's native audio player.

Two public responsibilities, deliberately separated:

  - PiperTTS.synthesize(text) -- runs the `piper` binary as a subprocess,
    passing `text` on stdin, and captures the resulting WAV at a predictable
    path under `output_dir`.  Returns a TTSResult.

  - PiperTTS.speak(text) -- synthesize() + best-effort playback.  A failed
    synthesis short-circuits before any playback attempt.  A failed playback
    never overwrites a successful synthesis result; `played` is simply False.

Two module-level helpers used by both:

  - _find_piper_binary(explicit_path) -- resolves the piper binary location:
    explicit file path → shutil.which fallback → None.
  - _player_command(wav_path) -- returns the platform-appropriate command list
    for playing a WAV file, or None if nothing is available.

A `runner` callable can be injected into PiperTTS for testing rather than
mocking subprocess directly.  The default runner calls `subprocess.run`.

Piper and the chosen voice model are both optional/external; an unavailable
binary or missing model_path returns TTSResult(success=False, source="unavailable")
without touching the runner.  Exceptions from the runner or a zero-returncode
run that produced no file both return TTSResult(success=False, source="error").
"""

import json
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional


# ─── Result ────────────────────────────────────────────────────────────────

@dataclass
class TTSResult:
    success: bool
    source: str                    # "piper" | "unavailable" | "error"
    audio_path: Optional[str] = None
    played: bool = False
    error: Optional[str] = None
    playback_error: Optional[str] = None  # set when a player was found and
                                           # started but didn't finish cleanly
                                           # (distinct from "no player found",
                                           # which just leaves played=False)


# ─── Module-level helpers ──────────────────────────────────────────────────

def _find_piper_binary(explicit_path: Optional[str]) -> Optional[str]:
    """
    Resolve the piper binary.

    1. If `explicit_path` is given, check shutil.which() first (handles bare
       names on PATH like "piper"), then treat it as a literal file path.
       Returns None if neither resolves.
    2. If `explicit_path` is None, fall back to shutil.which("piper").
    """
    if explicit_path is not None:
        # which() handles bare names on PATH
        found = shutil.which(explicit_path)
        if found:
            return explicit_path          # return the original form (tests expect this)
        if os.path.isfile(explicit_path):
            return explicit_path
        return None
    # No override: plain PATH lookup
    return shutil.which("piper")


def _player_command(wav_path: str) -> Optional[List[str]]:
    """Return the platform-appropriate command list for playing a WAV, or None."""
    system = platform.system()
    if system == "Darwin":
        return ["afplay", wav_path]
    if system == "Windows":
        # PowerShell one-liner, no extra install required
        return [
            "powershell",
            "-c",
            f"(New-Object Media.SoundPlayer '{wav_path}').PlaySync()",
        ]
    # Linux / everything else: prefer paplay (PulseAudio), fall back to aplay
    if shutil.which("paplay"):
        return ["paplay", wav_path]
    if shutil.which("aplay"):
        return ["aplay", wav_path]
    return None


def wav_duration_seconds(wav_path: str) -> Optional[float]:
    """Length of a WAV file in seconds, or None if it can't be read (missing
    file, not actually a WAV, etc). Used to size a playback timeout to the
    actual clip instead of guessing -- a fixed timeout either kills long
    audio mid-playback or wastes time waiting on short clips."""
    try:
        import wave
        with wave.open(wav_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except Exception:
        return None


# ─── Live-process registry ─────────────────────────────────────────────────
#
# Every real subprocess this module spawns (a one-shot synthesis, a playback
# command, or a long-lived --json-input session) is registered here while
# it's running. subprocess.run()'s children are NOT killed automatically
# when the parent Python process exits (notably on Windows), so a host app
# must call kill_all_active() from its own shutdown/close handler to stop
# in-flight synthesis or playback immediately instead of leaving it to run
# to completion after the window is gone.

_registry_lock = threading.Lock()
_active_procs: "set[subprocess.Popen]" = set()


def _register(proc: "subprocess.Popen") -> None:
    with _registry_lock:
        _active_procs.add(proc)


def _unregister(proc: "subprocess.Popen") -> None:
    with _registry_lock:
        _active_procs.discard(proc)


def kill_all_active() -> None:
    """Kill every piper/playback subprocess started by this module that is
    still running. Safe to call any time, including when nothing is active.
    Call this from the host app's window-close handler."""
    with _registry_lock:
        procs = list(_active_procs)
    for proc in procs:
        try:
            proc.kill()
        except Exception:
            pass


# ─── Default subprocess runner ─────────────────────────────────────────────

def _default_runner(args: List[str], input_text: str, timeout: float):
    """Real subprocess call.  Swapped via PiperTTS(runner=...) in tests.

    Uses Popen (not subprocess.run) so the child can be registered and
    killed on demand via kill_all_active() -- e.g. if the host app closes
    mid-synthesis or mid-playback.
    """
    proc = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _register(proc)
    try:
        stdout, stderr = proc.communicate(
            input=input_text.encode("utf-8"), timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise
    finally:
        _unregister(proc)
    return subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)


# ─── PiperTTS ──────────────────────────────────────────────────────────────

class PiperTTS:
    """
    Thin wrapper around the `piper` CLI for local, offline TTS.

    Instantiate once per script run, call synthesize() or speak().

    Parameters
    ----------
    model_path : str or None
        Path to the .onnx voice model file.  If None, every call returns
        TTSResult(success=False, source="unavailable").
    binary_path : str or None
        Explicit path or name for the piper binary.  Resolved via
        _find_piper_binary(); if resolution fails, returns "unavailable".
    output_dir : str
        Directory for generated WAV files.  Defaults to a system temp dir.
    play : bool
        Whether speak() should attempt playback after synthesis.
    timeout : float
        Subprocess timeout in seconds, used for the `piper` synthesis
        subprocess (and as a fallback playback timeout if a clip's
        duration can't be determined -- see playback_margin).
    playback_margin : float
        Extra seconds added on top of a clip's actual duration (read from
        the WAV header) when timing out the playback subprocess. Playback
        gets its own timeout, sized to the audio itself, rather than
        reusing `timeout` -- a fixed synthesis-sized timeout would kill
        any clip longer than that many seconds partway through.
    runner : callable or None
        Injected for tests.  Signature: (args, input_text, timeout) -> result
        with .returncode and .stderr attributes.
    keep_alive : bool
        If True (default), synthesize() first tries to reuse one long-lived
        `piper --json-input` process across calls instead of paying piper's
        model-load cost (reading the .onnx file, starting onnxruntime, the
        espeak-ng phonemizer) on every single call. Falls back automatically
        and permanently to the normal per-call path for the life of this
        instance the first time the persistent process fails to start, exits
        unexpectedly, or doesn't produce output in time -- so a piper build
        without --json-input support behaves exactly as before, just without
        the speed-up.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        binary_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        play: bool = True,
        timeout: float = 30.0,
        playback_margin: float = 15.0,
        runner: Optional[Callable] = None,
        keep_alive: bool = True,
    ):
        self.model_path = model_path
        self._binary_path = binary_path
        self.output_dir = output_dir or tempfile.gettempdir()
        self.play = play
        self.timeout = timeout
        self.playback_margin = playback_margin
        self._runner = runner or _default_runner
        self.keep_alive = keep_alive

        # Resolve binary once at construction time (mirrors HistoryStore pattern)
        self._resolved_binary = _find_piper_binary(binary_path)

        # Persistent --json-input session state
        self._persistent_proc: Optional[subprocess.Popen] = None
        self._persistent_lock = threading.Lock()
        self._persistent_broken = False

    # ── persistent --json-input session (speed-up) ──────────────────────────

    def _ensure_persistent(self) -> bool:
        """Start (or confirm alive) the long-lived piper process. Returns
        False if persistent mode is unusable, so the caller falls back."""
        if self._persistent_broken:
            return False
        if self._persistent_proc is not None and self._persistent_proc.poll() is None:
            return True
        try:
            proc = subprocess.Popen(
                [self._resolved_binary, "--model", self.model_path, "--json-input"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception:
            self._persistent_broken = True
            return False
        _register(proc)
        self._persistent_proc = proc
        return True

    def _stop_persistent(self) -> None:
        proc, self._persistent_proc = self._persistent_proc, None
        if proc is not None:
            _unregister(proc)
            try:
                proc.kill()
            except Exception:
                pass

    def _synthesize_persistent(self, text: str, wav_path: str) -> Optional[TTSResult]:
        """Try synthesizing via the persistent session. Returns a TTSResult
        on success, or None to mean "not handled -- use the normal path"."""
        with self._persistent_lock:
            if not self._ensure_persistent():
                return None
            proc = self._persistent_proc
            line = json.dumps({"text": text, "output_file": wav_path}) + "\n"
            try:
                proc.stdin.write(line.encode("utf-8"))
                proc.stdin.flush()
            except Exception:
                self._persistent_broken = True
                self._stop_persistent()
                return None

            # --json-input gives no completion signal on stdout, so poll for
            # the output file to appear and stop growing.
            deadline = time.monotonic() + self.timeout
            last_size = -1
            stable_checks = 0
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    self._persistent_broken = True
                    self._stop_persistent()
                    return None
                if os.path.isfile(wav_path):
                    size = os.path.getsize(wav_path)
                    if size > 0 and size == last_size:
                        stable_checks += 1
                        if stable_checks >= 2:
                            return TTSResult(success=True, source="piper",
                                             audio_path=wav_path)
                    else:
                        stable_checks = 0
                    last_size = size
                time.sleep(0.05)

            # Timed out -- give up on persistent mode for the rest of this run
            self._persistent_broken = True
            self._stop_persistent()
            return None

    # ── public API ─────────────────────────────────────────────────────────

    def synthesize(self, text: str,
                   output_path: Optional[str] = None) -> TTSResult:
        """
        Synthesise `text` to a WAV file.

        Returns TTSResult with success=True and audio_path set on success.
        Returns success=False with source="unavailable" if the model or binary
        are not set up; source="error" for any runtime failure.
        """
        # Pre-flight checks (no runner involved)
        if not self.model_path:
            return TTSResult(success=False, source="unavailable",
                             error="model_path is not set")
        if not self._resolved_binary:
            return TTSResult(success=False, source="unavailable",
                             error="piper binary not found")
        if not os.path.isfile(self.model_path):
            return TTSResult(success=False, source="unavailable",
                             error=f"voice model not found: {self.model_path}")
        config_path = self.model_path + ".json"
        if not os.path.isfile(config_path):
            return TTSResult(
                success=False, source="unavailable",
                error=(
                    f"voice model config not found: {config_path} "
                    "-- piper needs a matching <model>.onnx.json file sitting "
                    "next to the .onnx model. This is usually the cause of a "
                    "cryptic OS-level exit code (e.g. 3221226505 / "
                    "STATUS_DLL_NOT_FOUND) instead of a normal piper error."
                ),
            )

        # Validate text
        if not text or not text.strip():
            return TTSResult(success=False, source="error",
                             error="text must be non-empty")

        wav_path = output_path or os.path.join(self.output_dir, "voice_message.wav")

        if self.keep_alive:
            persistent_result = self._synthesize_persistent(text, wav_path)
            if persistent_result is not None:
                return persistent_result
            # Falls through to spawning piper fresh below -- either this was
            # a one-off failure this call, or persistent mode just got
            # marked broken and every future call takes this path directly.

        try:
            result = self._runner(
                [
                    self._resolved_binary,
                    "--model", self.model_path,
                    "--output_file", wav_path,
                ],
                text,
                self.timeout,
            )
        except Exception as exc:
            return TTSResult(success=False, source="error", error=str(exc))

        if result.returncode != 0:
            stderr_msg = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            return TTSResult(success=False, source="error",
                             error=stderr_msg or f"piper exited with code {result.returncode}")

        if not os.path.isfile(wav_path):
            return TTSResult(success=False, source="error",
                             error="piper exited 0 but produced no output file")

        return TTSResult(success=True, source="piper", audio_path=wav_path)

    def speak(self, text: str,
              output_path: Optional[str] = None) -> TTSResult:
        """
        synthesize() then (if play=True) attempt playback.

        A failed synthesis short-circuits; a failed playback never
        overwrites a successful synthesis result -- `played` stays False
        and, if a player was actually found and started, `playback_error`
        explains why it didn't finish (so the caller can tell that apart
        from "no player was available at all").
        """
        tts_result = self.synthesize(text, output_path=output_path)
        if not tts_result.success:
            return tts_result

        if not self.play:
            return tts_result   # played stays False

        cmd = _player_command(tts_result.audio_path)
        if cmd is None:
            # No player available -- not an error, just can't play
            return tts_result

        # Playback gets its own timeout, sized to the clip's actual length
        # (not self.timeout, which is meant for the synthesis subprocess).
        # Reusing a fixed ~30s synthesis timeout here is what was killing
        # any report long enough to take longer than that to read aloud --
        # the player process would get proc.kill()'d mid-sentence and the
        # TimeoutExpired exception below would silently swallow that,
        # reporting it as "no audio player found" even though one was
        # found and had started.
        duration = wav_duration_seconds(tts_result.audio_path)
        playback_timeout = (duration + self.playback_margin) if duration else None

        try:
            play_result = self._runner(cmd, "", playback_timeout)
            if play_result.returncode == 0:
                tts_result.played = True
            else:
                tts_result.playback_error = (
                    f"player exited with code {play_result.returncode}"
                )
        except subprocess.TimeoutExpired:
            tts_result.playback_error = (
                f"playback did not finish within {playback_timeout:.0f}s "
                f"and was stopped"
            )
        except Exception as exc:
            tts_result.playback_error = f"playback failed: {exc}"

        return tts_result