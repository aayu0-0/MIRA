"""
test_piper_tts.py — piper_tts.py (local TTS wrapper)

Uses PiperTTS's injected `runner` callable throughout, per the module's own
design ("A runner callable can be injected into PiperTTS for testing rather
than mocking subprocess directly") -- no real `piper` binary, no real audio
playback, no network.

Covers:
  - _find_piper_binary(): explicit path / PATH lookup / not-found cases
  - _player_command(): per-platform command selection
  - PiperTTS.synthesize(): unavailable (no model/binary), empty text,
    successful run, non-zero exit, exit-0-but-no-file, runner exception
  - PiperTTS.speak(): playback success/failure/no-player, and that a failed
    synthesis short-circuits before any playback attempt
  - kill_all_active(): registered fake processes get killed
"""

import os
import sys
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import piper_tts
from piper_tts import (
    PiperTTS, TTSResult, _find_piper_binary, _player_command,
    _register, _unregister, kill_all_active, _active_procs,
)


def _fake_runner_factory(returncode=0, stderr=b"", create_file=True):
    """Builds a runner(args, input_text, timeout) that mimics subprocess.run,
    optionally creating the requested --output_file so synthesize() sees a
    real file on disk."""
    def runner(args, input_text, timeout):
        if create_file and "--output_file" in args:
            idx = args.index("--output_file")
            out_path = args[idx + 1]
            with open(out_path, "wb") as f:
                f.write(b"RIFF....WAVEfake")
        return SimpleNamespace(returncode=returncode, stdout=b"", stderr=stderr)
    return runner


class TestFindPiperBinary(unittest.TestCase):

    def test_explicit_existing_file_path_returned(self):
        with tempfile.NamedTemporaryFile() as f:
            self.assertEqual(_find_piper_binary(f.name), f.name)

    def test_explicit_nonexistent_path_returns_none(self):
        self.assertIsNone(_find_piper_binary("/no/such/piper/binary/anywhere"))

    def test_explicit_bare_name_on_path_resolved_via_which(self):
        with mock.patch.object(shutil, "which", return_value="/usr/bin/piper"):
            self.assertEqual(_find_piper_binary("piper"), "piper")

    def test_none_falls_back_to_plain_path_lookup(self):
        with mock.patch.object(shutil, "which", return_value="/usr/local/bin/piper") as m:
            result = _find_piper_binary(None)
            m.assert_called_with("piper")
            self.assertEqual(result, "/usr/local/bin/piper")

    def test_none_and_not_on_path_returns_none(self):
        with mock.patch.object(shutil, "which", return_value=None):
            self.assertIsNone(_find_piper_binary(None))


class TestPlayerCommand(unittest.TestCase):

    def test_darwin_uses_afplay(self):
        with mock.patch.object(piper_tts.platform, "system", return_value="Darwin"):
            cmd = _player_command("/tmp/out.wav")
        self.assertEqual(cmd, ["afplay", "/tmp/out.wav"])

    def test_windows_uses_powershell(self):
        with mock.patch.object(piper_tts.platform, "system", return_value="Windows"):
            cmd = _player_command("C:\\out.wav")
        self.assertEqual(cmd[0], "powershell")
        self.assertIn("C:\\out.wav", cmd[-1])

    def test_linux_prefers_paplay(self):
        with mock.patch.object(piper_tts.platform, "system", return_value="Linux"), \
             mock.patch.object(shutil, "which", side_effect=lambda x: "/usr/bin/paplay" if x == "paplay" else None):
            cmd = _player_command("/tmp/out.wav")
        self.assertEqual(cmd, ["paplay", "/tmp/out.wav"])

    def test_linux_falls_back_to_aplay(self):
        with mock.patch.object(piper_tts.platform, "system", return_value="Linux"), \
             mock.patch.object(shutil, "which", side_effect=lambda x: "/usr/bin/aplay" if x == "aplay" else None):
            cmd = _player_command("/tmp/out.wav")
        self.assertEqual(cmd, ["aplay", "/tmp/out.wav"])

    def test_linux_no_player_returns_none(self):
        with mock.patch.object(piper_tts.platform, "system", return_value="Linux"), \
             mock.patch.object(shutil, "which", return_value=None):
            self.assertIsNone(_player_command("/tmp/out.wav"))


class TestSynthesize(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # synthesize() pre-flight-checks that model_path (and its sibling
        # <model>.onnx.json) actually exist on disk before ever touching the
        # injected runner, so the fixture needs real files here -- a fake
        # nonexistent path would always short-circuit to "unavailable".
        self.model_path = os.path.join(self.tmpdir, "model.onnx")
        open(self.model_path, "wb").close()
        open(self.model_path + ".json", "wb").close()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _tts(self, **overrides):
        kwargs = dict(
            model_path=self.model_path,
            binary_path=__file__,          # any real file -> resolves via os.path.isfile
            output_dir=self.tmpdir,
            play=False,
            keep_alive=False,
            runner=_fake_runner_factory(),
        )
        kwargs.update(overrides)
        return PiperTTS(**kwargs)

    def test_no_model_path_is_unavailable(self):
        tts = self._tts(model_path=None)
        result = tts.synthesize("hello")
        self.assertFalse(result.success)
        self.assertEqual(result.source, "unavailable")

    def test_binary_not_found_is_unavailable(self):
        tts = self._tts(binary_path="/no/such/binary/at/all")
        result = tts.synthesize("hello")
        self.assertFalse(result.success)
        self.assertEqual(result.source, "unavailable")

    def test_empty_text_is_error(self):
        tts = self._tts()
        result = tts.synthesize("   ")
        self.assertFalse(result.success)
        self.assertEqual(result.source, "error")

    def test_successful_synthesis(self):
        tts = self._tts()
        out_path = os.path.join(self.tmpdir, "out.wav")
        result = tts.synthesize("hello world", output_path=out_path)
        self.assertTrue(result.success)
        self.assertEqual(result.source, "piper")
        self.assertEqual(result.audio_path, out_path)
        self.assertTrue(os.path.isfile(out_path))

    def test_default_output_path_used_when_not_given(self):
        tts = self._tts()
        result = tts.synthesize("hello world")
        self.assertTrue(result.success)
        self.assertEqual(result.audio_path, os.path.join(self.tmpdir, "voice_message.wav"))

    def test_nonzero_returncode_is_error_with_stderr_message(self):
        tts = self._tts(runner=_fake_runner_factory(returncode=1, stderr=b"model load failed"))
        result = tts.synthesize("hello")
        self.assertFalse(result.success)
        self.assertEqual(result.source, "error")
        self.assertIn("model load failed", result.error)

    def test_nonzero_returncode_no_stderr_uses_generic_message(self):
        tts = self._tts(runner=_fake_runner_factory(returncode=2, stderr=b""))
        result = tts.synthesize("hello")
        self.assertFalse(result.success)
        self.assertIn("2", result.error)

    def test_zero_returncode_but_no_file_is_error(self):
        tts = self._tts(runner=_fake_runner_factory(returncode=0, create_file=False))
        result = tts.synthesize("hello")
        self.assertFalse(result.success)
        self.assertEqual(result.source, "error")
        self.assertIn("no output file", result.error)

    def test_runner_exception_is_caught_as_error(self):
        def boom(args, text, timeout):
            raise RuntimeError("subprocess exploded")
        tts = self._tts(runner=boom)
        result = tts.synthesize("hello")
        self.assertFalse(result.success)
        self.assertEqual(result.source, "error")
        self.assertIn("subprocess exploded", result.error)

    def test_persistent_mode_broken_falls_back_to_normal_path(self):
        tts = self._tts(keep_alive=True)
        tts._persistent_broken = True   # simulate a prior persistent-mode failure
        result = tts.synthesize("hello world")
        self.assertTrue(result.success)  # still succeeds via the normal runner path


class TestSpeak(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.tmpdir, "model.onnx")
        open(self.model_path, "wb").close()
        open(self.model_path + ".json", "wb").close()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _tts(self, **overrides):
        kwargs = dict(
            model_path=self.model_path,
            binary_path=__file__,
            output_dir=self.tmpdir,
            play=True,
            keep_alive=False,
            runner=_fake_runner_factory(),
        )
        kwargs.update(overrides)
        return PiperTTS(**kwargs)

    def test_failed_synthesis_short_circuits_before_playback(self):
        tts = self._tts(model_path=None)
        with mock.patch("piper_tts._player_command") as pc:
            result = tts.speak("hello")
        pc.assert_not_called()
        self.assertFalse(result.success)

    def test_play_false_never_attempts_playback(self):
        tts = self._tts(play=False)
        with mock.patch("piper_tts._player_command") as pc:
            result = tts.speak("hello")
        pc.assert_not_called()
        self.assertTrue(result.success)
        self.assertFalse(result.played)

    def test_successful_playback_sets_played_true(self):
        tts = self._tts()
        with mock.patch("piper_tts._player_command", return_value=["echo", "playing"]):
            result = tts.speak("hello")
        self.assertTrue(result.success)
        self.assertTrue(result.played)

    def test_no_player_available_leaves_played_false_but_still_success(self):
        tts = self._tts()
        with mock.patch("piper_tts._player_command", return_value=None):
            result = tts.speak("hello")
        self.assertTrue(result.success)
        self.assertFalse(result.played)

    def test_playback_nonzero_returncode_leaves_played_false(self):
        # runner for synth succeeds (creates file); playback call uses the
        # same runner but the fake ignores --output_file, so it'll return
        # the configured returncode without creating anything extra.
        play_runner = _fake_runner_factory(returncode=0)

        call_count = {"n": 0}
        real_runner = _fake_runner_factory(returncode=0)

        def runner(args, text, timeout):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return real_runner(args, text, timeout)  # synth call
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"no audio device")

        tts = self._tts(runner=runner)
        with mock.patch("piper_tts._player_command", return_value=["aplay", "x.wav"]):
            result = tts.speak("hello")
        self.assertTrue(result.success)     # synthesis still succeeded
        self.assertFalse(result.played)     # but playback did not

    def test_playback_exception_swallowed_synthesis_result_kept(self):
        call_count = {"n": 0}
        real_runner = _fake_runner_factory(returncode=0)

        def runner(args, text, timeout):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return real_runner(args, text, timeout)
            raise RuntimeError("player crashed")

        tts = self._tts(runner=runner)
        with mock.patch("piper_tts._player_command", return_value=["some_player", "x.wav"]):
            result = tts.speak("hello")
        self.assertTrue(result.success)
        self.assertFalse(result.played)


class TestKillAllActive(unittest.TestCase):

    def test_kills_every_registered_process(self):
        fake1 = mock.Mock()
        fake2 = mock.Mock()
        _register(fake1)
        _register(fake2)
        try:
            kill_all_active()
            fake1.kill.assert_called_once()
            fake2.kill.assert_called_once()
        finally:
            _unregister(fake1)
            _unregister(fake2)

    def test_safe_to_call_with_nothing_registered(self):
        try:
            kill_all_active()
        except Exception as exc:
            self.fail(f"kill_all_active raised with nothing registered: {exc}")

    def test_exception_from_one_kill_does_not_block_others(self):
        broken = mock.Mock()
        broken.kill.side_effect = RuntimeError("already dead")
        fine = mock.Mock()
        _register(broken)
        _register(fine)
        try:
            kill_all_active()
            fine.kill.assert_called_once()
        finally:
            _unregister(broken)
            _unregister(fine)


class TestTTSResultDataclass(unittest.TestCase):

    def test_defaults(self):
        r = TTSResult(success=True, source="piper")
        self.assertIsNone(r.audio_path)
        self.assertFalse(r.played)
        self.assertIsNone(r.error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
