"""Unit tests for the Claude Code hooks in .claude/hooks/.

Plain unit tests: each hook is run as a subprocess with crafted PreToolUse/
PostToolUse JSON on stdin, no network and no device involved.
"""

import json
import subprocess
from pathlib import Path

HOOKS = Path(__file__).resolve().parents[1] / ".claude" / "hooks"


def _run(hook: str, tool_input: dict, tool_name: str = "Bash") -> subprocess.CompletedProcess:
    payload = json.dumps({"tool_name": tool_name, "tool_input": tool_input})
    return subprocess.run(
        ["bash", str(HOOKS / hook)], input=payload, capture_output=True, text=True
    )


# --- protect-generated.sh ---------------------------------------------------------------


def test_protect_generated_blocks_gen_dir():
    r = _run("protect-generated.sh", {"file_path": "firmware/main/gen/labels.h"}, "Edit")
    assert r.returncode == 2
    assert "kws-fwgen" in r.stderr


def test_protect_generated_blocks_intent_cases():
    r = _run("protect-generated.sh", {"file_path": "firmware/test/intent_cases.h"}, "Edit")
    assert r.returncode == 2
    assert "gen-intent-cases.py" in r.stderr


def test_protect_generated_allows_other_files():
    r = _run("protect-generated.sh", {"file_path": "firmware/main/intent.c"}, "Edit")
    assert r.returncode == 0


# --- guard-git.sh ------------------------------------------------------------------------


def test_guard_git_blocks_add_dash_a():
    r = _run("guard-git.sh", {"command": "git add -A"})
    assert r.returncode == 2
    assert "stages everything" in r.stderr


def test_guard_git_blocks_add_dot():
    r = _run("guard-git.sh", {"command": "git add ."})
    assert r.returncode == 2


def test_guard_git_allows_add_specific_files():
    r = _run("guard-git.sh", {"command": "git add foo.py bar.py"})
    assert r.returncode == 0


def test_guard_git_blocks_commit_no_verify():
    r = _run("guard-git.sh", {"command": "git commit -m x --no-verify"})
    assert r.returncode == 2
    assert "--no-verify" in r.stderr


def test_guard_git_allows_plain_commit():
    r = _run("guard-git.sh", {"command": "git commit -m x"})
    assert r.returncode == 0


def test_guard_git_blocks_force_push_without_lease():
    r = _run("guard-git.sh", {"command": "git push --force origin main"})
    assert r.returncode == 2
    assert "force-with-lease" in r.stderr


def test_guard_git_allows_force_with_lease():
    r = _run("guard-git.sh", {"command": "git push --force-with-lease origin main"})
    assert r.returncode == 0


def test_guard_git_blocks_pr_create_without_base():
    r = _run("guard-git.sh", {"command": "gh pr create --title x --body y"})
    assert r.returncode == 2
    assert "--base main" in r.stderr


def test_guard_git_blocks_pr_create_with_other_base():
    r = _run("guard-git.sh", {"command": "gh pr create --base feature/foo --title x"})
    assert r.returncode == 2


def test_guard_git_allows_pr_create_with_base_main():
    r = _run("guard-git.sh", {"command": "gh pr create --base main --title x"})
    assert r.returncode == 0


def test_guard_git_ignores_unrelated_commands():
    r = _run("guard-git.sh", {"command": "ls -la"})
    assert r.returncode == 0


# --- audio-gate.sh -----------------------------------------------------------------------


def _tts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "dir1"
    d.mkdir()
    (d / "x.wav").touch()
    (d / "bad.wav").touch()
    (d / "tts_check.csv").write_text(
        "file,voice,engine,ok,reason,language,transcript\n"
        "x.wav,Anna,say,1,,de,Hallo\n"
        "bad.wav,Anna,say,0,too english,en,Hello\n"
    )
    return d


def test_audio_gate_allows_checked_ok_wav(tmp_path):
    d = _tts_dir(tmp_path)
    r = _run("audio-gate.sh", {"command": f"afplay {d / 'x.wav'}"})
    assert r.returncode == 0


def test_audio_gate_blocks_checked_failed_wav(tmp_path):
    d = _tts_dir(tmp_path)
    r = _run("audio-gate.sh", {"command": f"afplay {d / 'bad.wav'}"})
    assert r.returncode == 2
    assert "kws-tts-check" in r.stderr


def test_audio_gate_blocks_unchecked_wav(tmp_path):
    d = _tts_dir(tmp_path)
    r = _run("audio-gate.sh", {"command": f"afplay {d / 'nope.wav'}"})
    assert r.returncode == 2
    assert "kws-tts-check" in r.stderr


def test_audio_gate_blocks_unresolvable_remote_path_without_staging_dir():
    r = _run("audio-gate.sh", {"command": "ssh bar afplay ~/kws-walk/x.wav"})
    assert r.returncode == 2
    assert "KWS_TTS_DIR" in r.stderr


def test_audio_gate_maps_unresolvable_path_to_staging_dir(tmp_path, monkeypatch):
    d = _tts_dir(tmp_path)
    monkeypatch.setenv("KWS_TTS_DIR", str(d))
    r = _run("audio-gate.sh", {"command": "ssh bar afplay ~/kws-walk/x.wav"})
    assert r.returncode == 0


def test_audio_gate_blocks_non_german_voice():
    r = _run("audio-gate.sh", {"command": 'say -v "Samantha" hallo'})
    assert r.returncode == 2
    assert "German" in r.stderr


def test_audio_gate_allows_german_voice():
    r = _run("audio-gate.sh", {"command": 'say -v "Anna (German)" hallo'})
    assert r.returncode == 0


def test_audio_gate_bypass_env():
    r = _run("audio-gate.sh", {"command": "KWS_AUDIO_GATE=0 afplay /does/not/exist.wav"})
    assert r.returncode == 0


def test_audio_gate_ignores_unrelated_commands():
    r = _run("audio-gate.sh", {"command": "ls -la"})
    assert r.returncode == 0
