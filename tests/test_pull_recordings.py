import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pull-recordings.sh"


def _fake_drive(root: Path) -> Path:
    mnt = root / "KWSREC"
    (mnt / "spk03" / "licht").mkdir(parents=True)
    (mnt / "spk03" / "licht" / "001.wav").write_bytes(b"RIFF" + b"\0" * 40)
    (mnt / "spk03" / "session.csv").write_text(
        'prompt,file,ms,peak_dbfs,set,seed,ts\n"Licht",spk03/licht/001.wav,900,-6.0,words,7,1234\n'
    )
    (mnt / "recognise.log").write_text("[Log] 1234 Licht 0.91\n")
    return mnt


def _run(mnt: Path, dest: Path):
    env = {**os.environ, "KWSREC_MOUNT": str(mnt), "KWSREC_NO_EJECT": "1"}
    return subprocess.run(["bash", str(SCRIPT), str(dest)], env=env, capture_output=True, text=True)


def test_pull_copies_appends_and_clears(tmp_path):
    mnt = _fake_drive(tmp_path)
    dest = tmp_path / "recordings"
    r = _run(mnt, dest)
    assert r.returncode == 0, r.stderr
    assert (dest / "spk03" / "licht" / "001.wav").exists()
    rows = (dest / "sessions.csv").read_text().splitlines()
    assert rows[0] == "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts"
    assert rows[1].startswith("spk03,") and rows[1].endswith(",1234")
    assert (dest / "logs").glob("recognise-*.log")
    assert not (mnt / "spk03").exists()  # cleared after successful copy


def test_pull_is_idempotent_and_needs_a_drive(tmp_path):
    mnt = _fake_drive(tmp_path)
    dest = tmp_path / "recordings"
    assert _run(mnt, dest).returncode == 0
    assert _run(mnt, dest).returncode == 0  # empty drive → no-op, still success
    assert len((dest / "sessions.csv").read_text().splitlines()) == 2
    env = {**os.environ, "KWSREC_MOUNT": str(tmp_path / "nope"), "KWSREC_NO_EJECT": "1"}
    r = subprocess.run(["bash", str(SCRIPT), str(dest)], env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "KWSREC" in r.stderr
