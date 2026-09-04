import csv
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
    assert rows[0] == (
        "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts,fire_ms,wake_prob,"
        "device_intent,device_words,window_ms"
    )
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


def _fake_drive_with_field(root: Path) -> Path:
    mnt = _fake_drive(root)
    (mnt / "field" / "spk03").mkdir(parents=True)
    (mnt / "field" / "spk03" / "1-123456.wav").write_bytes(b"RIFF" + b"\0" * 40)
    (mnt / "field" / "spk03" / "field.csv").write_text(
        "file,fire_ms,wake_prob,device_intent,device_words,window_ms,ms,peak_dbfs\n"
        "1-123456.wav,123456,0.910,Licht an,Licht:0.93|an:0.88,2500,3500,-8.4\n"
    )
    return mnt


def test_pull_copies_field_takes_and_appends_device_columns(tmp_path):
    mnt = _fake_drive_with_field(tmp_path)
    dest = tmp_path / "recordings"
    r = _run(mnt, dest)
    assert r.returncode == 0, r.stderr
    assert (dest / "field" / "spk03" / "1-123456.wav").exists()
    assert not (dest / "field" / "spk03" / "field.csv").exists()  # the CSV folds into sessions.csv

    rows = list(csv.DictReader((dest / "sessions.csv").open()))
    field = [r for r in rows if r["set"] == "field"]
    assert len(field) == 1
    assert field[0]["speaker"] == "spk03"
    assert field[0]["prompt"] == ""
    assert field[0]["file"] == "field/spk03/1-123456.wav"
    assert field[0]["ms"] == "3500" and field[0]["peak_dbfs"] == "-8.4"
    assert field[0]["fire_ms"] == "123456" and field[0]["ts"] == "123456"
    assert field[0]["wake_prob"] == "0.910"
    assert field[0]["device_intent"] == "Licht an"
    assert field[0]["device_words"] == "Licht:0.93|an:0.88"
    # window_ms rides through: with ms it is the only thing that lets the host
    # tell a ring-truncated take from one the recogniser simply never answered.
    assert field[0]["window_ms"] == "2500"
    assert not (mnt / "field" / "spk03").exists()  # cleared after a successful copy
