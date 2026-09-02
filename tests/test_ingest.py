import os
import stat
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ingest.sh"


def _fake_bin(d: Path, name: str, body: str) -> None:
    p = d / name
    p.write_text("#!/usr/bin/env bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


def test_ingest_orders_commands_and_pulls(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log = tmp_path / "calls.log"
    # fake ssh: record the remote command; simulate the pull dir on "host" via a local dir
    remote = tmp_path / "remote"
    (remote / "spk02" / "licht").mkdir(parents=True)
    (remote / "spk02" / "licht" / "001.wav").write_bytes(b"RIFF")
    (remote / "sessions.csv").write_text("speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts\n")
    _fake_bin(
        bin_,
        "ssh",
        f'echo "ssh $*" >> "{log}"\n'
        'case "$*" in *"ls /dev/cu.usbmodem"*) echo /dev/cu.usbmodem101;; '
        '*"ls /Volumes/KWSREC"*) echo spk02;; esac\nexit 0\n',
    )
    # fake scp: just record the call — the copy-over is validated by the real pull
    # happening via the faked ssh + rsync below, not by scp actually moving bytes.
    _fake_bin(bin_, "scp", f'echo "scp $*" >> "{log}"\nexit 0\n')
    _fake_bin(
        bin_,
        "rsync",
        f'echo "rsync $*" >> "{log}"\n'
        'src="${@: -2:1}"; dst="${@: -1}"; mkdir -p "$dst"; '
        f'cp -R "{remote}/." "$dst/"\n',
    )
    env = {
        **os.environ,
        "PATH": f"{bin_}:{os.environ['PATH']}",
        "KWS_DATA_ROOT": str(tmp_path / "root"),
    }
    r = subprocess.run(
        ["bash", str(SCRIPT), "-H", "devhost"], env=env, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    calls = log.read_text()
    assert (
        calls.index("mode usb")
        < calls.index("pull-recordings.sh")
        < calls.index("rsync")
        < calls.index("mode menu")
    )
    incoming = list((tmp_path / "root" / "data" / "recordings" / "incoming").iterdir())
    assert len(incoming) == 1 and (incoming[0] / "spk02" / "licht" / "001.wav").exists()
    assert "--delete" not in calls


def test_ingest_fails_clearly_without_port(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _fake_bin(bin_, "ssh", "exit 0\n")  # never prints a port
    env = {
        **os.environ,
        "PATH": f"{bin_}:{os.environ['PATH']}",
        "KWS_DATA_ROOT": str(tmp_path / "root"),
    }
    r = subprocess.run(
        ["bash", str(SCRIPT), "-H", "devhost"], env=env, capture_output=True, text=True
    )
    assert r.returncode == 1 and "usbmodem" in r.stderr
