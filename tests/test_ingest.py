import os
import stat
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ingest.sh"


def _fake_bin(d: Path, name: str, body: str) -> None:
    p = d / name
    p.write_text("#!/usr/bin/env bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


def _remote_fixture(tmp_path: Path) -> Path:
    # simulates the pull dir pull-recordings.sh would have staged on the host
    remote = tmp_path / "remote"
    (remote / "spk02" / "licht").mkdir(parents=True)
    (remote / "spk02" / "licht" / "001.wav").write_bytes(b"RIFF")
    (remote / "sessions.csv").write_text(
        "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts\n"
        'spk02,2026-09-02T00:00:00Z,"Licht",spk02/licht/001.wav,900,-6.0,words,7,1234\n'
    )
    return remote


def _fake_ssh(bin_: Path, log: Path, remote: Path, wav_count_override: str | None = None) -> None:
    # answers the two probes (usbmodem port, KWSREC mount) and the post-rsync wav-count
    # verification query (matched on "find " — the only ssh call that uses it); anything
    # else (mode usb/menu, running pull-recordings.sh) is a no-op.
    count_cmd = (
        f"echo {wav_count_override}"
        if wav_count_override
        else f'find "{remote}" -name "*.wav" | wc -l'
    )
    _fake_bin(
        bin_,
        "ssh",
        f'echo "ssh $*" >> "{log}"\n'
        'case "$*" in '
        '*"ls /dev/cu.usbmodem"*) echo /dev/cu.usbmodem101;; '
        '*"ls /Volumes/KWSREC"*) echo spk02;; '
        f'*"find "*) {count_cmd};; '
        "esac\nexit 0\n",
    )


def _fake_scp_and_rsync(bin_: Path, log: Path, remote: Path) -> None:
    # scp: just record the call — the copy-over is validated by the faked ssh + rsync below.
    _fake_bin(bin_, "scp", f'echo "scp $*" >> "{log}"\nexit 0\n')
    _fake_bin(
        bin_,
        "rsync",
        f'echo "rsync $*" >> "{log}"\n'
        'src="${@: -2:1}"; dst="${@: -1}"; mkdir -p "$dst"; '
        f'cp -R "{remote}/." "$dst/"\n',
    )


def _env(tmp_path: Path, bin_: Path) -> dict:
    return {
        **os.environ,
        "PATH": f"{bin_}:{os.environ['PATH']}",
        "KWS_DATA_ROOT": str(tmp_path / "root"),
    }


def test_ingest_orders_commands_and_pulls(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log = tmp_path / "calls.log"
    remote = _remote_fixture(tmp_path)
    _fake_ssh(bin_, log, remote)
    _fake_scp_and_rsync(bin_, log, remote)
    r = subprocess.run(
        ["bash", str(SCRIPT), "-H", "devhost"],
        env=_env(tmp_path, bin_),
        capture_output=True,
        text=True,
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
    assert "rm -rf" not in calls  # the host-side stamped stage is never wiped by us
    assert "ingested 1 takes" in r.stdout


def test_ingest_fails_clearly_without_port(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _fake_bin(bin_, "ssh", "exit 0\n")  # never prints a port
    r = subprocess.run(
        ["bash", str(SCRIPT), "-H", "devhost"],
        env=_env(tmp_path, bin_),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1 and "usbmodem" in r.stderr


def test_ingest_ssh_unreachable_exits_3(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _fake_bin(bin_, "ssh", "exit 255\n")  # connection/auth failure, not "device not found"
    r = subprocess.run(
        ["bash", str(SCRIPT), "-H", "devhost"],
        env=_env(tmp_path, bin_),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 3 and "cannot reach host" in r.stderr


def test_ingest_wav_count_mismatch_exits_nonzero_and_keeps_host_copy(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log = tmp_path / "calls.log"
    remote = _remote_fixture(tmp_path)
    _fake_ssh(
        bin_, log, remote, wav_count_override="2"
    )  # host claims 2 wavs, local rsync only got 1
    _fake_scp_and_rsync(bin_, log, remote)
    r = subprocess.run(
        ["bash", str(SCRIPT), "-H", "devhost"],
        env=_env(tmp_path, bin_),
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "wav count mismatch" in r.stderr
    assert "devhost:~/kwsrec-pull" in r.stderr  # names the host dir that still holds the data
