"""kws_de.eta: ledger round-trip, the rate-median/percentile prediction math
(hand-checked), Timed, and the kws-eta CLI (predict/record/run/watch)."""

import json
import os
import re
import subprocess
import sys
import time

import pytest

from kws_de.eta import Timed, _percentile, format_eta, predict, record


def test_record_round_trips_through_the_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("KWS_TIMINGS", str(tmp_path / "timings.jsonl"))
    record("qc", size=10, seconds=5, note="hello")
    row = json.loads((tmp_path / "timings.jsonl").read_text().splitlines()[0])
    assert row["stage"] == "qc"
    assert row["size"] == 10
    assert row["seconds"] == 5
    assert row["note"] == "hello"
    assert re.fullmatch(r"[0-9a-f]{8}", row["host"])  # never the raw hostname
    assert "T" in row["ts"]  # iso8601


def test_percentile_hand_checked():
    # sorted [1,2,3,4,5]: p20 -> k=0.8 -> 1 + 0.8*(2-1) = 1.8; p80 -> k=3.2 -> 4 + 0.2*(5-4) = 4.2
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(vals, 20) == pytest.approx(1.8)
    assert _percentile(vals, 80) == pytest.approx(4.2)
    assert _percentile([7.0], 20) == 7.0


def test_predict_median_and_percentile_rate_scaled_by_size(tmp_path, monkeypatch):
    monkeypatch.setenv("KWS_TIMINGS", str(tmp_path / "timings.jsonl"))
    # size=1 for every record -> rate == seconds, so the hand-checked percentiles above apply
    for seconds in (1, 2, 3, 4, 5):
        record("build", size=1, seconds=seconds)
    pred = predict("build", 10)
    assert pred.n == 5
    assert pred.seconds == pytest.approx(30.0)  # median rate 3 * size 10
    assert pred.low == pytest.approx(18.0)  # 1.8 * 10
    assert pred.high == pytest.approx(42.0)  # 4.2 * 10


def test_predict_returns_none_on_empty_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("KWS_TIMINGS", str(tmp_path / "timings.jsonl"))
    assert predict("never-seen", 100) is None
    assert format_eta(predict("never-seen", 100), 100) == (
        "ETA unknown (first run of this stage; will be recorded)"
    )


def test_predict_excludes_failed_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("KWS_TIMINGS", str(tmp_path / "timings.jsonl"))
    record("train", size=1, seconds=100, note="failed")
    assert predict("train", 1) is None
    record("train", size=1, seconds=10)
    pred = predict("train", 1)
    assert pred.n == 1 and pred.seconds == pytest.approx(10.0)


def test_predict_uses_last_10_records_only(tmp_path, monkeypatch):
    monkeypatch.setenv("KWS_TIMINGS", str(tmp_path / "timings.jsonl"))
    for seconds in range(1, 16):  # 15 records, rate = seconds (size=1)
        record("many", size=1, seconds=seconds)
    pred = predict("many", 1)
    assert pred.n == 10  # only the last 10 (rates 6..15) count
    assert pred.seconds == pytest.approx(10.5)  # median of 6..15


def test_timed_records_success_and_reraises_on_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("KWS_TIMINGS", str(tmp_path / "timings.jsonl"))
    with Timed("stage-ok", size=5, note="v1"):
        time.sleep(0.01)
    lines = (tmp_path / "timings.jsonl").read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["note"] == "v1" and row["seconds"] > 0

    with pytest.raises(ValueError):
        with Timed("stage-fail", size=5, note="v1"):
            raise ValueError("boom")
    lines = (tmp_path / "timings.jsonl").read_text().splitlines()
    assert len(lines) == 2
    failed_row = json.loads(lines[1])
    assert failed_row["note"] == "failed"
    assert predict("stage-fail", 5) is None  # failed runs never feed a prediction


def _run_cli(args, env):
    return subprocess.run(
        [sys.executable, "-m", "kws_de.eta", *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_predict_record_and_run(tmp_path):
    env = dict(os.environ, KWS_TIMINGS=str(tmp_path / "timings.jsonl"))

    p = _run_cli(["predict", "cli-stage", "10"], env)
    assert p.returncode == 0
    assert "ETA unknown" in p.stdout

    p = _run_cli(["record", "cli-stage", "10", "20", "--note", "seed"], env)
    assert p.returncode == 0
    assert "recorded" in p.stdout

    p = _run_cli(["predict", "cli-stage", "10"], env)
    assert p.returncode == 0
    assert "ETA ~" in p.stdout

    p = _run_cli(["run", "cli-run", "1", "--", sys.executable, "-c", "print('hi')"], env)
    assert p.returncode == 0
    assert "hi" in p.stdout
    assert "ETA unknown" in p.stdout  # printed before the command ran

    fail_cmd = ["run", "cli-run-fail", "1", "--", sys.executable, "-c", "raise SystemExit(3)"]
    p = _run_cli(fail_cmd, env)
    assert p.returncode == 3
    ledger = (tmp_path / "timings.jsonl").read_text().splitlines()
    assert json.loads(ledger[-1])["note"] == "failed"


def test_cli_watch_once_reads_latest_step_and_optional_prediction(tmp_path, monkeypatch):
    env = dict(os.environ, KWS_TIMINGS=str(tmp_path / "timings.jsonl"))
    log = tmp_path / "train.log"
    log.write_text("Step #100\nsome other line\nStep #500\nStep #8200\n")

    p = _run_cli(
        ["watch", str(log), "--total", "20000", "--pattern", r"Step #(\d+)", "--once"], env
    )
    assert p.returncode == 0
    assert "step 8200/20000" in p.stdout
    assert "steps/min" in p.stdout and "ETA" in p.stdout

    monkeypatch.setenv("KWS_TIMINGS", str(tmp_path / "timings.jsonl"))
    record("mww-train", size=20000, seconds=1200)
    p = _run_cli(
        [
            "watch",
            str(log),
            "--total",
            "20000",
            "--pattern",
            r"Step #(\d+)",
            "--stage",
            "mww-train",
            "--once",
        ],
        env,
    )
    assert p.returncode == 0
    assert "predicted (history)" in p.stdout
