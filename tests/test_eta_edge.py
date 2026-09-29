"""Edge cases for kws_de.eta's ledger reader and ETA rendering that the main
eta suite doesn't pin: blank-line tolerance in the JSONL ledger (appended
across runs/machines), percentile endpoints, and the singular "run" wording."""

import json

import pytest

from kws_de import eta


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "timings.jsonl"
    monkeypatch.setenv("KWS_TIMINGS", str(path))
    return path


def test_rows_tolerate_blank_lines(ledger):
    rows = [
        {"stage": "qc", "size": 100.0, "seconds": 200.0, "host": eta.host_tag()},
        {"stage": "qc", "size": 100.0, "seconds": 300.0, "host": eta.host_tag()},
    ]
    # blank lines (trailing newline, stray blank between rows) must not crash the reader
    ledger.write_text(
        json.dumps(rows[0]) + "\n\n" + json.dumps(rows[1]) + "\n\n"
    )
    pred = eta.predict("qc", size=100.0)
    assert pred is not None
    assert pred.n == 2


def test_percentile_endpoints(ledger):
    vals = [1.0, 2.0, 3.0, 4.0]
    assert eta._percentile(vals, 0.0) == 1.0  # lo == hi branch at the low end
    assert eta._percentile(vals, 100.0) == 4.0  # clamped hi index at the top end


def test_format_eta_says_single_run_not_runs(ledger):
    ledger.write_text(
        json.dumps({"stage": "x", "size": 10.0, "seconds": 60.0, "host": eta.host_tag()})
        + "\n"
    )
    pred = eta.predict("x", size=10.0)
    text = eta.format_eta(pred, size=10.0)
    assert "from 1 run" in text
    assert "runs" not in text.replace("run", "")


def test_predict_ignores_zero_size_rows(ledger):
    ledger.write_text(
        json.dumps({"stage": "z", "size": 0.0, "seconds": 999.0, "host": eta.host_tag()})
        + "\n"
    )
    # a zero-size row has no rate -> predict() must refuse rather than divide by zero
    assert eta.predict("z", size=1.0) is None

