import numpy as np
from kws_de import config
from kws_de.eval import metrics, render_report, snr_sweep


def test_metrics_perfect_and_unknown_fa():
    y = np.array(
        [config.label_index(label) for label in ["Licht", "Wasser", "_unknown_", "_silence_"]]
    )
    perfect = metrics(y, y.copy())
    assert perfect["accuracy"] == 1.0
    assert perfect["unknown_false_accept"] == 0.0
    # unknown predicted as a command -> false accept = 1.0
    yp = y.copy()
    yp[2] = config.label_index("Licht")
    assert metrics(y, yp)["unknown_false_accept"] == 1.0


def test_snr_sweep_and_report():
    sweep = snr_sweep(lambda s: 0.9 if s >= 10 else 0.5, [20, 10, 0])
    assert sweep[20] == 0.9 and sweep[0] == 0.5
    md = render_report({"accuracy": 0.9, "snr_sweep": sweep})
    assert "Accuracy" in md and "SNR" in md
