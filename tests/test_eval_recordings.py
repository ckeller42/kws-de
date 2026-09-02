import numpy as np
import soundfile as sf

from kws_de import config
from kws_de import eval as ev
from kws_de.grammar import Intent


def test_prompt_intent():
    assert ev.prompt_intent("Licht Küche fünfzig Prozent") == Intent("Licht", "Küche", "fünfzig")


def test_eval_recordings_with_stub_model(tmp_path):
    root = tmp_path / "approved"
    licht = config.COMMAND_LABELS.index("Licht")
    sil = config.COMMAND_LABELS.index("_silence_")
    for spk in ("spk02", "spk03"):
        d = root / "words" / "Licht"
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / f"{spk}_001.wav", np.zeros(16000, np.float32), 16000, subtype="PCM_16")
    (root / "phrases" / "spk02").mkdir(parents=True)
    sf.write(
        root / "phrases" / "spk02" / "licht-an_001.wav",
        np.zeros(32000, np.float32),
        16000,
        subtype="PCM_16",
    )
    (root / "phrases" / "index.csv").write_text(
        "file,prompt,speaker\nspk02/licht-an_001.wav,Licht an,spk02\n"
    )
    (root / "negatives" / "spk02").mkdir(parents=True)
    sf.write(
        root / "negatives" / "spk02" / "hallo_001.wav",
        np.zeros(16000, np.float32),
        16000,
        subtype="PCM_16",
    )
    (root / "negatives" / "index.csv").write_text(
        "file,prompt,speaker\nspk02/hallo_001.wav,hallo,spk02\n"
    )

    def predict_fn(window):  # always "Licht" -> isolated 100%, e2e Rejection(missing action)
        p = np.zeros(len(config.COMMAND_LABELS), np.float32)
        p[licht] = 0.9
        p[sil] = 0.1
        return p

    res = ev.eval_recordings(root, predict_fn)
    assert res["label"] == "user-customised, in-training"
    assert res["isolated"]["spk02"]["acc"] == 1.0 and res["isolated"]["spk03"]["n"] == 1
    assert res["e2e"]["spk02"]["n"] == 1 and res["e2e"]["spk02"]["acc"] == 0.0
    assert res["false_accepts"]["spk02"]["n"] == 1
    md = ev.render_recordings_section(res)
    assert "user-customised, in-training" in md and "held-out" not in md.split("user-customised")[0]
