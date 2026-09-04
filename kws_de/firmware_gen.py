"""TF-free generator for the firmware's config-derived headers.

Everything C must agree with Python on — labels, prompts, MFCC constant
tables, a golden MFCC vector — is emitted here as `static const` data so
`firmware/main/mfcc.c` never re-derives librosa math. `kws-export --firmware`
(model_data.h, model_config.h) is the model-derived half."""

import argparse
import hashlib
import pathlib
import re
import tempfile

import numpy as np
import scipy.fft
import scipy.signal
from librosa.filters import mel as mel_filter

from kws_de import config, features
from kws_de.eval import build_catalog, intent_text

TOP_DB = 80.0
AMIN = 1e-10
DETECTOR = {"smooth_win": 3, "threshold": 0.5, "min_consecutive": 2, "gap_steps": 2}
_TRANS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def slug(text: str) -> str:
    s = text.lower().translate(_TRANS)
    return "-".join(w for w in "".join(c if c.isalnum() else " " for c in s).split())


def prompt_sets() -> tuple[list, list, list, list]:
    """(display, slug) pairs for words, sentences, negatives, and the wake set — in
    canonical (unshuffled) order; the device shuffles with its on-screen seed. The
    wake set is config.WAKE_WORD repeated config.WAKE_PROMPT_REPEATS times (a
    "Hey Bus"-only recording session, not a proper prompt catalog)."""
    words = [(label, slug(label)) for label in config.COMMAND_LABELS if not label.startswith("_")]
    sentences = []
    for it in build_catalog():
        text = intent_text(it)
        sentences.append((text, slug(text)))
    negs = [(p, slug(p)) for p in config.NEGATIVE_PROMPTS]
    wake = [(config.WAKE_WORD, slug(config.WAKE_WORD))] * config.WAKE_PROMPT_REPEATS
    return words, sentences, negs, wake


def mfcc_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Periodic Hann window, Slaney mel filterbank, ortho DCT-II rows — the
    three constant matrices librosa.feature.mfcc uses with our config."""
    win = scipy.signal.get_window("hann", config.WIN_SAMPLES, fftbins=True).astype(np.float32)
    mel = mel_filter(sr=config.SAMPLE_RATE, n_fft=config.WIN_SAMPLES, n_mels=config.N_MELS)
    dct = scipy.fft.dct(np.eye(config.N_MELS), type=2, norm="ortho", axis=0)[: config.N_MFCC]
    return win, mel.astype(np.float32), dct.astype(np.float32)


def mfcc_reference(x, win, mel, dct) -> np.ndarray:
    """Pure-numpy mirror of the C pipeline (what mfcc.c must reproduce)."""
    x = features._fit_length(x)
    n = config.N_FRAMES
    frames = np.stack(
        [x[i * config.HOP_SAMPLES : i * config.HOP_SAMPLES + config.WIN_SAMPLES] for i in range(n)]
    )
    spec = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2  # (n, 241)
    logmel = 10.0 * np.log10(np.maximum(AMIN, spec @ mel.T))  # (n, 40)
    logmel = np.maximum(logmel, logmel.max() - TOP_DB)
    return (logmel @ dct.T).astype(np.float32)  # (n, 10)


# 6 significant figures = float32's honest precision. Printing more (e.g. .8e)
# exposes libm's last-ULP differences between platforms (Apple vs glibc cos/log),
# which made the CI gen-fresh diff fail against locally committed headers even at
# pinned library versions. The C side reads these as `float`, so the dropped
# digits carry no information; host MFCC parity keeps its ~55x tolerance margin.
def _c_float_rows(name, arr) -> str:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        flat = ", ".join(f"{v:.5e}f" for v in arr)
        return f"static const float {name}[{arr.size}] = {{{flat}}};\n"
    rows = ",\n".join("  {" + ", ".join(f"{v:.5e}f" for v in row) + "}" for row in arr)
    dims = "".join(f"[{d}]" for d in arr.shape)
    return f"static const float {name}{dims} = {{\n{rows}\n}};\n"


def mel_bands(mel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compress the mel filterbank to its non-zero band per filter.

    A triangular mel filter touches 4-33 of the 241 FFT bins, so the dense
    40x241 matrix is 95% exact zeros — 38.5 KB of flash rodata read in full for
    every frame. Returned as (start, length, weights) with the weights
    concatenated in filter order.

    Dropping only *exact* zeros keeps the dot product bit-identical: `acc += 0 *
    power[k]` leaves a finite accumulator unchanged in IEEE-754, and the
    surviving terms are summed in the same ascending-bin order.
    """
    nz = mel != 0.0
    start = nz.argmax(axis=1).astype(np.int32)
    end = mel.shape[1] - nz[:, ::-1].argmax(axis=1)
    length = (end - start).astype(np.int32)
    interior = [m for m in range(mel.shape[0]) if not nz[m, start[m] : end[m]].all()]
    if interior:
        raise ValueError(f"mel filters {interior} have interior zeros — banding would drop terms")
    weights = np.concatenate([mel[m, start[m] : end[m]] for m in range(mel.shape[0])])
    return start, length, weights.astype(np.float32)


def _c_strings(name, items) -> str:
    body = ",\n".join(f'  "{s}"' for s in items)
    return f"static const char *const {name}[{len(items)}] = {{\n{body}\n}};\n"


def generate(out) -> None:
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    hdr = "/* generated by kws-fwgen — do not edit */\n#pragma once\n"

    labels = config.COMMAND_LABELS
    (out / "labels.h").write_text(
        hdr
        + f"#define KWS_NUM_LABELS {len(labels)}\n"
        + f"#define KWS_SILENCE_INDEX {labels.index('_silence_')}\n"
        + f"#define KWS_UNKNOWN_INDEX {labels.index('_unknown_')}\n"
        + _c_strings("KWS_LABELS", labels)
    )

    words, sentences, negs, wake = prompt_sets()
    p = hdr
    for tag, items in (("WORD", words), ("SENTENCE", sentences), ("NEG", negs), ("WAKE", wake)):
        p += f"#define KWS_NUM_{tag}_PROMPTS {len(items)}\n"
        p += _c_strings(f"KWS_{tag}_PROMPTS", [d for d, _ in items])
        p += _c_strings(f"KWS_{tag}_SLUGS", [s for _, s in items])
    (out / "prompts.h").write_text(p)

    win, mel, dct = mfcc_tables()
    fc = hdr + "#include <stdint.h>\n"
    fc += f"#define KWS_SAMPLE_RATE {config.SAMPLE_RATE}\n#define KWS_WIN {config.WIN_SAMPLES}\n"
    fc += f"#define KWS_HOP {config.HOP_SAMPLES}\n#define KWS_N_MELS {config.N_MELS}\n"
    fc += f"#define KWS_N_MFCC {config.N_MFCC}\n#define KWS_N_FRAMES {config.N_FRAMES}\n"
    fc += f"#define KWS_N_BINS {config.WIN_SAMPLES // 2 + 1}\n"
    fc += f"#define KWS_TOP_DB {TOP_DB}f\n#define KWS_AMIN {AMIN}f\n"
    for k, v in DETECTOR.items():
        fc += f"#define KWS_{k.upper()} {v}{'f' if isinstance(v, float) else ''}\n"
    mel_start, mel_len, mel_w = mel_bands(mel)
    fc += f"#define KWS_MEL_NNZ {mel_w.size}\n"
    fc += (
        _c_float_rows("KWS_WINDOW", win)
        + _c_int_rows("KWS_MEL_START", mel_start, "uint16_t")
        + _c_int_rows("KWS_MEL_LEN", mel_len, "uint16_t")
        + _c_float_rows("KWS_MEL_W", mel_w)
        + _c_float_rows("KWS_DCT", dct)
    )
    (out / "features_config.h").write_text(fc)

    rng = np.random.default_rng(0)
    t = np.arange(config.CLIP_SAMPLES) / config.SAMPLE_RATE
    sig = (
        0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 1300 * t)
        + 0.05 * rng.standard_normal(t.size)
    )
    # a quarter second of digital silence exercises the top_db clamp
    sig[: config.SAMPLE_RATE // 4] = 0.0
    pcm = np.clip(np.round(sig * 32767), -32768, 32767).astype(np.int16)
    ref = mfcc_reference(pcm.astype(np.float32) / 32768.0, win, mel, dct)
    tv = hdr + "#include <stdint.h>\n"
    tv += f"static const int16_t TV_PCM[{pcm.size}] = {{{', '.join(map(str, pcm.tolist()))}}};\n"
    tv += _c_float_rows("TV_MFCC", ref)
    (out / "test_vectors.h").write_text(tv)

    # Wake front-end golden vector. pymicro_features is an optional extra
    # (`--extra wake`); without it we leave the committed header alone rather
    # than emitting a fabricated one.
    try:
        wt_pcm, wt_feat = wake_test_vector()
    except ImportError:
        print("WARNING: pymicro-features absent — skipping wake_test_vectors.h")
        return
    wt = hdr + "#include <stdint.h>\n"
    wt += f"#define WT_ROWS {wt_feat.shape[0]}\n"
    wt_body = ", ".join(map(str, wt_pcm.tolist()))
    wt += f"static const int16_t WT_PCM[{wt_pcm.size}] = {{{wt_body}}};\n"
    wt += _c_int_rows("WT_FEATURES", wt_feat, "int8_t")
    (out / "wake_test_vectors.h").write_text(wt)


# --- wake front-end golden vector -------------------------------------------
# `firmware/main/wakefront.c` wraps the same vendored TFLite-Micro microfrontend
# that `pymicro_features` compiles, with the same FrontendConfig, so the two must
# agree bit-for-bit. These constants only describe how the frontend's uint16
# output becomes the int8 model input; the derivation is documented in
# `firmware/main/wakefront.h`.
WAKE_STRIDE = 160  # 10 ms at 16 kHz — one feature row per stride
WAKE_FEATURES = 40
# pymicro_features scales the raw uint16 by this before handing it to Python
# (src/micro_features.cpp: FLOAT32_SCALE). Exactly representable, so the raw
# integer is recoverable without loss.
WAKE_FLOAT32_SCALE = 0.0390625
# ESPHome micro_wake_word.cpp::generate_features_(): int8 = (v*256 + 333)/666 - 128.
WAKE_VALUE_SCALE = 256
WAKE_VALUE_DIV = 666


def wake_int8(raw: int) -> int:
    """microWakeWord's uint16-frontend-value -> int8-model-input requantisation."""
    v = (raw * WAKE_VALUE_SCALE + WAKE_VALUE_DIV // 2) // WAKE_VALUE_DIV - 128
    return max(-128, min(127, v))


def wake_features(pcm: np.ndarray) -> np.ndarray:
    """Reference int8 feature rows for `pcm`, straight from pymicro_features."""
    from pymicro_features import MicroFrontend

    fe = MicroFrontend()
    rows = []
    for i in range(0, len(pcm) - WAKE_STRIDE + 1, WAKE_STRIDE):
        out = fe.process_samples(pcm[i : i + WAKE_STRIDE].tobytes())
        if not out.features:
            continue
        rows.append([wake_int8(round(f / WAKE_FLOAT32_SCALE)) for f in out.features])
    return np.array(rows, dtype=np.int8)


def wake_test_vector() -> tuple[np.ndarray, np.ndarray]:
    """1 s of synthetic tone+noise (same recipe as the MFCC vector) and the int8
    feature rows the C front-end must reproduce exactly."""
    rng = np.random.default_rng(1)
    t = np.arange(config.SAMPLE_RATE) / config.SAMPLE_RATE
    sig = (
        0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 1300 * t)
        + 0.05 * rng.standard_normal(t.size)
    )
    sig[: config.SAMPLE_RATE // 4] = 0.0  # silence lead-in exercises the noise estimator
    pcm = np.clip(np.round(sig * 32767), -32768, 32767).astype(np.int16)
    return pcm, wake_features(pcm)


def _c_int_rows(name, arr, ctype) -> str:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        flat = ", ".join(str(int(v)) for v in arr)
        return f"static const {ctype} {name}[{arr.size}] = {{{flat}}};\n"
    rows = ",\n".join("  {" + ", ".join(str(int(v)) for v in row) + "}" for row in arr)
    dims = "".join(f"[{d}]" for d in arr.shape)
    return f"static const {ctype} {name}{dims} = {{\n{rows}\n}};\n"


_FLOAT_RE = re.compile(r"-?\d+\.\d+e[+-]\d+f")


# Compare a committed generated header against a freshly generated one. The C
# float tables (mel/DCT/window, TV_MFCC) are computed through numpy/scipy whose
# transcendental and matmul kernels are CPU-SIMD- and BLAS-order-dependent, so
# they are NOT byte-reproducible across machines (CI runner vs a dev laptop)
# even at pinned library versions. Byte-exact comparison is therefore the wrong
# gate: structure (every non-float character) must match exactly, values only
# within a tolerance that clears last-digit hardware noise but catches any real
# config/generator change (those move values by far more). The tolerance is set
# from the measured cross-platform delta: TV_MFCC's near-zero coefficients are
# float-cancellation noise (the reference matmul even overflows), so a dev
# laptop and the CI runner disagree by up to ~1e-4 absolute (and ~100% relative
# on the ~1e-5 coefficients). atol=1e-3 clears that with margin; it is still far
# below int8 resolution and any genuine change, which shifts values by O(1) or
# alters the structure the mask above already pins.
def _headers_match(committed: str, fresh: str) -> bool:
    if _FLOAT_RE.sub("<f>", committed) != _FLOAT_RE.sub("<f>", fresh):
        return False
    a = np.array([float(s[:-1]) for s in _FLOAT_RE.findall(committed)])
    b = np.array([float(s[:-1]) for s in _FLOAT_RE.findall(fresh)])
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-2, atol=1e-3))


def check(committed_dir) -> list[str]:
    """Return the names of committed headers that differ (structurally, or
    numerically beyond tolerance) from a fresh generation. Empty list = OK."""
    committed_dir = pathlib.Path(committed_dir)
    with tempfile.TemporaryDirectory() as tmp:
        generate(tmp)
        bad = []
        # wake_test_vectors.h is int-only and produced by deterministic C, so it
        # compares byte-exact through _headers_match's (float-free) string path.
        # It is skipped entirely when pymicro-features is not installed.
        names = ["labels.h", "prompts.h", "features_config.h", "test_vectors.h"]
        if (pathlib.Path(tmp) / "wake_test_vectors.h").exists():
            names.append("wake_test_vectors.h")
        for f in names:
            if not _headers_match(
                (committed_dir / f).read_text(), (pathlib.Path(tmp) / f).read_text()
            ):
                bad.append(f)
        return bad + _model_stamp_drift(committed_dir)


def _model_stamp_drift(committed_dir) -> list[str]:
    """Does the embedded model still match the id stamped beside it?

    model_data.h (the bytes the device runs) and model_config.h (the
    quantisation constants, plus KWS_MODEL_ID whose sha8 is over exactly those
    bytes) come out of one `kws-export --firmware` run but are two files. A
    hand-edit or a half-finished regeneration desynchronises them silently, and
    everything downstream -- including the generated inference, which is built
    from model_data.h -- would then be self-consistently wrong. Needs no data
    root, so it runs in CI.

    The second half only warns: when the .tflite the stamp names is present and
    no longer hashes to the stamp, a retrain has rewritten models/<name>.tflite
    without regenerating the firmware headers. That is a developer-machine
    condition (and exactly the trap the command codegen walked into), not a
    broken commit -- CI has no models/ at all.
    """
    from kws_de import codegen

    committed_dir = pathlib.Path(committed_dir)
    # Both files are kws-export --firmware's output, not this module's, so a
    # directory this module just generated into legitimately has neither.
    if not all((committed_dir / f).exists() for f in ("model_config.h", "model_data.h")):
        return []
    config_h = (committed_dir / "model_config.h").read_text()
    stamp = re.search(r'KWS_MODEL_ID\s+"([^@"]+)@([0-9a-f]+)', config_h)
    if not stamp:
        return ["model_config.h (no KWS_MODEL_ID stamp to check model_data.h against)"]
    name, sha8 = stamp.group(1), stamp.group(2)
    embedded = hashlib.sha256(codegen.model_bytes(committed_dir / "model_data.h")).hexdigest()[:8]
    if embedded != sha8:
        return [f"model_data.h (embedded model is {embedded}, KWS_MODEL_ID says {sha8})"]
    source = config.MODELS_DIR / name
    if source.exists() and hashlib.sha256(source.read_bytes()).hexdigest()[:8] != sha8:
        print(
            f"WARNING: {name} has been re-exported since the firmware headers were written "
            f"(the device still runs {sha8}) -- run kws-export --firmware, then regenerate "
            "the inference with kws-codegen"
        )
    return []


def main() -> None:  # pragma: no cover - I/O wrapper
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="firmware/main/gen")
    ap.add_argument(
        "--check",
        metavar="DIR",
        help="verify committed headers in DIR are current (structure exact, "
        "floats within tolerance) instead of writing; exit 1 on mismatch",
    )
    args = ap.parse_args()
    if args.check:
        bad = check(args.check)
        if bad:
            raise SystemExit("stale generated headers: " + ", ".join(bad))
        return
    generate(args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
