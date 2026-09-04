"""Byte-for-byte parity between the generated C and the TFLite interpreter.

These tests compile the generated C against esp-nn's ANSI-C kernels and run
the interpreter on the same inputs. Zero LSB difference is the requirement.
They need the models (KWS_DATA_ROOT) and a working `cc`, so they skip cleanly
where either is missing.
"""

import pathlib
import shutil
import subprocess

import numpy as np
import pytest

from kws_de import codegen, config, tflite_graph

tf = pytest.importorskip("tensorflow")

REPO = pathlib.Path(__file__).resolve().parents[1]
TEST_DIR = REPO / "firmware" / "test"
GEN_DIR = REPO / "firmware" / "main" / "gen"
WAKE = config.MODELS_DIR / "hey_bus.tflite"
# The command model the FIRMWARE embeds, as the C array in the repo -- not
# models/command*.tflite, which a training run rewrites without touching the
# device headers, and which is therefore a different model. Parity only means
# something against the bytes gen/command_infer.c was generated from.
COMMAND = GEN_DIR / "model_data.h"

# The layers the spec names: the command model's first 3x3 conv over a single
# input channel, its 3x3 depthwise, its 1x1 conv and its (per-channel) FC; the
# wake model's 5x1 stem conv, a 21x1 depthwise, a 1x1 conv and the 1088->1 FC.
# op 7 (MEAN) and op 9 (SOFTMAX) are the command model's own requantisation --
# S-2: before this, they were covered only by the insensitive whole-model
# check (see test_whole_command_model_matches_the_interpreter).
LAYERS = [
    ("command", COMMAND, 0, "CONV_2D 3x3x1"),
    ("command", COMMAND, 1, "DEPTHWISE_CONV_2D 3x3"),
    ("command", COMMAND, 2, "CONV_2D 1x1"),
    ("command", COMMAND, 7, "MEAN"),
    ("command", COMMAND, 8, "FULLY_CONNECTED 32->23 per-channel"),
    ("command", COMMAND, 9, "SOFTMAX"),
    ("wake", WAKE, 14, "CONV_2D 5x1 stem"),
    ("wake", WAKE, 32, "DEPTHWISE_CONV_2D 21x1"),
    ("wake", WAKE, 18, "CONV_2D 1x1"),
    ("wake", WAKE, 36, "FULLY_CONNECTED 1088->1"),
]

needs_cc = pytest.mark.skipif(shutil.which("cc") is None, reason="no host C compiler")


def _make(target: str, **make_vars: str) -> None:
    cmd = ["make", "-C", str(TEST_DIR), target] + [f"{k}={v}" for k, v in make_vars.items()]
    subprocess.run(cmd, check=True)


def _interpreter_output(blob: bytes, op, inputs):
    """The op's exact answer from the reference kernels, via a one-op probe
    model -- the same route the LOGISTIC table will take in a later task.

    BUILTIN_REF is not a detail: the desktop interpreter's default resolver
    hands int8 convolutions to the XNNPACK delegate, which requantises with a
    single rounding step. TFLM's reference kernels and esp-nn both use
    gemmlowp's double rounding (SaturatingRoundingDoublingHighMul followed by
    RoundingDivideByPOT), so the delegate disagrees with the device by 1 LSB on
    roughly 0.2% of outputs. The reference kernels are what the device runs.
    """
    itp = tf.lite.Interpreter(
        model_content=tflite_graph.probe_model(blob, op),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    itp.allocate_tensors()
    detail_in, detail_out = itp.get_input_details()[0], itp.get_output_details()[0]
    itp.set_tensor(detail_in["index"], inputs)
    itp.invoke()
    return itp.get_tensor(detail_out["index"]).astype(np.int8).ravel()


@needs_cc
@pytest.mark.parametrize(
    ("model", "path", "op_index", "what"),
    LAYERS,
    ids=[f"{m}-op{i}" for m, _, i, _ in LAYERS],
)
def test_layer_is_byte_identical(model, path, op_index, what):
    """Generate one real layer, compile it against esp-nn's ANSI kernels and
    demand every output byte match the interpreter's."""
    if not path.exists():
        pytest.skip(f"{path} absent (KWS_DATA_ROOT)")
    blob = codegen.model_bytes(path)
    graph = tflite_graph.read_graph(blob)
    op = graph.ops[op_index]
    assert op.name in codegen.EMITTERS, f"op {op_index} is {op.name}, not {what}"
    in_t = graph.tensors[op.inputs[0]]
    rng = np.random.default_rng(op_index)
    inputs = rng.integers(-128, 128, size=in_t.shape, dtype=np.int8)

    expect = _interpreter_output(blob, op, inputs)
    codegen.write_probe_vectors(blob, op, inputs, expect, GEN_DIR)
    _make("test_infer_parity")
    result = subprocess.run(
        [str(TEST_DIR / "test_infer_parity")], capture_output=True, text=True, check=True
    )
    assert f"conv parity: 0/{expect.size} bytes differ" in result.stdout, result.stdout
    assert "test_infer_parity OK" in result.stdout


WAKE_TAKES = config.DATA_DIR / "recordings" / "approved" / "wake"
needs_wake = pytest.mark.skipif(not WAKE.exists(), reason=f"{WAKE} absent (KWS_DATA_ROOT)")
# COMMAND itself is always present (it is in the repo); what these tests still
# need is the feature split they draw their real clips from, which lives under
# DATA_DIR -- skip on what is actually used, so a data root with models but no
# splits skips instead of erroring.
needs_command = pytest.mark.skipif(
    not (config.DATA_DIR / "features_v3_test.npz").exists(),
    reason="KWS_DATA_ROOT feature splits absent",
)


def _interpreter(blob: bytes):
    """The reference resolver, for the reason spelled out in _interpreter_output."""
    itp = tf.lite.Interpreter(
        model_content=blob,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    itp.allocate_tensors()
    return itp, itp.get_input_details()[0], itp.get_output_details()[0]


def _wake_clip(blob: bytes, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(inputs, outputs) for every 3-row step of one clip, from a freshly reset
    interpreter -- the states carry across steps and must not carry across
    clips, exactly as on the device."""
    itp, detail_in, detail_out = _interpreter(blob)
    itp.reset_all_variables()
    inputs, probs = [], []
    for start in range(0, len(rows) - 2, 3):
        window = rows[start : start + 3]
        itp.set_tensor(detail_in["index"], window[None, ...].astype(np.int8))
        itp.invoke()
        inputs.append(window.ravel())
        probs.append(itp.get_tensor(detail_out["index"]).ravel())
    return np.array(inputs, np.int8), np.array(probs, np.uint8)


@needs_wake
@needs_cc
@pytest.mark.skipif(not WAKE_TAKES.exists(), reason="approved/wake absent")
def test_whole_wake_model_matches_the_interpreter_on_every_step(tmp_path):
    """The golden feature vector plus the ten approved wake takes, run through
    the generated C as a streaming sequence and compared step by step -- a
    state bug that the last step happens to agree on still fails here.

    Generates into `tmp_path`, not the committed `firmware/main/gen` (S-3):
    this test's real-clip vectors are recordings-derived and never committed,
    so regenerating them must not touch the tracked `wake_infer.{c,h}` --
    `GEN_DIR=<tmp_path>` on the `make` command line points the build there
    instead, leaving the working tree exactly as `kws-codegen` last wrote it.
    """
    import soundfile as sf

    from kws_de import firmware_gen

    pytest.importorskip("pymicro_features")
    blob = WAKE.read_bytes()
    _, golden = firmware_gen.wake_test_vector()
    takes = sorted(WAKE_TAKES.rglob("*.wav"))
    assert len(takes) == 10, f"expected 10 approved wake takes, found {len(takes)}"

    rows = [golden]
    for path in takes:
        pcm, rate = sf.read(path, dtype="int16")
        assert rate == config.SAMPLE_RATE
        rows.append(firmware_gen.wake_features(pcm))
    clips = [_wake_clip(blob, r) for r in rows]

    files = codegen.generate(blob, "wake")
    for filename, text in files.items():
        (tmp_path / filename).write_text(text)
    codegen.write_infer_vectors("wake", clips, tmp_path)
    _make("test_wake_parity", GEN_DIR=str(tmp_path))
    result = subprocess.run(
        [str(TEST_DIR / "test_wake_parity")], capture_output=True, text=True, check=True
    )
    steps = sum(len(i) for i, _ in clips)
    assert f"wake parity: 0/{steps} steps differ" in result.stdout, result.stdout


REAL_CLIPS = 64  # S-1: >= 64 real feature windows, kept to a few seconds' compile


@needs_command
@needs_cc
def test_whole_command_model_matches_the_interpreter(tmp_path):
    """The command model is the only one with MEAN and SOFTMAX, both of which
    carry their own requantisation, so it gets its own whole-model check.

    S-1: 4 synthetic random-int8 vectors alone are nearly insensitive here --
    softmax's int8 output saturates, so a 1-LSB error in MEAN's multiplier or
    a wrong fold shift can leave all 92 compared bytes unchanged (see the
    review). Real MFCC feature windows exercise the actual input distribution
    the classifier runs on, so most of the clips below are drawn from the
    test split (kws_de.dataset.load_split, prefix "features_v3" -- the real-
    speech rebuild, matching the command model's [1, 49, 10, 1] input); the
    cheap synthetic vectors stay too as an edge-of-range check.

    Generates into `tmp_path`, not `firmware/main/gen` (S-3): same reasoning
    as the wake test above. command_infer.{c,h} are committed now, so writing
    into the tracked directory would not merely be `git status` noise -- it
    would overwrite the shipped artefacts with a build fixture.
    """
    from kws_de import dataset

    blob = codegen.model_bytes(COMMAND)
    itp, detail_in, detail_out = _interpreter(blob)
    in_scale, in_zp = detail_in["quantization"]
    rng = np.random.default_rng(7)

    inputs, expect = [], []

    def _run(sample: np.ndarray) -> None:
        itp.set_tensor(detail_in["index"], sample.astype(np.int8))
        itp.invoke()
        inputs.append(sample.ravel())
        expect.append(itp.get_tensor(detail_out["index"]).ravel().astype(np.int8))

    for _ in range(4):
        _run(rng.integers(-128, 128, size=detail_in["shape"], dtype=np.int8))

    features, _, _ = dataset.load_split("test", "features_v3")
    assert features.shape[1:] == tuple(int(d) for d in detail_in["shape"][1:3]), features.shape
    real = rng.choice(len(features), size=REAL_CLIPS, replace=False)
    for i in real:
        # Same quantisation the eval harness uses for this model
        # (kws_de.eval._tflite_predict): round(feature / scale + zero_point).
        q = np.round(features[i] / in_scale + in_zp).astype(np.int8)
        _run(q[None, ..., None])

    files = codegen.generate(blob, "command")
    for filename, text in files.items():
        (tmp_path / filename).write_text(text)
    # One clip per vector: the model is stateless, so grouping is only for
    # WAKE_CLIPS/COMMAND_CLIPS to report how many real clips ran.
    clips = [(np.array([i]), np.array([e])) for i, e in zip(inputs, expect, strict=True)]
    codegen.write_infer_vectors("command", clips, tmp_path)
    _make("test_command_parity", GEN_DIR=str(tmp_path))
    result = subprocess.run(
        [str(TEST_DIR / "test_command_parity")], capture_output=True, text=True, check=True
    )
    total_bytes = len(inputs) * len(expect[0])
    assert f"command parity: 0/{total_bytes}" in result.stdout, result.stdout
    assert f"{len(inputs)} clips" in result.stdout, result.stdout
