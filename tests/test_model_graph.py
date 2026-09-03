import shutil
import subprocess

import numpy as np
import tensorflow as tf

from kws_de.model_graph import analyze, to_dot


def _tiny_tflite() -> bytes:
    """A throwaway 2-layer int8 model -- not a real kws-de architecture, just
    enough CONV_2D + DEPTHWISE_CONV_2D structure to exercise the graph reader
    without depending on a trained artifact."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((8, 8, 1)),
            tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu"),
            tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu"),
        ]
    )
    rng = np.random.default_rng(0)
    rep = rng.standard_normal((4, 8, 8, 1)).astype(np.float32)

    def rep_gen():
        for i in range(rep.shape[0]):
            yield [rep[i : i + 1]]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def test_analyze_reads_compute_ops_and_macs():
    g = analyze(_tiny_tflite())
    op_names = {o["op_name"] for o in g["compute_ops"]}
    assert "CONV_2D" in op_names
    assert "DEPTHWISE_CONV_2D" in op_names
    assert g["rings"] == []  # a plain conv model has no streaming ring state
    for info in g["infos"].values():
        assert info["macs"] > 0
        assert info["weights"] > 0
    # conv: 4 filters x 3x3 x 1 in-channel = 36 weights, over an 8x8 output -> 2,304 MACs
    conv_info = g["infos"][int(g["compute_ops"][0]["index"])]
    assert conv_info["weights"] == 36
    assert conv_info["macs"] == 36 * 8 * 8


def test_dot_output_is_well_formed_graphviz():
    dot = to_dot(_tiny_tflite(), title="tiny test model")
    assert dot.startswith("digraph model {")
    assert dot.rstrip().endswith("}")
    assert dot.count("{") == dot.count("}")
    assert "->" in dot
    assert "tiny test model" in dot

    dot_bin = shutil.which("dot")
    if dot_bin is None:
        return  # structural check above is all we can do without graphviz installed
    result = subprocess.run([dot_bin, "-Tsvg"], input=dot.encode(), capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr.decode()
