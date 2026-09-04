Models
=======

The two on-device models — command recogniser and wake word — as trained
and measured today, with the data they're trained on. Full experiment
log, method, and every number's provenance: ``docs/paper-notes.md``
(dated entries). Architecture diagrams and the paper: the project's
`published docs <https://ckeller42.github.io/kws-de/sphinx/>`_.

Command model
--------------

DS-CNN, 23 classes (:need:`REQ_FW_23_CLASSES`): 4 devices (``Licht``,
``Kühlschrank``, ``Heizung``, ``Aufstelldach``), 4 light zones, 13 actions
(including the four light-level words), plus ``_unknown_`` and
``_silence_``. ``kws_de.models.build_dscnn`` takes a ``width`` parameter
(default 32, channels on every conv/depthwise-separable block); trained
via ``kws-train --v2 [--qat] [--width N]``, exported via ``kws-export
--v2 --firmware [--qat] [--width N]`` (:need:`REQ_MODEL_QAT`).

Stock v2 vs. user-customised v3 vs. QAT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three points on the same architecture, in the order they were measured.
``kws-eval --recordings`` always reports real-recordings accuracy under
one of exactly two labels, never mixed — ``held-out`` or
``user-customised, in-training`` (:need:`REQ_PIPE_EVAL_LABELS`):

.. list-table::
   :header-rows: 1

   * - Model
     - Synthetic test acc.
     - Size
     - spk01 word acc.
     - spk02 word acc.
     - False accepts
   * - v2 stock (TTS-dominated, no device recordings; 2026-09-02)
     - not separately measured (catalog end-to-end: 0.689, TTS phrases)
     - 20,216 B
     - 0.19 (n=16), *held-out*
     - 0.27 (n=45), *held-out*
     - 0/6, *held-out*
   * - v3 PTQ (device recordings folded into train; 2026-09-03)
     - 88.0 %
     - 18,296 B
     - 0.538 (n=13), *user-customised*
     - 0.553 (n=38), *user-customised*
     - 0/10, *user-customised*
   * - v3 QAT (10 fine-tune epochs; 2026-09-03)
     - **91.2 %**
     - 17,880 B
     - **0.615** (n=13), *user-customised*
     - **0.737** (n=38), *user-customised*
     - 0/10, *user-customised*

The v2 row is what the recorder pipeline exists to fix: a model reporting
~0.9 on its own synthetic/MSWC split recognised roughly a quarter of what
a real microphone heard, on speakers it had never trained on. Folding
those speakers' own recordings into train (:need:`REQ_PIPE_RECORDINGS_IN_BUILD`)
and re-training turns the same two speakers' figures into
``user-customised, in-training`` numbers — a different, legitimate
question ("how well does it now know the person who trained it"), never
quoted as a held-out generalisation number. QAT
(:need:`REQ_MODEL_QAT`) recovers all of PTQ's INT8 accuracy loss against
the float model and adds 1.8 points beyond it, and moves both speakers'
real-voice accuracy up together with it — not a detection/accuracy
trade-off, an improvement on every axis measured. False-accept rate is
0/10 on both PTQ and QAT (spk01 has no recorded negatives yet). Held-out
phrase accuracy (4 clips, 1 speaker) was 0/4 for both models — too small
a sample to read anything into.

Width sweep
~~~~~~~~~~~~

``build_dscnn``'s ``width`` swept against the same ``features_v3`` recipe
as the width-32 QAT baseline above (2026-09-03):

.. list-table::
   :header-rows: 1

   * - Width
     - INT8 test acc.
     - Params
     - MACs
     - Size
     - spk01 acc. (n=13)
     - spk02 acc. (n=38)
     - False accepts (n=10)
   * - 32 (baseline)
     - **91.2 %**
     - 5,879
     - 2,070,496
     - 17,880 B
     - **0.615**
     - **0.737**
     - 0
   * - 24
     - 88.7 %
     - 3,839
     - 1,270,632
     - 14,528 B
     - 0.462
     - 0.605
     - 0
   * - 16
     - 84.7 %
     - 2,183
     - 658,928
     - 11,528 B
     - 0.385
     - 0.500
     - 0
   * - 12
     - skipped (16 already missed the recommendation bar)
     - 1,499
     - 423,636
     - —
     - —
     - —
     - —

**Conclusion: keep width 32.** Width 24 already misses the ≤1.0-point
INT8-test-accuracy bar (a 2.5-point drop), and both narrower widths lose
isolated-word accuracy on *both* real speakers versus the baseline —
narrowing the channel count trades away real-voice recognition, not just
a fraction of a synthetic-test-set point. Width 12 was skipped under that
same stopping rule; distillation from the width-32 model was not
attempted for the narrower widths (``kws_de.distill.distill()`` only
supports a fixed-width DS-CNN student against a KWT teacher, not a
same-architecture narrower student).

Export health gate
~~~~~~~~~~~~~~~~~~~~

``kws-export`` refuses to write a firmware header for a broken model
(:need:`REQ_FW_MODEL_HEALTH_GATE`): ``kws_de.export.assert_model_healthy``
requires at least 50 % held-out accuracy and at least 10 predicted
classes before ``model_data.h``/``model_config.h`` are regenerated. This
guards a real, already-recurring failure mode — a mode-collapsed export
that looked fine by exit code — and passed for every model in the tables
above, all 23 classes represented in predictions.

Command model anatomy
~~~~~~~~~~~~~~~~~~~~~~~

DS-CNN at width 32 (:need:`REQ_FW_23_CLASSES`), read straight from
``command_v3_qat.tflite`` (the QAT variant from the tables above) via
``kws_de.model_graph`` -- the same module renders the wake model below, so
neither diagram can drift from the ``.tflite`` that ships. Unlike the wake
model, this is a *non-streaming* CNN: every Invoke reprocesses the full
49x10 MFCC window from scratch, so there is no ring state to draw.

.. graphviz:: _generated/command.dot

.. note::

   Regenerate after any command-model retrain, from the repo root:

   .. code-block:: console

      $ export KWS_DATA_ROOT=/path/to/data-root
      $ uv run --no-sync kws-model-graph "$KWS_DATA_ROOT/models/command_v3_qat.tflite" \
          --out docs/sphinx/_generated/command.dot --title "Command DS-CNN (v3, QAT)"

.. list-table::
   :header-rows: 1

   * - Op
     - Input
     - Weights
     - Output
     - MACs
   * - CONV_2D 3x3, 1->32 (stem)
     - 1x49x10x1
     - 32x3x3x1
     - 1x49x10x32
     - 141,120
   * - DEPTHWISE_CONV_2D 3x3 x32 (block 1)
     - 1x49x10x32
     - 1x3x3x32
     - 1x49x10x32
     - 141,120
   * - CONV_2D 1x1, 32->32 (block 1)
     - 1x49x10x32
     - 32x1x1x32
     - 1x49x10x32
     - 501,760
   * - DEPTHWISE_CONV_2D 3x3 x32 (block 2)
     - 1x49x10x32
     - 1x3x3x32
     - 1x49x10x32
     - 141,120
   * - CONV_2D 1x1, 32->32 (block 2)
     - 1x49x10x32
     - 32x1x1x32
     - 1x49x10x32
     - 501,760
   * - DEPTHWISE_CONV_2D 3x3 x32 (block 3)
     - 1x49x10x32
     - 1x3x3x32
     - 1x49x10x32
     - 141,120
   * - CONV_2D 1x1, 32->32 (block 3)
     - 1x49x10x32
     - 32x1x1x32
     - 1x49x10x32
     - 501,760
   * - MEAN (global avg pool, head)
     - 1x49x10x32
     - --
     - 1x32
     - --
   * - FULLY_CONNECTED 32->23 (head)
     - 1x32
     - 23x32
     - 1x23
     - 736
   * - SOFTMAX (head)
     - 1x23
     - --
     - 1x23
     - --
   * - **Total**
     -
     -
     -
     - **2,070,496**

Every conv/depthwise weight is reused at all 490 output positions (49x10,
the full spectrogram), which is why 4,960 weights cost 2,070,496 MACs --
the opposite of the wake model below, where every weight fires exactly
once per Invoke. There is no receptive-field growth to reason about either:
the model sees the whole ~1 s command window on every call, not a rolling
slice of it.

Wake model ("Hey Bus")
------------------------

microWakeWord, a *streaming* TFLite-Micro model (int8, input ``[1,3,40]``,
output ``[1,1]``) trained fully locally (no Colab/GPU needed) via the
mWW upstream training config. Five measured points so far:

**Round 1 — feasibility, all-synthetic (E4).** 2,000 synthetic positives
(Piper ``de_DE-mls-medium``, "hey bus"/"hej bus"), negatives = mWW's own
~5.9 GB ambient/no-speech/speech sets, full 10,000-step config, ~6m41s on
CPU/Metal on an M4 laptop. Best checkpoint: val recall 71.65 %, precision
100 %, avg-viable-recall 0.649, ~2.9 false-accepts/hour; at cutoff 0.99,
false-reject 0.39 and **2.0 false-accepts/hour**. Proved the feasibility
claim (a custom German wake word trains on a laptop in minutes) but this
model had no real "Hey Bus" positives and no reverb augmentation.

**Round 2 — first on-device test, root cause found.** Round 1's model
never fired on a real speaker (per-2 s peak probability 0.00–0.13 saying
"Hey Bus"), although the front-end is bit-exact
(:need:`REQ_FW_WAKE_FRONTEND_PARITY`). A host probe through the identical
int8 feature path explained it: the model output ≥0.99 for *any* Piper
sentence in its training voice (not just "hey bus") and ≈0.004 for "hey
bus" in unseen Piper voices — with all positives synthetic and all
negatives real recordings, the cheapest separating feature the model
found was TTS-vs-real, not the phrase. Held-out recall on the same
synthetic distribution (round 1's 71.65 %) could not reveal this.

**Round 4 — TTS hard negatives, reverb, multi-voice ("v4", 2026-09-03).**
Retrained with TTS hard negatives (near-misses, the command vocabulary,
everyday sentences), reverb augmentation, and multi-voice positives (the
mls checkpoint's speakers plus other German Piper/macOS voices, two Piper
voices held out for the probe): 9,000 + 9,000 clips, 20k steps, 58,080 B.

.. list-table::
   :header-rows: 1

   * - Probe
     - Round 1
     - Round 4
   * - Host probe: unseen-voice "hey bus" fires
     - 1 of 4
     - 3 of 4 (one seen voice still peaks 0.99 on "licht küche an")
   * - Device, real speaker, 2 s peak trace on "Hey Bus"
     - 0.13
     - 0.83–0.99
   * - Device, silence/room noise
     - —
     - ≤0.44

The device detection gate moved from 0.99 to **0.85 × 2 consecutive
steps** (1.5 s refractory) on round 4's numbers. A synthetic clip played
through a laptop speaker fired 3 of 3. Real "Hey Bus" takes from the
guided recording session were still the next step — false-accept rate on
real speech was unmeasured at this point.

**Round 5 — real positives, user-customised by design (2026-09-03).** Ten
real "Hey Bus" takes (two sessions of a main user, pulled and QC-approved
by the remote ingest loop) were added to the round-4 recipe as their own
feature set (sampling weight 5).

.. list-table::
   :header-rows: 1

   * - Probe (device gate: 0.85 × 2 steps)
     - Round 4
     - Round 5
   * - Real "Hey Bus" takes fired
     - 4 of 10
     - **10 of 10** (peak 0.996 on every clip)
   * - One-session-only variant, fired on the *other*, unseen session
     - —
     - 5 of 5
   * - TTS non-wake worst peak
     - 0.988
     - 0.758
   * - Generic synthetic "hey bus" via laptop speaker
     - 3 of 3 (0.96–0.99)
     - 0 of 3 (0.59–0.64)

The price of round 5 is generic-voice margin: a synthetic Piper clip that
fired reliably in round 4 stops firing at all. This is the intended
trade, stated plainly: **the wake model is customised to the device's
main users**, the same "user-customised, in-training" policy the command
model follows (:need:`REQ_PIPE_EVAL_LABELS`), not tuned to work for an
arbitrary voice.

How a new main user adds their voice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Hey Bus aufnehmen** on the device menu bumps the speaker id and prompts
5 single-take reads of "Hey Bus" (:need:`REQ_FW_RECORD_WAKE_SET`) straight
to the success screen. Those takes go through the same loop as the
command recordings: ingest (:need:`REQ_PIPE_INGEST`) pulls the session,
QC (:need:`REQ_PIPE_QC_CONTENT`) verifies the wake-set transcript matches
``(hey|hej|he|hei)(bus|buss|bos|boss)``, and a retrain folds the new
speaker's takes into the wake training set as their own weighted feature
set — the same shape round 5 used for the first main user. See
:doc:`pipeline` for the full ingest -> QC -> build -> train -> export
loop and ``scripts/data-loop.sh`` for running it in one command.

Wake model anatomy
~~~~~~~~~~~~~~~~~~~~

microWakeWord's streaming graph, read straight from ``hey_bus.tflite`` (the
round-5 model measured above) via ``kws_de.model_graph`` -- one node per
compute op, one "ring N x C" node per resource-variable state
(:need:`REQ_FW_WAKE_FRONTEND_PARITY`; those rings are what let a 3-frame,
30 ms Invoke see 1.85 s of context, see below).

.. graphviz:: _generated/hey_bus.dot

.. note::

   Regenerate after any wake-model retrain (:need:`REQ_FW_RECORD_WAKE_SET`),
   from the repo root:

   .. code-block:: console

      $ export KWS_DATA_ROOT=/path/to/data-root
      $ uv run --no-sync kws-model-graph "$KWS_DATA_ROOT/models/hey_bus.tflite" \
          --out docs/sphinx/_generated/hey_bus.dot --title "Hey Bus wake model (round 5)"

.. list-table:: Layer table (compute and state ops of the 49 in the graph)
   :header-rows: 1

   * - Op
     - Input
     - Weights
     - Output
     - MACs
   * - ring 0 (read -> concat -> write newest 2) [state]
     - 2x40 + 3x40
     - --
     - 5x40
     - -- (80 B state)
   * - CONV_2D 5x1, 40->32 (stem)
     - 1x5x1x40
     - 32x5x1x40
     - 1x1x1x32
     - 6,400
   * - ring 1 (block 1) [state]
     - 4x32 + 1x32
     - --
     - 5x32
     - -- (128 B state)
   * - DEPTHWISE_CONV_2D 5x1 x32 (block 1)
     - 1x5x1x32
     - 1x5x1x32
     - 1x1x1x32
     - 160
   * - CONV_2D 1x1, 32->64 (block 1)
     - 1x1x1x32
     - 64x1x1x32
     - 1x1x1x64
     - 2,048
   * - ring 2 (block 2) [state]
     - 8x64 + 1x64
     - --
     - 9x64
     - -- (512 B state)
   * - DEPTHWISE_CONV_2D 9x1 x64 (block 2)
     - 1x9x1x64
     - 1x9x1x64
     - 1x1x1x64
     - 576
   * - CONV_2D 1x1, 64->64 (block 2)
     - 1x1x1x64
     - 64x1x1x64
     - 1x1x1x64
     - 4,096
   * - ring 3 (block 3) [state]
     - 12x64 + 1x64
     - --
     - 13x64
     - -- (768 B state)
   * - DEPTHWISE_CONV_2D 13x1 x64 (block 3)
     - 1x13x1x64
     - 1x13x1x64
     - 1x1x1x64
     - 832
   * - CONV_2D 1x1, 64->64 (block 3)
     - 1x1x1x64
     - 64x1x1x64
     - 1x1x1x64
     - 4,096
   * - ring 4 (block 4) [state]
     - 20x64 + 1x64
     - --
     - 21x64
     - -- (1,280 B state)
   * - DEPTHWISE_CONV_2D 21x1 x64 (block 4)
     - 1x21x1x64
     - 1x21x1x64
     - 1x1x1x64
     - 1,344
   * - CONV_2D 1x1, 64->64 (block 4)
     - 1x1x1x64
     - 64x1x1x64
     - 1x1x1x64
     - 4,096
   * - ring 5 (head) [state]
     - 16x64 + 1x64
     - --
     - 17x64 -> 1,088
     - -- (1,024 B state)
   * - FULLY_CONNECTED 1088->1 (head)
     - 1x1,088
     - 1x1,088
     - 1x1
     - 1,088
   * - LOGISTIC -> QUANTIZE (head)
     - 1x1
     - --
     - 1x1 uint8
     - --
   * - **Per Invoke** (plus 6 VAR_HANDLE, 6 READ/ASSIGN pairs, 6 STRIDED_SLICE, 2 RESHAPE, CALL_ONCE)
     -
     -
     -
     - **24,736** (+ 3,792 B state)

**Why every weight is one MAC.** Each Invoke produces exactly one output
row per layer: the stem reads 5 rows and emits 1, every depthwise block
reads its ring plus that 1 new row and emits 1. So a weight is used once
per step, and the 24,736 weights are the 24,736 MACs -- the opposite of the
command model above, where every weight is reused at 490 spatial positions.

**1.85 s of context from 3 new frames.** Temporal reach adds up through the
rings: the head sees 17 stem-rate rows, block 4 stretches each by 20,
block 3 by 12, block 2 by 8, block 1 by 4 -- 61 rows at one row per 30 ms
step, each row covering 5 input frames -- about ``60 x 30 ms + 50 ms ~=
1.85 s``. Enough for "Hey Bus" plus the pause before it; state resets on
mode entry.

**MACs equals weights, only here.** The wake model's output spatial size is
always 1x1 (a streaming model advances the ring by one row per Invoke
instead of recomputing a window), so MACs = weights count exactly, unlike
the command model's 49x10 = 490x multiplier above.

Generated inference
---------------------

Both shipped models run as generated C, not through the TFLite Micro
interpreter (:need:`REQ_FW_INFER_GENERATED`). ``kws-codegen`` reads the same
``.tflite`` graph the diagrams above are drawn from and emits straight-line
calls into esp-nn's ESP32-S3 kernels with the requantisation constants folded
in, so the op sequence in the generated C is exactly the op sequence in the
figures:

.. list-table::
   :header-rows: 1

   * - Model
     - Source read by ``kws-codegen``
     - Generated ops, in order
     - Arena / state

   * - Command
     - ``firmware/main/gen/model_data.h``
     - CONV_2D, then 3x (DEPTHWISE_CONV_2D, CONV_2D 1x1), MEAN,
       FULLY_CONNECTED, SOFTMAX — 10 ops
     - 31,360 B arena, 0 B state, plus 19,888 B of the shared esp-nn scratch
       region; arena in PSRAM by default (Kconfig
       ``KWS_INFER_COMMAND_ARENA``), scratch always internal

   * - Wake
     - ``firmware/main/gen/wake_model_data.h``
     - the streaming stem CONV_2D, the depthwise/1x1 stack over ring buffers,
       FULLY_CONNECTED, LOGISTIC, QUANTIZE — 14 ops after the streaming
       rewrite
     - 128 B arena, 4,200 B ring state (3,792 B of history plus one step's
       408 B of new rows), 15,552 B of the shared scratch region; all internal
       SRAM — small, and it runs every 30 ms

The scratch region is one 19,888 B array for both models, not a block of each
model's arena: esp-nn's kernels reach their scratch through file-static
globals, one per kernel family for the whole image, so separate regions would
be handed to the wrong model's kernels — and scratch is written, not just
read. The firmware serialises the two evaluations
(``firmware/main/infer_lock.h``); they only contend inside an assist window.

Both are generated from the C array the firmware embeds, not from
``models/*.tflite``: a retrain rewrites the ``.tflite`` without touching the
device headers, so only ``*_model_data.h`` is guaranteed to be the bytes the
device actually runs — and the command model's had already drifted from its
``.tflite`` when this was written. ``kws-fwgen --check`` guards that from both
sides now: it fails when the embedded array stops matching the sha8 in
``KWS_MODEL_ID``, and warns when the ``.tflite`` the stamp names has been
re-exported since. Because the headers are in the repository, both freshness
checks also run in CI, where ``models/`` does not exist at all. Regenerate
either with:

.. code-block:: console

   $ uv run --no-sync kws-codegen firmware/main/gen/model_data.h \
       --name command --out firmware/main/gen
   $ uv run --no-sync kws-codegen firmware/main/gen/wake_model_data.h \
       --name wake --out firmware/main/gen

Data provenance
-----------------

Real speech is real by construction: MSWC-German clips (CC-BY-4.0) mined
by keyword, and — since v3 — the device's own guided-recorder sessions,
QC'd and segmented through the recording pipeline
(:need:`REQ_PIPE_QC_AUDIO`, :need:`REQ_PIPE_SEGMENT`). Coverage is
structurally uneven, not a data-collection shortfall: scanning ~2.5
million MSWC-de examples found real clips for only 7 of 24 grounded
command words (``Licht``, ``Kühlschrank``, ``Heizung``, ``Wasser``,
``aus``, ``auf``, ``Außen``), with every zone word, ``an``, and every
light-level/mode word at zero — synthetic fill is therefore unavoidable,
not a shortcut (``docs/paper-notes.md`` §3, E2). Of the 21 v2 command
words, 4 are 100 % real MSWC clips, 2 are a real/TTS mix, and 15 are
100 % TTS-synthesized (macOS ``say`` plus, from v3 on, Piper neural
voices discovered from the local voice cache); every TTS clip also gets
one pitch/tempo-perturbed copy at build time. Splits are speaker-disjoint
throughout — a synthetic voice/engine combo counts as one "speaker" for
split purposes, same as a real MSWC speaker id — so a rebuild never
straddles a voice or a real speaker across train/val/test. Full
per-word/per-split counts and licensing: ``docs/DATASHEET.md``.

See :doc:`traceability` for how every requirement referenced above is
verified.
