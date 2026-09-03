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
