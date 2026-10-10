# Grammar-constrained decoding at window close (review S1)

Status: proposal, 2026-10-10. Host-only first; firmware port only if the host gate passes.

## 1. Problem

The device path is: per-step posteriors → `stream_push()` (smooth, threshold 0.5, run of 2
steps fires one word) → fired words → `intent_parse()`, with one retry (`intent_rescore()`:
substitute the runner-up at a single `_unknown_` fire). Every decision is made before the
grammar can weigh in, and the grammar then accepts or rejects a token list all-or-nothing.

Where that costs (E41–E43, E58, E68):

- Phrase exact-intent sits at 19/247 for the deployed `86b7105e` and 28–33/247 for the E68
  level-matched models, against 51/51 on isolated words. The classifier is not the bottleneck.
- In running speech the correct word's top-1 run is 1–2 steps wide (isolated: 5–7). A run of 1
  never fires; a run where the target is runner-up never fires at all. E41's argmax-only
  oracle is 0.29: no run-based decoder over top-1 can pass it. A decoder over the full
  posterior can.
- `Licht _unknown_ an` parses as `Licht → an` and the zone is lost even when `Küche` is the
  confident runner-up, because rescoring only runs when the plain parse fails (E58).
- The grammar is tiny: 49 valid intents, each a sequence of 2–3 of the 21 command words
  (4 devices, 4 zones for `Licht` only, 13 actions with per-device validity). Compounds and
  scene triggers are not model classes yet and are out of scope until the next retrain.

## 2. Proposal

Replace "fire words, then parse" with "align every valid intent against the window's
posteriors, take the best, accept on a margin". The grammar is the search space, not a filter.

### 2.1 Scoring

For the assist window keep the posterior of every step: `T ≤ 25` steps (2.5 s at the 100 ms
cadence) × 23 labels. A candidate token sequence `w1..wL` (L = 2 or 3) is the left-to-right
chain `bg w1 bg w2 … wL bg`: every step is explained either by the token it sits on or as
background (`_silence_` + `_unknown_` mass). Viterbi over that chain gives the best path and
its total log-probability; each token must take at least one step, two tokens may abut.
`O((2L+1)·T)` per candidate, 49 candidates: ≈ 8,600 multiply-adds per window close.

Because every hypothesis explains the whole window, a 3-token intent competes with its 2-token
parent on evidence rather than on length: `Licht Küche an` beats `Licht an` exactly when the
`Küche` steps are explained better as `Küche` than as background. (A first draft scored only
the aligned tokens by geometric mean; the unit test showed that always prefers the shorter
intent, so it was replaced before any data was looked at.) Each token step must clear a floor
(`INTENT_RESCORE_FLOOR` 0.25 to start) so no intent is assembled from noise. Whether to align
the raw or the 3-step-smoothed posteriors is a sweep axis: smoothing flattens the one-step
peaks E41 found to a third of their height.

`# ponytail: background is silence+unknown mass only; a step dominated by a command word
outside the hypothesis costs every hypothesis the same. Add a per-class background only if the
sweep shows invented words.`

### 2.2 Accept rule

Rank candidates by path score (per-step geometric mean of the path probability). Accept the
best when its confidence (geometric mean of the token-step posteriors along its path) is
`≥ tau` and its path-score margin over the best candidate with a different intent is `≥ delta`. Otherwise reject. Both constants live in
`config` and are swept in §3; the spec-§10 temperature calibration is skipped for rung 1 (the
ordering of candidates does not depend on it; add it only if `tau` fails to separate phrases
from negatives).

### 2.3 Wake tail

The device's first window reaches back over the "…Bus" tail, which the classifier reads as
`aus` (E44). The alignment masks `aus` and `_unknown_` at steps inside `ASSIST_WAKE_TAIL_MS`
(450 ms), the same rule `assist_gate_in_wake_tail()` applies to fires today. Nothing else
changes about audio framing.

### 2.4 Where it runs

Window close, once per assist window. Order at window close: constrained decode first; if it
rejects, the existing `intent_parse` → `intent_rescore` path runs unchanged as the fallback.
The fire-based path keeps running during the window for `window_words` / field capture; the
log line gains `intent: <text> (aligned: <score>)` so field takes record which path answered.
The sweep in §3 also measures the reverse order (fires first, align as fallback) so the
choice is evidence, not taste.

## 3. Measurement before any firmware (E70, host-only)

One script, `scripts/sweep-align.py`, in the shape of `scripts/sweep-decoder.py`: run each
model once over the 247 approved phrases and 85 negatives, cache the per-step posteriors,
replay every `(tau, delta, g, floor)` combination from the cache. Models: deployed `86b7105e`
and the three E68 `gain2` seeds, so the result is reported over seeds like the deploy rule.

Reported per model:

| figure | today (device path) | target |
|---|---|---|
| phrases exact intent, n=247 | 19 deployed / 28–33 gain2 | ≥ today + 10 clips |
| false accepts on negatives, n=85 | 0 / 0, 0, 1 | ≤ today |
| alignment oracle (expected intent is the best candidate, no tau) | — | reported; the new ceiling |
| by phrase length (2 vs 3 words), zone dropped / zone invented | — | no regression on 2-word phrases |

`tau`/`delta` are chosen on spk10's 97 phrases and reported on the other speakers' 150, so the
headline number is not the tuning set. Negatives are never tuned on; the false-accept column
is the gate. Isolated-word figures are untouched by construction (a different path).

Pass: every model clears the target rows at one shared `(tau, delta, g, floor)`. Then §4.
Fail: write the entry, keep the Python reference as the measurement tool, and the next lever
is the wider-context model (E8 transducer or two-window input), not another decoder.

**Result (E70, 2026-10-10): pass.** One setting shared by the three gain2 seeds (smooth 3,
floor 0.10, tau 0.70, delta 0): phrases 28 / 28 / 33 → 72 / 80 / 77 of 247, false accepts
0 / 0 / 1 → 0 / 0 / 1 (unchanged), report set 20 / 20 / 22 → 46 / 48 / 51 of 150. The deployed
model needs its own tau (floor 0.25, tau 0.70: 19 → 36, 0 FA); at the gain2 setting it accepts 2
negatives, so `tau` is exported with the model, not fixed in the firmware. `delta` only costs
and is not ported. Order: align first (fires first lands within 1–2 clips, and only align first
fixes the E58 case). Details in `docs/paper-notes.md` E70.

## 4. Firmware port (second PR, only after §3 passes)

- `kws_de/window_intent.py`: `align(posteriors, labels, tau, delta, g, floor) -> Intent | Rejection`
  and `decode_window()` gains the new order. This stays the single Python reference of the
  device path (E58).
- `firmware/main/recognise.cc`: keep the window's smoothed posteriors in a ring
  (`float[25][23]`, 2.3 kB of the 37.8 kB free internal RAM; `uint8` if that is too much).
- `firmware/main/intent.c`: `intent_align()` over the generated grammar tables already in
  `gen/grammar.h`; candidate enumeration is a loop over `KWS_DEVICE_ACTIONS` ×
  `KWS_ZONED_DEVICE_MASK`, no new table.
- `firmware/main/wake.cc`: window-close order per §2.4.
- Parity: `scripts/gen-intent-cases.py` emits align cases (one per branch: accept, below tau,
  below delta, token under floor, tail mask, 2- vs 3-token tie) that `test_intent.c` checks
  against the Python reference, same mechanism as the 11 rescore cases; `tests/test_window_intent.py`
  pins `tau`/`delta`/`g`/`floor` to the C header.
- Flashing and the field session are the owner's; the PR ships host-tested only.

## 5. Alternatives considered

- **Widen `intent_rescore` to every slot** (n-best over fires' runner-ups, spec §10 rung 1 as
  written): smallest diff, but it still only sees fired steps. It cannot recover a word whose
  run was 1 step or whose peak was runner-up throughout, which E41 says is the common case.
- **Beam / lattice over steps**: general, but the grammar has 49 sequences; enumeration is
  smaller code and exact.
- **Catalog DP / CTC / transducer**: rejected before (E8, E16 all-blank collapse); revisit
  only with real phrase data at scale.

## 6. Out of scope

Compounds and scene triggers (not model classes), audio-framing changes, model retraining,
temperature calibration (rung 2 if needed), any change to the isolated-word path or the
deploy rule.
