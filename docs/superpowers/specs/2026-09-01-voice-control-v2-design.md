# Design: voice control v2 — wake word + slot commands

Date: 2026-09-01
Status: approved (brainstorming), pending spec review
Builds on: `docs/superpowers/specs/2026-08-31-kws-de-design.md` (v1 single-word KWS)

## 1. Goal & boundary

Recognise a **wake word followed by a slotted command** — `"Hey Bus" → <device> [zone] <action>`
(e.g. "Licht Küche an") — fully offline on the ESP32-S3, delivered as tested Python models plus
a grammar layer, exported INT8 with on-device budget gates. Same boundary as v1: this repo
produces the models and the recognition/grammar logic; the ESP-IDF firmware and the camper
control write are a **separate follow-on plan**.

Success criteria:

1. A tiny always-on **wake-word** model detects "Hey Bus" with a low false-accept rate
   (target ≤ 1 false accept / hour at a fixed detection rate), full-INT8, within an always-on
   compute budget.
2. A **streaming command recogniser** emits the keyword sequence for `<device> [zone] <action>`.
3. A pure **grammar / slot-filler** composes a validated intent `{device, zone?, action}` and
   rejects malformed sequences.
4. Full-intent accuracy (all slots correct) reported on real + TTS speech across an SNR sweep.
5. Both models pass INT8 + size/MACs/arena/latency budgets. Reproducible; no data/binaries
   committed; no machine-specific or third-party-app provenance in the repo.

## 2. Two-stage architecture

```text
[always-on] wake detector "Hey Bus"        ← tiny, cheap, runs continuously
      │  fires only on wake
      ▼
 command window (~3 s)  →  streaming command recogniser  →  keyword event sequence
      device {Licht, Heizung, Kühlschrank, Wasser}
      zone   {Küche, Bad, Decke, Außen}   (optional slot)
      action {an, aus}
      ▼
 grammar / slot-filler  →  intent {device, zone?, action}
      ▼
 (→ camper control write — firmware plan, out of scope here)
```

The two stages are deliberate: the cheap wake gate keeps the heavier recogniser (and, later,
the edge box) asleep until someone speaks — the low-power "wake on voice" goal. The command
model runs **only** inside the post-wake window.

## 3. Vocabulary

| Slot | Words | Data source |
|---|---|---|
| wake | `Hey Bus` | custom phrase → TTS + optional recordings (no public corpus) |
| device | `Licht, Heizung, Kühlschrank, Wasser` | reused from v1 (MSWC + TTS as in v1) |
| zone *(optional)* | `Küche, Bad, Decke, Außen` | MSWC where available, else TTS |
| action | `an, aus` | `an`/`aus` are very common → good MSWC coverage |

- A bare `<device> <action>` (no zone, e.g. "Licht an") targets the device's default/master.
- The concrete zone set is configurable; it maps to the camper's real lighting zones at firmware
  integration time (out of scope here — this spec treats zones purely as vocabulary + a grammar slot).
- Rare words (`Hey Bus`, room names) are TTS-filled with the same macOS-`say` German-voice
  approach v1 used for its thin words; per-word real-clip counts are reported honestly.

## 4. Components (all Mac + CI testable)

- `kws_de/wake.py` — wake-word model (tiny DS-CNN, classes: `wake` / `_not_`) + a streaming
  detector wrapper. Extra-small for always-on.
- command recogniser — the v1 DS-CNN (`kws_de.model`) retrained on the expanded slot vocab,
  run in streaming mode.
- `kws_de/stream.py` — ring buffer over incoming frames; runs the model every hop; **posterior
  smoothing** (moving average over a short window) + threshold + **debounce** to convert the
  continuous posterior stream into discrete keyword *events* with timestamps.
- `kws_de/grammar.py` — **pure** slot-filler: an ordered keyword-event sequence → a validated
  `Intent` (`device`, optional `zone`, `action`), or a rejection with a reason. No I/O, no model
  — fully unit-tested and portable (the reusable state-machine idea).
- `kws_de/export.py` (extended) — INT8 export for **both** models → `.tflite` + C arrays +
  metadata; budget gates for each (the wake model held to a tighter always-on budget).

Interfaces (indicative):

- `stream.KeywordStream(model, labels, smooth_win, threshold, refractory)` → `.push(frame) -> list[Event]`
- `grammar.parse(events: list[Event]) -> Intent | Rejection`
- `Intent(device: str, zone: str | None, action: str)`

## 5. Streaming recognition algorithm

1. Frames arrive (same 16 kHz / MFCC front-end as v1 — reused, keeps the host↔device golden-vector
   contract).
2. A sliding window of frames feeds the model every hop; the model outputs per-class posteriors.
3. Posteriors are smoothed over a short trailing window; a class fires an **event** when its
   smoothed posterior crosses a threshold, subject to a **refractory** period (debounce) so one
   spoken word yields exactly one event.
4. Events accumulate over the command window and are handed to the grammar in order.

The wake detector is the same machinery specialised to one class (`wake`), tuned for a **low
false-accept rate** rather than top-1 accuracy, and run continuously instead of in a window.

## 6. Grammar / slot-filler

A small, pure state machine over the event sequence:

- Expected order: `device → (zone)? → action`.
- `device` + `action` with no `zone` → intent targeting the default/master.
- Out-of-order, missing `device` or `action`, or duplicate slots → `Rejection(reason)`.
- Unknown/`_unknown_` events are ignored (open-world rejection carries over from v1).

Kept free of model and hardware concerns so it is exhaustively unit-testable and portable to the
firmware later without change.

## 7. Evaluation

- **Wake word (headline):** **false-accepts per hour** at a fixed detection rate, plus detection
  rate, across the SNR sweep (clean → van-noise). This — not top-1 accuracy — is the always-on bar.
- **Command:** per-slot accuracy AND **full-intent accuracy** (every slot correct), on real + TTS
  speech, across the SNR sweep. Report which words are real vs TTS-filled (per-word counts), same
  honesty rules as v1.
- **Grammar:** unit tests over event sequences — valid, zone-omitted, out-of-order, missing-slot,
  duplicate, unknown-interspersed.
- **Budgets:** both models full-INT8; size / MACs / arena / estimated latency within budget; the
  wake model additionally within an always-on compute budget.

## 8. Algorithmic aspects (documented, cited)

- **Streaming KWS with posterior smoothing.** Sliding-window inference + posterior smoothing +
  confidence/threshold to spot keywords in a continuous stream. → Chen, Parada, Heigold, *Small-
  footprint keyword spotting using deep neural networks*, ICASSP 2014.
- **CNN keyword spotting.** Convolutional models for small-footprint KWS. → Sainath & Parada,
  *Convolutional neural networks for small-footprint keyword spotting*, Interspeech 2015.
- **Non-streaming → streaming conversion.** Converting a trained model to a streaming detector,
  ring-buffer inference. → Rybakov et al., *Streaming Keyword Spotting on Mobile Devices*,
  Interspeech 2020 (arXiv:2005.06720).
- **Task/data framing + open-world `_unknown_`.** → Warden, *Speech Commands* (arXiv:1804.03209).
- **INT8 integer-only inference, DS-CNN, TFLite-Micro, MFCC** — unchanged from the v1 spec §12;
  the same references and host↔device golden-vector discipline apply to both models here.

Wake-word FA/hour methodology: run the always-on detector over long noise/background recordings
containing no wake phrase, count false fires, normalise to per-hour at the operating threshold
chosen for the target detection rate.

## 9. Non-goals (YAGNI)

Firmware / BLE / camper control integration (separate plan). Free-form natural language.
Variable word order beyond the fixed `device [zone] action` grammar. Whole-phrase or CTC models
(rejected during brainstorming — combinatorial data / real-time cost). Cloud anything.

## 10. Documentation & diagrams

Same convention as v1: Sphinx + `sphinx-likec4`, LikeC4 model in `docs/likec4/*.c4` — add a
two-stage pipeline view (wake → window → streaming recogniser → grammar → intent) and a dynamic
sequence view of the runtime. Prose (myst) documents the algorithmic aspects with the citations
in §8. (The docs build remains a separate workflow.)

## 11. Milestones (detailed plan follows via writing-plans)

1. Expanded-vocab dataset (device + zone + action words; MSWC + TTS-fill) reusing v1's data code.
2. Wake-word dataset ("Hey Bus" TTS + optional recordings) + tiny wake model + FA/hour eval.
3. Streaming detector (`stream.py`) — ring buffer, posterior smoothing, debounce, events.
4. Grammar / slot-filler (`grammar.py`) — pure, exhaustively unit-tested.
5. Retrain the command recogniser on the slot vocab; streaming eval (per-slot + full-intent).
6. INT8 export + budget gates for both models.
7. Eval report — wake FA/hour, full-intent accuracy, SNR sweep, budgets, honest data provenance.
8. Sphinx + likec4 docs: two-stage pipeline + dynamic views, algorithms prose.
