# Field Capture in Assistent Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In Assistent mode, store the audio of each real interaction (wake phrase + 2.5 s command window) together with what the device itself recognised, so real usage becomes training data and yields a field-accuracy figure.

**Architecture:** The wake task already knows the ring position of a wake fire; the gate already knows when the window closes. On the closing edge the wake task hands the recorder a *span* (`start`, `len`) plus the device's own prediction, and the record task — which is idle in Assistent mode — copies that span out of the always-on audio ring and writes one WAV plus one `field.csv` row. Nothing is written while the recogniser runs. On the workstation the existing ingest → QC path carries the take: Whisper transcribes it, the wake phrase is cut off as a `wake` clip, the rest is normalised and run through the *same* `kws_de.grammar.parse` the device uses, and the resulting intent becomes the label (or, when it does not parse, negative/`_unknown_` material). The device's own intent rides along in `qc.csv` as `device_intent`/`agrees` and is scored, never used as a label.

**Tech Stack:** ESP-IDF v5.5.5 / FreeRTOS / LVGL 9 (C and C++ firmware), plain `cc` host tests (`firmware/test/Makefile`), bash (`scripts/pull-recordings.sh`, `scripts/ingest.sh`), Python 3 + numpy + soundfile + mlx-whisper (`kws_de/qc.py`, `kws_de/eval.py`), pytest, Sphinx + sphinx-needs (`docs/sphinx/`).

**Spec:** `docs/superpowers/specs/2026-09-03-field-capture-design.md`

## Global Constraints

Copied from the spec's decisions and non-goals. Every task's requirements implicitly include this section.

- Capture is **opt-in**: a visible toggle, off at boot until the user turns it on once, the state persisted in NVS under the existing `kws` namespace and restored at boot.
- **No file I/O during the window.** The copy and the FAT write happen after the window closes, on the record task — never while the recogniser is active. A FAT write costs 100–300 ms on flash (measured in wave 2).
- **Everything is kept.** Non-parsable transcripts become negative / unknown material; they are never dropped. Only an empty transcript or an audio-gate failure rejects a take.
- Labels are **auto-derived**: Whisper transcript + `kws_de.grammar.parse`. No hand labelling.
- The **device prediction is never a label**. It is stored next to the Whisper label and scored against it (`agrees`), nothing more.
- Field takes are **in-training** like guided takes (they enter `approved/` through the same paths, so `kws_de.data.merge_recordings` picks them up unchanged).
- A take is one interaction: **wake phrase + window** — `start = fire_pos − 1.0 s`, `len = 3.5 s`.
- Layout: `storage_root()/field/spkNN/<boot-ms>.wav` (16 kHz mono int16, like every take; speaker id = the current NVS id, no bump per interaction) plus `storage_root()/field/spkNN/field.csv` with columns `file,fire_ms,wake_prob,device_intent,device_words,window_ms,ms,peak_dbfs`.
- **Storage floor:** below the same floor the recorder uses (`STORAGE_MIN_FREE_BYTES`, 200 KB) the take is dropped with one log line `field: dropped, storage low` and a counter in `status`.
- **Gates unchanged:** the QC audio gate applies as today; the content gate for `set == "field"` approves anything with a non-empty transcript.
- **No host names or paths in committed files.** The remote Mac's name, the `KWS_DATA_ROOT` absolute path and any username never appear in the repo (`kws_de.eval._relative_to_data_root` is the existing guard). This plan names the flashing helper scripts; committed docs must not.
- **Screens:** nothing else changes for the user — green flash + beep + recognised intent as today. The "REC" badge is the only visible difference.
- **Non-goals:** capturing missed wakes, any cloud component, model changes.

---

## Context: this branch, and the two branches moving underneath it

Read before Task 1. The spec was written against interfaces that partly do not exist yet on `spec/field-capture`:

- **`storage_root()` does not exist here.** `firmware/main/storage.h` has `storage_mount/unmount/free_bytes/wl_handle` and the literal `/rec` is hard-coded in `record.c`, `wake.cc` and `recognise.cc`. `feat/sdcard-recording` is to introduce a card root, `storage_root()` and USB exposure, but at the time of writing that branch has **no commits of its own** (`git diff main...feat/sdcard-recording` is empty). Task 2 therefore adds the two-line `storage_root()` accessor itself, returning `"/rec"` — exactly the shape the microSD branch generalises. **Rebase check at execution time:** if `storage_root()` is already declared in `firmware/main/storage.h`, skip that step and use theirs.
- **`wake.cc` and `recognise.cc` are being changed by `feat/generated-inference`** (an inference path behind `CONFIG_KWS_INFER_GENERATED`, plus `firmware/test/Makefile` and three new parity tests). Task 1 adds a Makefile target and Task 2 edits both `.cc` files. **Rebase check at execution time:** `firmware/test/Makefile`, `firmware/main/wake.cc`, `firmware/main/recognise.cc`. The edits in this plan are additive and touch neither the Invoke path nor the arena, so a rebase should be mechanical.
- **The audio ring is already big enough and already in PSRAM.** `AUDIO_RING_SAMPLES` is `KWS_SAMPLE_RATE * 10` (10 s) and `audio_start()` allocates it with `MALLOC_CAP_SPIRAM`. The spec's "raise it if needed" is therefore a no-op, and the spec's PSRAM-vs-SRAM risk does not apply. The `static_assert` is still added, so a later shrink of the ring fails the build instead of silently truncating takes.
- **There is no grammar on the device.** `firmware/main/` has `stream.c` (the debounce/fire logic) but nothing that composes an intent; `grep -rn "grammar\|intent" firmware/main/` returns nothing. So `device_intent` is written as the **ordered fired command words, space-joined** (`"Licht Küche an"`) and `device_words` as `word:conf` entries joined by `|`. The agreement in QC runs that word list through the same `kws_de.grammar.parse` the label uses, comparing `Intent` to `Intent` — the same answer a C grammar port would give, without the port. This is the one place the plan deviates from the spec's wording; it is called out again in the self-review.

---

## File Structure

| Path | Create / Modify | Responsibility |
|---|---|---|
| `firmware/main/field.h` | Create | Field-capture constants, `field_state_t`, `field_take_t`, the ring `static_assert`. |
| `firmware/main/field.c` | Create | Pure C: toggle state, fire position, the copy-span arithmetic. No FreeRTOS, no clock, no globals. |
| `firmware/test/test_field.c` | Create | Host test for the span arithmetic and the toggle. |
| `firmware/test/Makefile` | Modify | `test_field` target + `TESTS` entry. |
| `firmware/main/storage.h` / `.c` | Modify | `storage_root()` (skip if the microSD branch already landed it). |
| `firmware/main/recognise.h` / `.cc` | Modify | `window_intent` / `window_words` in `recognise_status_t`: the device's prediction for one window. |
| `firmware/main/wake.h` / `.cc` | Modify | Hold `field_state_t`, persist the toggle in NVS, post the take on the window's closing edge. |
| `firmware/main/record.h` / `.c` | Modify | `REC_CMD_FIELD_TAKE`, `record_post_field_take()`, the ring copy, the WAV + `field.csv` write, the storage floor and its counters. |
| `firmware/main/ui/ui_assist.c` | Modify | "Aufnahme" switch + "REC" badge. |
| `firmware/main/console.c` | Modify | `field on\|off`; `status` reports the toggle and the counters. |
| `scripts/pull-recordings.sh` | Modify | Pull `field/`, append its rows to `sessions.csv` with the four device columns. |
| `tests/test_pull_recordings.py` | Modify | Field pull + column mapping. |
| `tests/test_ingest.py` | Modify | A field take passes the existing wav/row count verification. |
| `kws_de/qc.py` | Modify | `field` content mode, wake split, grammar labels, `device_intent`/`agrees`, the report's Field section. |
| `tests/test_qc.py` | Modify | Field-mode tests; the two `counts ==` assertions gain the new keys. |
| `kws_de/eval.py` | Modify | `field_figures()`, `render_field_section()`, `--qc`, wiring into the recordings section. |
| `tests/test_eval_recordings.py` | Modify | Field figures + rendered section. |
| `docs/sphinx/requirements.rst` | Modify | `REQ_FW_FIELD_CAPTURE`, `REQ_PIPE_FIELD_LABELS`. |
| `docs/sphinx/tests.rst` | Modify | Four test entries linking those requirements. |
| `docs/sphinx/firmware.rst`, `pipeline.rst` | Modify | Prose: the Assistent-mode toggle; the field branch of the QC rules. |
| `firmware/README.md` | Modify | Manual test checklist entry. |
| `docs/paper-notes.md` | Modify | Research-log entry with the first real field numbers. |

Task order follows the spec's §7: firmware capture + host tests + on-device check (Tasks 1–2), ingest (Task 3), QC field mode (Task 4), eval Field section (Task 5), docs (Task 6).

---

### Task 1: Field-capture window arithmetic (pure C, host-tested)

The gate's twin: a small piece of pure logic that decides *whether* a take may be written and *which* span of the ring it is. Kept out of `wake.cc` for the same reason `assist_gate.c` is: it is the part worth testing on the host.

**Files:**

- Create: `firmware/main/field.h`
- Create: `firmware/main/field.c`
- Test: `firmware/test/test_field.c`
- Modify: `firmware/test/Makefile:3` (the `TESTS` list) and a new target after the `test_assist_gate` rule

**Interfaces:**

- Consumes: `ASSIST_WINDOW_MS` (2500) from `firmware/main/assist_gate.h`; `AUDIO_RING_SAMPLES` from `firmware/main/audio.h`; `KWS_SAMPLE_RATE` (16000) from `firmware/main/gen/features_config.h`.
- Produces, for Tasks 2:
  - `void field_reset(field_state_t *f)`
  - `void field_set_enabled(field_state_t *f, bool on)`
  - `void field_on_wake(field_state_t *f, uint32_t fire_pos)`
  - `bool field_take_span(const field_state_t *f, uint32_t *start, uint32_t *len)`
  - `void field_disarm(field_state_t *f)`
  - `typedef struct { bool enabled; bool armed; uint32_t fire_pos; uint32_t taken; uint32_t dropped; } field_state_t;`
  - `typedef struct { uint32_t start; uint32_t len; uint32_t fire_ms; float wake_prob; char intent[64]; char words[96]; } field_take_t;`
  - `FIELD_PREROLL_MS` 1000, `FIELD_TAKE_MS` 3500, `FIELD_PREROLL_SAMPLES` 16000, `FIELD_WINDOW_SAMPLES` 40000, `FIELD_TAKE_SAMPLES` 56000

- [ ] **Step 1: Write the failing host test**

Create `firmware/test/test_field.c`:

```c
/* Host test for the field-capture window arithmetic (REQ_FW_FIELD_CAPTURE). */
#include "field.h"
#include <assert.h>
#include <stdio.h>

int main(void)
{
    field_state_t f;
    uint32_t start = 0, len = 0;

    /* Off by default: a wake fire arms nothing and no span is offered. Capture
       is opt-in — this is the assertion that says so. */
    field_reset(&f);
    assert(!f.enabled);
    field_on_wake(&f, 100000);
    assert(!field_take_span(&f, &start, &len));

    /* Enabled: the take is pre-roll + window and ends exactly at the window's close. */
    field_set_enabled(&f, true);
    field_on_wake(&f, 100000);
    assert(field_take_span(&f, &start, &len));
    assert(start == 100000 - FIELD_PREROLL_SAMPLES);
    assert(len == FIELD_TAKE_SAMPLES);
    assert(start + len == 100000 + FIELD_WINDOW_SAMPLES);

    /* A second fire inside an open window keeps the FIRST fire's position:
       assist_gate extends the window, so one interaction stays one take, and
       its pre-roll still holds the wake phrase the user actually said. */
    field_on_wake(&f, 120000);
    assert(field_take_span(&f, &start, &len));
    assert(start == 100000 - FIELD_PREROLL_SAMPLES);

    /* Disarmed once the take has been handed over: no second copy of it. */
    field_disarm(&f);
    assert(!field_take_span(&f, &start, &len));

    /* A fire in the first second after boot shortens the take instead of
       reading in front of the start of the ring. */
    field_reset(&f);
    field_set_enabled(&f, true);
    field_on_wake(&f, 8000);
    assert(field_take_span(&f, &start, &len));
    assert(start == 0);
    assert(len == 8000 + FIELD_WINDOW_SAMPLES);

    /* Turning capture off drops a pending take: the toggle is the control. */
    field_reset(&f);
    field_set_enabled(&f, true);
    field_on_wake(&f, 100000);
    field_set_enabled(&f, false);
    assert(!field_take_span(&f, &start, &len));

    printf("test_field OK\n");
    return 0;
}
```

- [ ] **Step 2: Add the target to the host-test Makefile**

In `firmware/test/Makefile`, append `test_field` to the `TESTS` list on line 3:

```make
TESTS := test_mfcc test_stream test_wav test_prompts test_vad test_wakefront test_assist_gate test_field
```

and add this rule directly after the `test_assist_gate` rule (the recipe line is indented with a real **tab**, as every make recipe must be):

<!-- markdownlint-disable MD010 -->

```make
test_field: test_field.c ../main/field.c
	$(CC) $(CFLAGS) -o $@ $^
```

<!-- markdownlint-enable MD010 -->

- [ ] **Step 3: Run the test to verify it fails**

Run: `make -C firmware/test test_field`
Expected: FAIL — `fatal error: 'field.h' file not found`

- [ ] **Step 4: Write the header**

Create `firmware/main/field.h`:

```c
/**
 * @file field.h
 * @brief Field capture in Assistent mode: which span of the audio ring one
 * real interaction occupies, and whether it may be written at all.
 *
 * Capture is opt-in and the toggle is persisted, so the decision "is there a
 * take, and where is it" is a tiny piece of state plus one subtraction. It
 * lives here, in pure C with no FreeRTOS, no clock and no globals, for the
 * same reason assist_gate.c does: it is the part worth testing on the host.
 *
 * Positions are absolute sample counts as returned by audio_write_pos().
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "assist_gate.h"
#include "audio.h"
#include "gen/features_config.h"

/** Audio kept in front of the wake fire, so the take contains the wake phrase. */
#define FIELD_PREROLL_MS 1000
/** One take: the pre-roll plus the assist window the recogniser listened in. */
#define FIELD_TAKE_MS (FIELD_PREROLL_MS + ASSIST_WINDOW_MS)
#define FIELD_PREROLL_SAMPLES (KWS_SAMPLE_RATE * FIELD_PREROLL_MS / 1000)
#define FIELD_WINDOW_SAMPLES (KWS_SAMPLE_RATE * ASSIST_WINDOW_MS / 1000)
#define FIELD_TAKE_SAMPLES (FIELD_PREROLL_SAMPLES + FIELD_WINDOW_SAMPLES)
/** Worst case between the window closing and the record task starting the copy
 *  (a full recogniser step plus scheduling), rounded up to 0.2 s. */
#define FIELD_COPY_LATENCY_SAMPLES (KWS_SAMPLE_RATE / 5)

/* The ring must still hold the whole take when the copy starts. It is 10 s
   today, so this is a guard against a future shrink, not a constraint. */
_Static_assert(AUDIO_RING_SAMPLES >= FIELD_TAKE_SAMPLES + FIELD_COPY_LATENCY_SAMPLES,
               "audio ring must hold pre-roll + assist window + the copy latency");

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Capture state. Zero-initialise via field_reset(); `fire_pos` is only
 *  meaningful while `armed`. */
typedef struct {
    bool enabled;       /**< The user's toggle, restored from NVS at boot. */
    bool armed;         /**< A wake fire is waiting for its window to close. */
    uint32_t fire_pos;  /**< audio_write_pos() at the fire that armed us. */
    uint32_t taken;     /**< Field takes saved since boot. */
    uint32_t dropped;   /**< Field takes dropped because storage was low. */
} field_state_t;

/** @brief One take handed from the wake task to the record task. The device's
 *  own prediction travels with the audio; it is scored on the workstation and
 *  is never used as a label. */
typedef struct {
    uint32_t start;     /**< Absolute position of the take's first sample. */
    uint32_t len;       /**< Samples to copy out of the ring. */
    uint32_t fire_ms;   /**< ms since boot of the wake fire (names the file). */
    float wake_prob;    /**< Wake probability at that fire. */
    char intent[64];    /**< Ordered fired command words, space-joined; "" if none. */
    char words[96];     /**< The same fires as "<word>:<conf>", joined by '|'. */
} field_take_t;

/** @brief Reset to disabled with zeroed accounting. */
void field_reset(field_state_t *f);
/** @brief Turn capture on/off. Turning it off also drops a pending take. */
void field_set_enabled(field_state_t *f, bool on);
/** @brief A wake fire arrived at ring position @p fire_pos. No-op while disabled. */
void field_on_wake(field_state_t *f, uint32_t fire_pos);
/**
 * @brief The span to copy for the armed take.
 * @return false if capture is off or nothing is armed; otherwise @p start and
 *         @p len describe the take and `start + len` is the window's end.
 */
bool field_take_span(const field_state_t *f, uint32_t *start, uint32_t *len);
/** @brief Forget the armed take (it has been handed to the recorder). */
void field_disarm(field_state_t *f);

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 5: Write the implementation**

Create `firmware/main/field.c`:

```c
#include "field.h"
#include <string.h>

void field_reset(field_state_t *f)
{
    memset(f, 0, sizeof *f);
}

void field_set_enabled(field_state_t *f, bool on)
{
    f->enabled = on;
    if (!on) f->armed = false;
}

void field_on_wake(field_state_t *f, uint32_t fire_pos)
{
    if (!f->enabled) return;
    /* A fire inside an open window extends it (assist_gate_on_wake) rather than
       starting a second one, so the take keeps the first fire's position: one
       interaction, one take, whose pre-roll holds the phrase actually spoken. */
    if (f->armed) return;
    f->armed = true;
    f->fire_pos = fire_pos;
}

bool field_take_span(const field_state_t *f, uint32_t *start, uint32_t *len)
{
    if (!f->enabled || !f->armed) return false;
    /* ponytail: a fire less than the pre-roll into the ring is treated as a
       boot-time fire and the take is shortened to what exists. The same test is
       true once every 74 h, when the uint32 sample counter wraps; the cost is
       one truncated take, not a read of stale audio. Track the ring's own start
       position if that ever matters. */
    if (f->fire_pos < FIELD_PREROLL_SAMPLES) {
        *start = 0;
        *len = f->fire_pos + FIELD_WINDOW_SAMPLES;
    } else {
        *start = f->fire_pos - FIELD_PREROLL_SAMPLES;
        *len = FIELD_TAKE_SAMPLES;
    }
    return true;
}

void field_disarm(field_state_t *f)
{
    f->armed = false;
}
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `make -C firmware/test test_field && ./firmware/test/test_field`
Expected: `test_field OK`

- [ ] **Step 7: Run the whole host suite (nothing else moved)**

Run: `make -C firmware/test`
Expected: every binary runs, ending with `host tests OK`

- [ ] **Step 8: Commit**

```bash
git add firmware/main/field.h firmware/main/field.c firmware/test/test_field.c firmware/test/Makefile
git commit -m "$(cat <<'EOF'
feat(firmware): field-capture window arithmetic, host-tested

Pure C, no FreeRTOS: the opt-in toggle, the wake fire's ring position, and the
pre-roll + assist-window span one interaction occupies. A static_assert ties the
audio ring length to that span so a future shrink fails the build.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016YVjMuh5AT7hGvYf4EtfUM
EOF
)"
```

---

### Task 2: Firmware capture — toggle, take, CSV, badge, console

Wires Task 1's logic into the device: the toggle (persisted, visible, console-settable), the device prediction per window, the post on the window's closing edge, and the copy + write on the record task.

**Files:**

- Modify: `firmware/main/storage.h`, `firmware/main/storage.c` (add `storage_root()` — skip if the microSD branch already landed it)
- Modify: `firmware/main/recognise.h:14-20` (`recognise_status_t`), `firmware/main/recognise.cc:170-180` (the status mutex block) and `recognise.cc`'s `recognise_listen_for`
- Modify: `firmware/main/wake.h`, `firmware/main/wake.cc:43-52` (statics), `wake.cc:93-106` (restart), `wake.cc:178-191` (the assist block), `wake.cc:213-221` (`wake_start`)
- Modify: `firmware/main/record.h`, `firmware/main/record.c`
- Modify: `firmware/main/ui/ui_assist.c`
- Modify: `firmware/main/console.c`
- Test: on-device check (no host test — this is task wiring and FAT I/O)

**Interfaces:**

- Consumes: everything Task 1 produced; `audio_write_pos()`, `audio_read(uint32_t end, int16_t *dst, uint32_t n)` (`audio.h`); `storage_free_bytes()`, `STORAGE_MIN_FREE_BYTES` (`storage.h`); `wav_write_header(uint8_t *hdr, uint32_t n_samples, uint32_t sr)`, `WAV_HEADER_BYTES` (`wav.h`); `assist_gate_on_wake/tick` (`assist_gate.h`).
- Produces:
  - `const char *storage_root(void)` — `"/rec"`
  - `bool wake_field_get(void)`, `void wake_field_set(bool on)` (`wake.h`)
  - `void record_post_field_take(const field_take_t *t)` (`record.h`)
  - `REC_CMD_FIELD_TAKE` in `record_cmd_t`
  - `uint32_t field_takes; uint32_t field_dropped;` appended to `record_status_t`
  - `char window_intent[64]; char window_words[96];` appended to `recognise_status_t`
  - Console: `field on|off`; `status` gains a `field <on|off> takes <N> dropped <N>` line
  - On device: `storage_root()/field/spkNN/<boot-ms>.wav` + `storage_root()/field/spkNN/field.csv` with header `file,fire_ms,wake_prob,device_intent,device_words,window_ms,ms,peak_dbfs`

- [ ] **Step 1: Add `storage_root()` (skip if the microSD branch landed it)**

Check first: `grep -n storage_root firmware/main/storage.h`. If it prints a declaration, skip this step and use it as it stands. Otherwise, in `firmware/main/storage.h` after the `storage_wl_handle` declaration:

```c
/** @brief Mount point of the recording storage ("/rec"). Every path the
 *  recorder writes is built from this, so moving the tree onto a card is a
 *  change here and nowhere else. */
const char *storage_root(void);
```

and in `firmware/main/storage.c`, at the end of the file:

```c
const char *storage_root(void) { return "/rec"; }
```

- [ ] **Step 2: Give the recogniser a per-window prediction**

In `firmware/main/recognise.h`, append two members to `recognise_status_t` (after `fired_count`):

```c
    char window_intent[64];  /**< Command words fired since the last recognise_listen_for(), in order, space-joined ("Licht Küche an"); empty if none. */
    char window_words[96];   /**< The same fires as "<word>:<conf>" entries joined by '|' ("Licht:0.93|an:0.88"). */
```

In `firmware/main/recognise.cc`, inside the existing `xSemaphoreTake(s_lock, ...)` block, directly after `if (fired >= 0) s_st.fired_count++;`:

```c
        if (fired >= 0) {
            /* Field capture's device prediction: what THIS window fired, in
               order, kept next to the audio so the workstation can score the
               deployed model against Whisper's label. Never a label itself. */
            const char *w = KWS_LABELS[fired];
            size_t n = strlen(s_st.window_intent);
            snprintf(s_st.window_intent + n, sizeof s_st.window_intent - n, "%s%s", n ? " " : "", w);
            n = strlen(s_st.window_words);
            snprintf(s_st.window_words + n, sizeof s_st.window_words - n, "%s%s:%.2f",
                     n ? "|" : "", w, (double)probs[fired]);
        }
```

and in `recognise_listen_for()`, clear both when a new window opens:

```c
extern "C" void recognise_listen_for(uint32_t ms)
{
    xSemaphoreTake(s_lock, portMAX_DELAY);
    s_st.window_intent[0] = 0;
    s_st.window_words[0] = 0;
    xSemaphoreGive(s_lock);
    s_off_at_us = esp_timer_get_time() + (int64_t)ms * 1000;
    s_active = true;
}
```

Add `#include <cstring>` to `recognise.cc` if `strlen` is not already available there.

- [ ] **Step 3: Recorder side — the command, the payload slot, the write**

In `firmware/main/record.h`, add the include and the new command, status fields and post function:

```c
#include "field.h"
```

```c
    REC_CMD_PAUSE,               /**< Pause the recorder (goes idle, waits for a command). */
    REC_CMD_FIELD_TAKE,          /**< Assist mode: copy one field take out of the audio ring and save it.
                                      The payload is set by record_post_field_take(). */
```

```c
    int saved_takes;                       /**< Takes saved since the last REC_CMD_START_SESSION. */
    uint32_t field_takes;                  /**< Field takes saved since boot. */
    uint32_t field_dropped;                /**< Field takes dropped: storage below STORAGE_MIN_FREE_BYTES. */
```

```c
/** @brief Post one field take (assist mode). Copies @p t into the recorder's
 *  single pending slot and posts REC_CMD_FIELD_TAKE. Non-blocking. */
void record_post_field_take(const field_take_t *t);
```

In `firmware/main/record.c`, add `#include "assist_gate.h"` and, next to the other statics:

```c
static field_take_t s_field_pending;          /* payload for REC_CMD_FIELD_TAKE */
```

Then, after `save_take()`:

```c
void record_post_field_take(const field_take_t *t)
{
    /* ponytail: one pending slot. The next fire cannot arrive before the wake
       refractory (1.5 s) plus the 2.5 s window, and a save costs ~300 ms, so
       the slot is always free by then; if it ever is not, the newer take wins.
       Give it a queue if the window ever gets shorter than a save. */
    xSemaphoreTake(s_lock, portMAX_DELAY);
    s_field_pending = *t;
    xSemaphoreGive(s_lock);
    record_cmd_t cmd = REC_CMD_FIELD_TAKE;
    xQueueSend(s_cmds, &cmd, 0);
}

/* storage_root()/field/spkNN/<boot-ms>.wav plus one field.csv row. Runs on the
   record task AFTER the assist window has closed: the recogniser is already
   off, so the 100-300 ms FAT write cannot lengthen a recognise step. The
   speaker id is the current NVS id and is never bumped here — one boot of one
   user is one field directory. */
static void save_field_take(void)
{
    field_take_t t;
    char speaker[8];
    xSemaphoreTake(s_lock, portMAX_DELAY);
    t = s_field_pending;
    strlcpy(speaker, s_st.speaker, sizeof speaker);
    xSemaphoreGive(s_lock);

    if (storage_free_bytes() < STORAGE_MIN_FREE_BYTES) {
        ESP_LOGW(TAG, "field: dropped, storage low");
        xSemaphoreTake(s_lock, portMAX_DELAY); s_st.field_dropped++; xSemaphoreGive(s_lock);
        return;
    }
    if (t.len > TAKE_MAX) t.len = TAKE_MAX;
    audio_read(t.start + t.len, s_take, t.len);
    int peak = 0;
    for (uint32_t i = 0; i < t.len; i++) {
        int a = s_take[i] < 0 ? -s_take[i] : s_take[i];
        if (a > peak) peak = a;
    }
    float peak_dbfs = 20.f * log10f((peak > 0 ? peak : 1) / 32768.f);

    char dir[96], path[160], csv[128], name[24];
    snprintf(dir, sizeof dir, "%s/field", storage_root());              mkdir(dir, 0777);
    snprintf(dir, sizeof dir, "%s/field/%s", storage_root(), speaker);  mkdir(dir, 0777);
    snprintf(name, sizeof name, "%lu.wav", (unsigned long)t.fire_ms);
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) { ESP_LOGE(TAG, "field: open %s failed", path); return; }
    uint8_t hdr[WAV_HEADER_BYTES];
    wav_write_header(hdr, t.len, KWS_SAMPLE_RATE);
    fwrite(hdr, 1, sizeof hdr, f);
    fwrite(s_take, sizeof(int16_t), t.len, f);
    fclose(f);

    snprintf(csv, sizeof csv, "%s/field.csv", dir);
    struct stat st; int fresh = stat(csv, &st) != 0;
    FILE *c = fopen(csv, "a");
    if (!c) { ESP_LOGE(TAG, "field: csv open failed"); return; }
    if (fresh) fputs("file,fire_ms,wake_prob,device_intent,device_words,window_ms,ms,peak_dbfs\n", c);
    fprintf(c, "%s,%lu,%.3f,%s,%s,%d,%lu,%.1f\n", name, (unsigned long)t.fire_ms,
            (double)t.wake_prob, t.intent, t.words, ASSIST_WINDOW_MS,
            (unsigned long)(t.len * 1000 / KWS_SAMPLE_RATE), peak_dbfs);
    fclose(c);
    xSemaphoreTake(s_lock, portMAX_DELAY); s_st.field_takes++; xSemaphoreGive(s_lock);
    ESP_LOGI(TAG, "field: saved %s (%lu samples, intent \"%s\")", path,
             (unsigned long)t.len, t.intent);
}
```

and in `record_task()`'s `switch (cmd)`, add:

```c
        case REC_CMD_FIELD_TAKE:
            /* Assist mode only, where the guided recorder is paused. If a
               guided session is running, ignore it rather than corrupt the
               take in progress. */
            if (s_paused) save_field_take();
            break;
```

- [ ] **Step 4: Wake side — the toggle, its NVS home, the post on the closing edge**

In `firmware/main/wake.h`, before the closing `#ifdef __cplusplus`:

```c
/** @brief Is field capture on? Restored from NVS ("kws"/"field") at wake_start(). */
bool wake_field_get(void);
/** @brief Turn field capture on/off and persist it. Off drops any pending take. */
void wake_field_set(bool on);
```

In `firmware/main/wake.cc`, add `#include "field.h"`, `#include "record.h"` and `#include "nvs.h"`, and next to the other statics:

```c
static field_state_t s_field;        /* assist mode only: opt-in capture of real interactions */
static uint32_t s_fire_ms;           /* ms-since-boot of the fire that armed s_field */
static float s_fire_prob;            /* wake probability at that fire */
```

Add, above `wake_task`:

```c
/* The window has closed: hand the recorder the span to copy. NOTHING is written
   here — the copy and the FAT write happen on the record task, with the
   recogniser already off, so no I/O can lengthen a recognise step. */
static void post_field_take(void)
{
    field_take_t t = {};
    if (!field_take_span(&s_field, &t.start, &t.len)) return;
    field_disarm(&s_field);
    recognise_status_t rst;
    recognise_get_status(&rst);
    t.fire_ms = s_fire_ms;
    t.wake_prob = s_fire_prob;
    strlcpy(t.intent, rst.window_intent, sizeof t.intent);
    strlcpy(t.words, rst.window_words, sizeof t.words);
    record_post_field_take(&t);
}
```

In the `s_restart` block, after `assist_gate_reset(&s_gate);`:

```c
            field_disarm(&s_field);     /* a new session never inherits a pending take */
```

Replace the body of the `if (assist) { ... }` block that opens/closes the window with:

```c
            if (assist) {
                if (fired) {
                    assist_gate_on_wake(&s_gate, now_ms);
                    field_on_wake(&s_field, audio_write_pos());
                    s_fire_ms = now_ms;
                    s_fire_prob = prob;
                }
                bool listen = assist_gate_tick(&s_gate, now_ms);
                if (listen != s_listening) {
                    s_listening = listen;
                    /* Opening the window hands the recogniser its own deadline
                       so it stops even if this task stops being scheduled. */
                    if (listen) {
                        recognise_listen_for(ASSIST_WINDOW_MS);
                    } else {
                        recognise_set_active(false);
                        post_field_take();
                    }
                    ESP_LOGI(TAG, "assist: recogniser %s (window %lu)", listen ? "on" : "off",
                             (unsigned long)s_gate.windows);
                }
            }
```

At the top of `wake_start()`, restore the toggle:

```c
    nvs_handle_t h;
    if (nvs_open("kws", NVS_READWRITE, &h) == ESP_OK) {
        uint8_t on = 0;
        if (nvs_get_u8(h, "field", &on) != ESP_OK) on = 0;   /* off until turned on once */
        field_set_enabled(&s_field, on != 0);
        nvs_close(h);
    }
```

and at the end of the file:

```c
extern "C" bool wake_field_get(void) { return s_field.enabled; }

extern "C" void wake_field_set(bool on)
{
    field_set_enabled(&s_field, on);
    nvs_handle_t h;
    if (nvs_open("kws", NVS_READWRITE, &h) != ESP_OK) return;
    nvs_set_u8(h, "field", on ? 1 : 0);
    nvs_commit(h);
    nvs_close(h);
}
```

- [ ] **Step 5: UI — the "Aufnahme" switch and the "REC" badge**

In `firmware/main/ui/ui_assist.c`, extend the statics and add the callback:

```c
static lv_obj_t *scr, *l_state, *l_big, *l_stats, *l_rec, *sw_field;
```

```c
/* The toggle reads its own state rather than the event target: LVGL 9 hands
   back a void*, and the switch is a file static anyway. */
static void on_field(lv_event_t *e)
{
    (void)e;
    bool on = lv_obj_has_state(sw_field, LV_STATE_CHECKED);
    wake_field_set(on);
    if (on) lv_obj_clear_flag(l_rec, LV_OBJ_FLAG_HIDDEN);
    else lv_obj_add_flag(l_rec, LV_OBJ_FLAG_HIDDEN);
}
```

In `ui_show_assist()`, after the `l_stats` block and before the Menu button:

```c
    /* Opt-in field capture: a switch the user must turn on once, and a "REC"
       badge that is the only visible difference while it is on. */
    l_rec = lv_label_create(scr);
    lv_label_set_text(l_rec, "REC");
    lv_obj_set_style_text_color(l_rec, lv_palette_main(LV_PALETTE_RED), 0);
    lv_obj_align(l_rec, LV_ALIGN_TOP_RIGHT, -12, 12);

    lv_obj_t *lf = lv_label_create(scr);
    lv_label_set_text(lf, "Aufnahme");
    lv_obj_set_style_text_color(lf, lv_color_hex(0x8a94a0), 0);
    lv_obj_align(lf, LV_ALIGN_BOTTOM_LEFT, 12, -24);

    sw_field = lv_switch_create(scr);
    lv_obj_align(sw_field, LV_ALIGN_BOTTOM_LEFT, 100, -28);
    lv_obj_add_event_cb(sw_field, on_field, LV_EVENT_VALUE_CHANGED, NULL);
    if (wake_field_get()) lv_obj_add_state(sw_field, LV_STATE_CHECKED);
    else lv_obj_add_flag(l_rec, LV_OBJ_FLAG_HIDDEN);
```

- [ ] **Step 6: Console — `field on|off` and the `status` line**

In `firmware/main/console.c`, add a branch after the `wakefire` branch in `handle_line()`:

```c
    } else if (strcmp(cmd, "field") == 0) {
        char *arg = strtok(NULL, " ");
        if (!arg) { printf("err missing on|off\n"); return; }
        if (strcmp(arg, "on") == 0) wake_field_set(true);
        else if (strcmp(arg, "off") == 0) wake_field_set(false);
        else { printf("err unknown field arg %s\n", arg); return; }
        printf("ok\n");
```

and in the `status` branch, after the `models` line:

```c
        record_status_t fst;
        record_get_status(&fst);
        printf("field %s takes %lu dropped %lu\n", wake_field_get() ? "on" : "off",
               (unsigned long)fst.field_takes, (unsigned long)fst.field_dropped);
```

- [ ] **Step 7: Run the host suite (field.c must still build and pass standalone)**

Run: `make -C firmware/test`
Expected: ends with `host tests OK`

- [ ] **Step 8: Build the firmware**

Run: `docker run --rm -v "$PWD/firmware:/project" -w /project espressif/idf:v5.5.5 idf.py build`
Expected: `Project build complete.` and a `kws_de_fw.bin` size line; no new warnings from `field.c`, `record.c`, `wake.cc`, `recognise.cc`, `ui_assist.c`, `console.c`.

- [ ] **Step 9: Flash and check on the device**

The CoreS3 hangs off the remote Mac, so flashing and the console go through the helper scripts (referenced here only — never in a committed doc):

```bash
S=~/.claude/skills/flashing-cores3-on-bar
$S/flash-bar.sh -l 8                        # flash + 8 s of boot log
$S/console-bar.sh status                    # expect: "field off takes 0 dropped 0" ... "ok"
$S/console-bar.sh 'field on'                # expect: "ok"
$S/console-bar.sh -l 4 'mode assist'        # expect: "ok"; the screen shows the REC badge
$S/console-bar.sh -l 12 wakefire            # inject one fire, then say a command
$S/console-bar.sh status                    # expect: "field on takes 1 dropped 0"
```

Then power-cycle the device (unplug/replug) and run `$S/console-bar.sh status` again: expect `field on` — the toggle survived the reboot.

Check the three things the spec asks for:

1. the toggle persists (the `status` line after the power cycle);
2. a take appears — put the device in USB mode (`$S/console-bar.sh 'mode usb'`) and confirm `field/spkNN/<boot-ms>.wav` plus `field.csv` on the `KWSREC` drive;
3. the recognise step time does not change during the window — in the 12 s log from the `wakefire` line, the `recognise` task's `step <N> ms` lines inside the window must stay in their usual range (~46 ms), with no 100–300 ms outlier, which is what "no I/O during the window" means in practice. Record the observed range; it is the number Task 6 writes into `docs/paper-notes.md`.

- [ ] **Step 10: Commit**

```bash
git add firmware/main/storage.h firmware/main/storage.c firmware/main/recognise.h firmware/main/recognise.cc \
        firmware/main/wake.h firmware/main/wake.cc firmware/main/record.h firmware/main/record.c \
        firmware/main/ui/ui_assist.c firmware/main/console.c
git commit -m "$(cat <<'EOF'
feat(firmware): opt-in field capture in Assistent mode

A wake fire arms a take; when the window closes the wake task hands the record
task the ring span plus the device's own prediction, and the record task — idle
in Assistent mode — copies and writes it. No file I/O happens while the
recogniser runs. Toggle persisted in NVS, shown as an "Aufnahme" switch and a
REC badge, settable over the console; takes and storage-low drops are reported
by `status`.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016YVjMuh5AT7hGvYf4EtfUM
EOF
)"
```

---

### Task 3: Ingest — pull `field/` and carry the device columns

**Files:**

- Modify: `scripts/pull-recordings.sh:22` (the `sessions.csv` header) and the `spk*/` loop
- Test: `tests/test_pull_recordings.py`, `tests/test_ingest.py`

**Interfaces:**

- Consumes: the device layout Task 2 produces — `field/spkNN/<boot-ms>.wav` and `field/spkNN/field.csv` with header `file,fire_ms,wake_prob,device_intent,device_words,window_ms,ms,peak_dbfs`.
- Produces, for Task 4: `sessions.csv` with the header
  `speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts,fire_ms,wake_prob,device_intent,device_words`,
  one row per field take with `set=field`, `prompt` empty, `file=field/spkNN/<boot-ms>.wav`, `seed` empty, `ts` = `fire_ms`; guided rows keep their nine columns and read back as `None` for the four new ones.
- `scripts/ingest.sh` is unchanged: its wav-count and row-count verification already covers field takes one-to-one.

- [ ] **Step 1: Write the failing pull test**

Add to `tests/test_pull_recordings.py`:

```python
def _fake_drive_with_field(root: Path) -> Path:
    mnt = _fake_drive(root)
    (mnt / "field" / "spk03").mkdir(parents=True)
    (mnt / "field" / "spk03" / "123456.wav").write_bytes(b"RIFF" + b"\0" * 40)
    (mnt / "field" / "spk03" / "field.csv").write_text(
        "file,fire_ms,wake_prob,device_intent,device_words,window_ms,ms,peak_dbfs\n"
        "123456.wav,123456,0.910,Licht an,Licht:0.93|an:0.88,2500,3500,-8.4\n"
    )
    return mnt


def test_pull_copies_field_takes_and_appends_device_columns(tmp_path):
    mnt = _fake_drive_with_field(tmp_path)
    dest = tmp_path / "recordings"
    r = _run(mnt, dest)
    assert r.returncode == 0, r.stderr
    assert (dest / "field" / "spk03" / "123456.wav").exists()
    assert not (dest / "field" / "spk03" / "field.csv").exists()  # the CSV folds into sessions.csv

    rows = list(csv.DictReader((dest / "sessions.csv").open()))
    field = [r for r in rows if r["set"] == "field"]
    assert len(field) == 1
    assert field[0]["speaker"] == "spk03"
    assert field[0]["prompt"] == ""
    assert field[0]["file"] == "field/spk03/123456.wav"
    assert field[0]["ms"] == "3500" and field[0]["peak_dbfs"] == "-8.4"
    assert field[0]["fire_ms"] == "123456" and field[0]["ts"] == "123456"
    assert field[0]["wake_prob"] == "0.910"
    assert field[0]["device_intent"] == "Licht an"
    assert field[0]["device_words"] == "Licht:0.93|an:0.88"
    assert not (mnt / "field" / "spk03").exists()  # cleared after a successful copy
```

Add `import csv` at the top of the file.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_pull_recordings.py::test_pull_copies_field_takes_and_appends_device_columns -v`
Expected: FAIL — `assert False` on `(dest / "field" / "spk03" / "123456.wav").exists()`

- [ ] **Step 3: Extend the pull script**

In `scripts/pull-recordings.sh`, change the header line (line 22) to:

```bash
[[ -f $sessions ]] || echo "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts,fire_ms,wake_prob,device_intent,device_words" > "$sessions"
```

and add this loop directly after the existing `for spk in "$mnt"/spk*/; do ... done` loop:

```bash
# Field takes (Assistent mode, opt-in): same shape as a guided take plus what
# the device itself recognised. field.csv is folded into sessions.csv rather
# than copied, so QC reads one file for every set.
for spk in "$mnt"/field/spk*/; do
  spk=${spk%/}; name=$(basename "$spk")
  rsync -a --exclude field.csv "$spk/" "$dest/field/$name/"
  if [[ -f $spk/field.csv ]]; then
    # field.csv: file,fire_ms,wake_prob,device_intent,device_words,window_ms,ms,peak_dbfs
    # sessions:  speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts,fire_ms,wake_prob,device_intent,device_words
    tail -n +2 "$spk/field.csv" | awk -F, -v OFS=, -v s="$name" -v p="$pulled" \
      '{print s, p, "", "field/" s "/" $1, $7, $8, "field", "", $2, $2, $3, $4, $5}' >> "$sessions"
  fi
  rm -rf "$spk"
  echo "pulled field/$name"
done
rmdir "$mnt/field" 2>/dev/null || true
```

- [ ] **Step 4: Run the pull tests**

Run: `uv run pytest tests/test_pull_recordings.py -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Write the failing ingest test**

Add to `tests/test_ingest.py`:

```python
def _remote_fixture_with_field(tmp_path: Path) -> Path:
    # one guided take + one field take: ingest.sh's wav/row verification must
    # count both, since each field wav has exactly one sessions.csv row.
    remote = _remote_fixture(tmp_path)
    (remote / "field" / "spk02").mkdir(parents=True)
    (remote / "field" / "spk02" / "123456.wav").write_bytes(b"RIFF")
    with (remote / "sessions.csv").open("a") as fh:
        fh.write("spk02,2026-09-02T00:00:00Z,,field/spk02/123456.wav,3500,-8.4,field,,123456,123456,0.910,Licht an,Licht:0.93|an:0.88\n")
    return remote


def test_ingest_counts_field_takes(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log = tmp_path / "calls.log"
    remote = _remote_fixture_with_field(tmp_path)
    _fake_ssh(bin_, log, remote)
    _fake_scp_and_rsync(bin_, log, remote)
    r = subprocess.run(
        ["bash", str(SCRIPT), "-H", "devhost"],
        env=_env(tmp_path, bin_),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "ingested 2 takes" in r.stdout
    dest = next((tmp_path / "root" / "data" / "recordings" / "incoming").iterdir())
    assert (dest / "field" / "spk02" / "123456.wav").exists()
```

Also update `_remote_fixture`'s `sessions.csv` header to the 13-column one, so both tests read the same shape:

```python
    (remote / "sessions.csv").write_text(
        "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts,fire_ms,wake_prob,device_intent,device_words\n"
        'spk02,2026-09-02T00:00:00Z,"Licht",spk02/licht/001.wav,900,-6.0,words,7,1234\n'
    )
```

- [ ] **Step 6: Run the ingest tests**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: PASS, 5 tests — `ingest.sh` itself needed no change.

- [ ] **Step 7: Commit**

```bash
git add scripts/pull-recordings.sh tests/test_pull_recordings.py tests/test_ingest.py
git commit -m "$(cat <<'EOF'
feat(scripts): pull field takes and carry the device columns

pull-recordings.sh pulls field/ like the other sets and folds each field.csv row
into sessions.csv as set=field with fire_ms, wake_prob, device_intent and
device_words appended. ingest.sh needed no change: its wav/row verification
already counts field takes one-to-one.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016YVjMuh5AT7hGvYf4EtfUM
EOF
)"
```

---

### Task 4: QC field mode — wake split, grammar labels, agreement

**Files:**

- Modify: `kws_de/qc.py` (`CAP_MS`, `Take`, `QcRow`, `content_gate`, `_slug_of`, `judge`, `read_sessions`, `run_qc`, two new helpers)
- Test: `tests/test_qc.py`

**Interfaces:**

- Consumes: Task 3's `sessions.csv` columns; `kws_de.grammar.parse(events: list[str]) -> Intent | Rejection`; `kws_de.eval.intent_text(intent) -> str`; the existing `normalise`, `vocab`, `label_for_token`, `_WAKE_RE`, `_next_no`, `_append_index`, `segment_word`.
- Produces, for Task 5:
  - `qc.csv` gains the columns `device_intent` (verbatim from the device) and `agrees` (`"1"`, `"0"`, or `""` when the take did not parse)
  - `run_qc()`'s return dict gains `"field_takes"`, `"field_parsable"`, `"field_agree"` (ints)
  - `report.md` gains a `## Field` section
  - `field_wake_split(tr: Transcript) -> tuple[float | None, list[str]]`
  - `field_intent(tokens: list[str]) -> Intent | Rejection`
  - `WAKE_MAX_S = 1.3`, `WAKE_TAIL_S = 0.15`
- Field takes reach `approved/wake/<spkNN>/`, `approved/phrases/<spkNN>/`, `approved/words/<label>/` and `approved/negatives/<spkNN>/` through the **existing** write paths, so `kws_de.data.merge_recordings` and `kws_de.eval.eval_recordings` need no change to see them.

- [ ] **Step 1: Write the failing unit tests for the two new pure helpers**

Add to `tests/test_qc.py`:

```python
def test_field_wake_split_cuts_the_wake_phrase_and_returns_the_command():
    tr = {
        "text": "Hey Bus Licht Küche an",
        "words": [
            {"word": "Hey", "start": 0.10, "end": 0.35},
            {"word": "Bus", "start": 0.36, "end": 0.60},
            {"word": "Licht", "start": 1.40, "end": 1.70},
            {"word": "Küche", "start": 1.75, "end": 2.05},
            {"word": "an", "start": 2.10, "end": 2.30},
        ],
    }
    cut_s, tokens = qc.field_wake_split(tr)
    assert cut_s == pytest.approx(0.75)  # end of "Bus" + WAKE_TAIL_S
    assert tokens == ["licht", "küche", "an"]


def test_field_wake_split_ignores_a_late_or_absent_wake_phrase():
    # the phrase matches but lands after WAKE_MAX_S -> not this take's wake word
    late = {
        "text": "Hey Bus an",
        "words": [
            {"word": "Hey", "start": 1.20, "end": 1.45},
            {"word": "Bus", "start": 1.46, "end": 1.80},
            {"word": "an", "start": 1.90, "end": 2.10},
        ],
    }
    assert qc.field_wake_split(late) == (None, ["hey", "bus", "an"])
    # no wake phrase at all: nothing is cut, everything is command material
    plain = {"text": "Licht an", "words": [{"word": "Licht", "start": 0.1, "end": 0.4}]}
    assert qc.field_wake_split(plain) == (None, ["licht", "an"])


def test_field_intent_uses_the_same_grammar_as_the_device():
    from kws_de.grammar import Intent, Rejection

    assert qc.field_intent(["licht", "küche", "an"]) == Intent("Licht", "Küche", "an")
    # filler is dropped, exactly as the sentence prompts' token filter does
    assert qc.field_intent(["mach", "licht", "bitte", "an"]) == Intent("Licht", None, "an")
    # ordinary speech has no command tokens at all -> a Rejection, i.e. kept as
    # negative / _unknown_ material, never dropped
    assert isinstance(qc.field_intent(["wann", "fahren", "wir", "los"]), Rejection)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_qc.py -k field -v`
Expected: FAIL with `AttributeError: module 'kws_de.qc' has no attribute 'field_wake_split'`

- [ ] **Step 3: Add the two helpers and the field content mode**

In `kws_de/qc.py`, extend `CAP_MS` (a take is 3500 ms; the cap leaves margin for a boot-shortened one):

```python
CAP_MS = {"words": 4000, "sentences": 6000, "negatives": 6000, "wake": 4000, "field": 4000}
```

Add the two constants next to `_WAKE_RE`:

```python
# A wake phrase later than this into a field take is not that take's wake phrase
# (the take starts 1.0 s before the fire, so the phrase sits in the first ~1 s).
WAKE_MAX_S = 1.3
# Kept after the end of "bus", so the wake clip is not cut mid-plosive.
WAKE_TAIL_S = 0.15
```

Add the helpers after `label_for_token`:

```python
def field_wake_split(tr: Transcript) -> tuple[float | None, list[str]]:
    """Split a field take's transcript into (seconds at which the wake clip
    ends, the normalised command tokens after it). Returns `(None, all tokens)`
    when the first two words are not the wake phrase inside the first
    WAKE_MAX_S seconds — the take is then all command (or all junk), and
    nothing is cut off as a wake clip."""
    words = tr.get("words", [])
    if len(words) >= 2:
        glued = "".join(normalise(words[0]["word"]) + normalise(words[1]["word"]))
        if _WAKE_RE.fullmatch(glued) and float(words[1]["end"]) <= WAKE_MAX_S:
            rest = [t for w in words[2:] for t in normalise(w["word"])]
            return float(words[1]["end"]) + WAKE_TAIL_S, rest
    return None, normalise(tr.get("text", ""))


def field_intent(tokens: list[str]):
    """The Whisper-derived label for a field take: the command tokens mapped
    back onto config labels and run through the SAME grammar the device uses
    (`kws_de.grammar.parse`), with non-vocabulary words dropped — the identical
    filter `required_tokens(..., "sentences")` applies to a guided prompt. An
    `Intent` is a phrase label; a `Rejection` means the take is kept as
    negative / `_unknown_` material, never dropped."""
    from kws_de.grammar import parse

    v = vocab()
    return parse([label_for_token(t) for t in tokens if t in v])
```

Add the field branch to `content_gate`, directly after the `negatives` branch:

```python
    if set_name == "field":
        # Everything is kept: a field take is real usage, and speech the grammar
        # cannot parse is exactly the negative/`_unknown_` material the model
        # needs. Only silence (or a transcriber that returned nothing) rejects.
        return (1.0, None) if heard else (0.0, "empty_transcript")
```

- [ ] **Step 4: Run the helper tests**

Run: `uv run pytest tests/test_qc.py -k field -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Write the failing end-to-end field tests**

Add to `tests/test_qc.py`:

```python
def _field_session(tmp_path, device_intent: str) -> Path:
    inc = tmp_path / "incoming" / "f1"
    _wav(inc / "field" / "spk05" / "123456.wav", _tone(ms=3500))
    (inc / "sessions.csv").write_text(
        "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts,"
        "fire_ms,wake_prob,device_intent,device_words\n"
        "spk05,t,,field/spk05/123456.wav,3500,-10,field,,123456,123456,0.910,"
        f"{device_intent},Licht:0.93|an:0.88\n"
    )
    return inc


def _field_transcriber(p: Path):
    return {
        "text": "Hey Bus Licht Küche an",
        "words": [
            {"word": "Hey", "start": 0.10, "end": 0.35},
            {"word": "Bus", "start": 0.36, "end": 0.60},
            {"word": "Licht", "start": 1.40, "end": 1.70},
            {"word": "Küche", "start": 1.75, "end": 2.05},
            {"word": "an", "start": 2.10, "end": 2.30},
        ],
    }


def test_run_qc_field_take_splits_wake_labels_by_grammar_and_scores_agreement(tmp_path):
    inc = _field_session(tmp_path, "Licht Küche an")
    qcd, appr = tmp_path / "qc" / "f1", tmp_path / "approved"
    counts = qc.run_qc(inc, qcd, appr, _field_transcriber)

    assert counts["field_takes"] == 1
    assert counts["field_parsable"] == 1
    assert counts["field_agree"] == 1
    assert counts["wake_written"] == 1

    # the wake phrase became a wake clip, cut at the end of "Bus" + 0.15 s
    wake_files = sorted((appr / "wake" / "spk05").glob("*.wav"))
    assert len(wake_files) == 1
    sig, sr = sf.read(wake_files[0], always_2d=True)
    assert 0.70 <= len(sig) / sr <= 0.80

    # the command became an approved phrase with the grammar-derived prompt,
    # segmented into word clips exactly like a guided sentence take
    idx = list(csv.DictReader((appr / "phrases" / "index.csv").open()))
    assert idx[0]["prompt"] == "Licht Küche an" and idx[0]["speaker"] == "spk05"
    assert (appr / "phrases" / "spk05" / "123456_001.wav").exists()
    assert {p.parent.name for p in (appr / "words").rglob("*.wav")} == {"Licht", "Küche", "an"}

    # provenance: the device's own intent is recorded and scored, never used
    row = list(csv.DictReader((qcd / "qc.csv").open()))[0]
    assert row["set"] == "field" and row["verdict"] == "approve"
    assert row["device_intent"] == "Licht Küche an"
    assert row["agrees"] == "1"
    assert "## Field" in (qcd / "report.md").read_text()


def test_run_qc_field_take_records_disagreement_without_relabelling(tmp_path):
    inc = _field_session(tmp_path, "Licht an")  # device missed the zone
    qcd, appr = tmp_path / "qc" / "f1", tmp_path / "approved"
    counts = qc.run_qc(inc, qcd, appr, _field_transcriber)
    assert counts["field_parsable"] == 1 and counts["field_agree"] == 0
    row = list(csv.DictReader((qcd / "qc.csv").open()))[0]
    assert row["agrees"] == "0"
    # the LABEL still comes from Whisper + grammar, never from the device
    idx = list(csv.DictReader((appr / "phrases" / "index.csv").open()))
    assert idx[0]["prompt"] == "Licht Küche an"


def test_run_qc_field_take_that_does_not_parse_is_kept_as_a_negative(tmp_path):
    inc = _field_session(tmp_path, "")
    qcd, appr = tmp_path / "qc" / "f1", tmp_path / "approved"

    def transcriber(p: Path):
        return {
            "text": "wann fahren wir los",
            "words": [{"word": "wann", "start": 0.2, "end": 0.5}],
        }

    counts = qc.run_qc(inc, qcd, appr, transcriber)
    assert counts["field_takes"] == 1 and counts["field_parsable"] == 0
    assert counts["wake_written"] == 0  # no wake phrase in the transcript
    idx = list(csv.DictReader((appr / "negatives" / "index.csv").open()))
    assert idx[0]["prompt"] == "wann fahren wir los"  # the transcript is the prompt
    assert (appr / "negatives" / "spk05" / "123456_001.wav").exists()
    row = list(csv.DictReader((qcd / "qc.csv").open()))[0]
    assert row["verdict"] == "approve" and row["agrees"] == ""


def test_run_qc_field_take_with_an_empty_transcript_is_rejected(tmp_path):
    inc = _field_session(tmp_path, "")
    qcd, appr = tmp_path / "qc" / "f1", tmp_path / "approved"
    counts = qc.run_qc(inc, qcd, appr, lambda p: {"text": "", "words": []})
    assert counts["rejected"] == 1 and counts["field_parsable"] == 0
    row = list(csv.DictReader((qcd / "qc.csv").open()))[0]
    assert row["reason"] == "empty_transcript"
```

Update the two existing `assert counts == {...}` blocks for the new keys:

- in `test_run_qc_word_naming_avoids_bare_vs_phrase_collision_and_is_idempotent`:

```python
    assert counts == {
        "takes": 4,
        "approved": 3,
        "rejected": 1,
        "words_written": 4,
        "words_skipped": 0,
        "wake_written": 0,
        "field_takes": 0,
        "field_parsable": 0,
        "field_agree": 0,
    }
```

- in `test_run_qc_writes_approved_wake_set_and_is_idempotent`, the assertions are on individual keys (`counts["wake_written"]`, `counts["approved"]`, `counts["rejected"]`) and on `counts2 == counts`, so they need no change.

- [ ] **Step 6: Run them to verify they fail**

Run: `uv run pytest tests/test_qc.py -k field -v`
Expected: FAIL with `KeyError: 'field_takes'`

- [ ] **Step 7: Carry the device columns through `Take`, `QcRow`, `read_sessions` and `judge`**

In `kws_de/qc.py`:

```python
@dataclass
class Take:
    file: Path
    set: str
    prompt: str
    speaker: str
    device_intent: str = ""   # what the device itself recognised (field takes only)
    device_words: str = ""    # "<word>:<conf>" entries joined by '|'
```

```python
@dataclass
class QcRow:
    file: str
    set: str
    prompt: str
    speaker: str
    verdict: str
    reason: str
    transcript: str
    match_score: float
    rms_dbfs: float
    peak_dbfs: float
    dur_ms: int
    device_intent: str = ""   # verbatim from the device; NEVER used as a label
    agrees: str = ""          # "1"/"0" device vs Whisper, "" when the take didn't parse
```

In `read_sessions`, inside the `Take(...)` construction:

```python
                    device_intent=r.get("device_intent") or "",
                    device_words=r.get("device_words") or "",
```

(`csv.DictReader` yields `None` for a column a nine-column guided row does not reach, hence the `or ""`.)

In `judge`, in the `QcRow(...)` construction:

```python
        dur_ms=m.get("dur_ms", 0),
        device_intent=take.device_intent,
```

- [ ] **Step 8: Make `_slug_of` handle a bare `<boot-ms>.wav` name**

```python
def _slug_of(path: Path) -> str:
    # "hallo-welt_001.wav" -> "hallo-welt"; a field take's "123456.wav" -> "123456"
    return re.sub(r"_\d{3}\.wav$", "", path.name).removesuffix(".wav")
```

- [ ] **Step 9: Add the field branch to `run_qc`**

At the top of `run_qc`'s body, next to the existing local imports:

```python
    from kws_de.eval import intent_text
    from kws_de.grammar import Intent
```

Change the counter initialisation line to:

```python
    n_words = n_skipped = n_wake = 0
    n_field = n_field_wake = n_field_parsable = n_field_agree = 0
```

Insert this block immediately after `if row.verdict != "approve": continue` and before `if t.set == "words":`:

```python
        if t.set == "field":
            n_field += 1
            cut_s, tokens = field_wake_split(tr)
            if cut_s is not None:
                # the wake phrase is a real "Hey Bus" positive: file it exactly
                # where the guided wake set goes, so it trains the wake model too
                sig, sr = sf.read(t.file, dtype="float32", always_2d=True)
                d = approved / "wake" / t.speaker
                dst = d / f"{t.speaker}_{_next_no(d, t.speaker)}.wav"
                dst.parent.mkdir(parents=True, exist_ok=True)
                sf.write(dst, sig[: int(cut_s * sr), 0], sr, subtype="PCM_16")
                written.append(str(dst.relative_to(approved)))
                _append_index(
                    approved / "wake" / "index.csv",
                    {
                        "file": str(dst.relative_to(approved)),
                        "prompt": config.WAKE_WORD,
                        "speaker": t.speaker,
                    },
                )
                n_wake += 1
                n_field_wake += 1
            got = field_intent(tokens)
            if isinstance(got, Intent):
                n_field_parsable += 1
                # Provenance only: the device's own words go through the SAME
                # grammar, and the two Intents are compared. The device never
                # supplies the label — `got` does.
                row.agrees = "1" if field_intent(normalise(t.device_intent)) == got else "0"
                n_field_agree += row.agrees == "1"
                # From here the take IS an approved sentence take with a derived
                # prompt: same phrase copy, same index row, same word segmentation.
                t = Take(
                    file=t.file, set="sentences", prompt=intent_text(got), speaker=t.speaker
                )
            else:
                # Kept, not dropped: unparsable field speech is `_unknown_`
                # material, with the transcript itself as its prompt.
                t = Take(
                    file=t.file, set="negatives", prompt=row.transcript, speaker=t.speaker
                )
```

Add the counts to the return dict:

```python
        "wake_written": n_wake,
        "field_takes": n_field,
        "field_parsable": n_field_parsable,
        "field_agree": n_field_agree,
```

- [ ] **Step 10: Add the report's Field section**

In `run_qc`, just before the `report.md` write, build the section:

```python
    if n_field:
        agree_rate = f"{n_field_agree / n_field_parsable:.3f}" if n_field_parsable else "n/a"
        field_section = (
            f"\n## Field\n\n{n_field} field takes, {n_field_parsable} parsable, "
            f"{n_field_wake} wake clips, device-Whisper agreement {agree_rate}.\n"
        )
    else:
        field_section = ""
```

and append `+ field_section` to the `(qc_dir / "report.md").write_text(...)` expression, after the segmentation-gaps part.

- [ ] **Step 11: Run the QC tests**

Run: `uv run pytest tests/test_qc.py -v`
Expected: PASS — all tests, including the four new field ones and the two updated `counts ==` assertions.

- [ ] **Step 12: Run the full suite and the linters**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: all tests pass; `All checks passed!`; formatter reports no files would be reformatted.

- [ ] **Step 13: Commit**

```bash
git add kws_de/qc.py tests/test_qc.py
git commit -m "$(cat <<'EOF'
feat(qc): field mode — wake split, grammar labels, device agreement

A field take is transcribed whole; a wake phrase in the first 1.3 s is cut off
into approved/wake/, and the rest is run through the same kws_de.grammar the
device uses. A parsable command becomes an approved phrase with the derived
prompt (and its word clips); anything else is kept as negative/_unknown_
material with the transcript as its prompt. qc.csv gains device_intent and
agrees, and report.md a Field section — the device's prediction is scored,
never used as a label.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016YVjMuh5AT7hGvYf4EtfUM
EOF
)"
```

---

### Task 5: Eval — the Field section of `kws-eval --recordings`

**Files:**

- Modify: `kws_de/eval.py` (`eval_recordings`, `render_recordings_section`, `main`, two new functions)
- Test: `tests/test_eval_recordings.py`

**Interfaces:**

- Consumes: Task 4's `qc.csv` columns `set`, `speaker`, `device_intent`, `agrees`.
- Produces:
  - `field_figures(qc_root) -> dict` with keys `"per_speaker"` (`{spk: {"takes": int, "parsable": int, "agree": int}}`), `"takes"`, `"parsable"`, `"agree"`
  - `render_field_section(field: dict) -> str`
  - `eval_recordings(approved, predict_fn, *, step_ms=100, manifest_path=None, qc_root=None)` — the result dict gains `"field"`
  - CLI: `kws-eval --recordings <dir> [--qc <dir>]`, default `data/recordings/qc`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval_recordings.py`:

```python
def _qc_root(tmp_path):
    """Two stamps' qc.csv: spk02 with 3 field takes (2 parsable, 1 agreeing),
    spk03 with 1 parsable and agreeing, plus a guided row that must be ignored."""
    root = tmp_path / "qc"
    cols = (
        "file,set,prompt,speaker,verdict,reason,transcript,match_score,"
        "rms_dbfs,peak_dbfs,dur_ms,device_intent,agrees\n"
    )
    (root / "s1").mkdir(parents=True)
    (root / "s1" / "qc.csv").write_text(
        cols
        + "a.wav,field,Licht an,spk02,approve,,Hey Bus Licht an,1.0,-20,-6,3500,Licht an,1\n"
        + "b.wav,field,Licht an,spk02,approve,,Hey Bus Licht aus,1.0,-20,-6,3500,Licht aus,0\n"
        + "c.wav,field,,spk02,approve,,wann fahren wir los,1.0,-20,-6,3500,,\n"
        + "d.wav,words,Licht,spk02,approve,,Licht,1.0,-20,-6,800,,\n"
    )
    (root / "s2").mkdir(parents=True)
    (root / "s2" / "qc.csv").write_text(
        cols + "e.wav,field,Heizung wärmer,spk03,approve,,Hey Bus Heizung wärmer,1.0,-20,-6,3500,Heizung wärmer,1\n"
    )
    return root


def test_field_figures_count_takes_parsable_and_agreement(tmp_path):
    fig = ev.field_figures(_qc_root(tmp_path))
    assert fig["takes"] == 4 and fig["parsable"] == 3 and fig["agree"] == 2
    assert fig["per_speaker"]["spk02"] == {"takes": 3, "parsable": 2, "agree": 1}
    assert fig["per_speaker"]["spk03"] == {"takes": 1, "parsable": 1, "agree": 1}


def test_recordings_section_carries_the_field_table(tmp_path):
    root, predict_fn = _build_approved(tmp_path)
    res = ev.eval_recordings(root, predict_fn, qc_root=_qc_root(tmp_path))
    md = ev.render_recordings_section(res)
    assert "## Field" in md
    assert "4 field takes, 3 parsable" in md
    assert "| spk02 | 3 | 2 | 0.500 |" in md
    assert "| spk03 | 1 | 1 | 1.000 |" in md
    assert "AT CAPTURE TIME" in md  # says whose accuracy this is


def test_recordings_section_has_no_field_table_without_field_takes(tmp_path):
    root, predict_fn = _build_approved(tmp_path)
    res = ev.eval_recordings(root, predict_fn)
    assert res["field"]["takes"] == 0
    assert "## Field" not in ev.render_recordings_section(res)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_eval_recordings.py -k field -v`
Expected: FAIL with `AttributeError: module 'kws_de.eval' has no attribute 'field_figures'`

- [ ] **Step 3: Add the two functions**

In `kws_de/eval.py`, after `_trained_speakers`:

```python
def field_figures(qc_root) -> dict:
    """Per-speaker field-capture figures, read from every `qc/<stamp>/qc.csv`:
    how many field takes there were, how many produced a parsable command, and
    how often the DEVICE's own intent agreed with the Whisper-derived one.

    That agreement is the field accuracy of the model deployed AT CAPTURE TIME
    — it does not measure the model this report evaluates, and the device's
    intent is never used as a label (`agrees` is written by `kws_de.qc.run_qc`,
    which labels from Whisper + the grammar). An empty `agrees` means the take
    did not parse: it is counted in `takes`, not in `parsable`."""
    import csv
    from collections import defaultdict
    from pathlib import Path

    per_spk: dict = defaultdict(lambda: {"takes": 0, "parsable": 0, "agree": 0})
    for path in sorted(Path(qc_root).glob("*/qc.csv")):
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("set") != "field":
                    continue
                s = per_spk[r["speaker"]]
                s["takes"] += 1
                if r.get("agrees"):
                    s["parsable"] += 1
                    s["agree"] += r["agrees"] == "1"
    return {
        "per_speaker": dict(per_spk),
        "takes": sum(s["takes"] for s in per_spk.values()),
        "parsable": sum(s["parsable"] for s in per_spk.values()),
        "agree": sum(s["agree"] for s in per_spk.values()),
    }


def render_field_section(field: dict) -> str:
    """Markdown for the field figures: per speaker, how many real interactions
    were captured, how many parsed, and the device-vs-Whisper agreement."""
    out = ["\n## Field\n"]
    out.append(
        f"{field['takes']} field takes, {field['parsable']} parsable. Agreement is "
        "the device's own intent vs the Whisper-derived one AT CAPTURE TIME — the "
        "accuracy of whatever model was deployed then, not of the model measured "
        "above. The field-derived phrases, words and negatives themselves are "
        "counted in the two figures above, under the same in-training / held-out "
        "labels as guided takes.\n"
    )
    out.append(
        "| speaker | field takes | parsable | device-Whisper agreement |\n|---|---|---|---|"
    )
    for spk in sorted(field["per_speaker"]):
        s = field["per_speaker"][spk]
        agree = s["agree"] / s["parsable"] if s["parsable"] else float("nan")
        out.append(f"| {spk} | {s['takes']} | {s['parsable']} | {agree:.3f} |")
    return "\n".join(out) + "\n"
```

- [ ] **Step 4: Wire them into the result and the section**

Change `eval_recordings`'s signature and add the figure to its result:

```python
def eval_recordings(
    approved, predict_fn, *, step_ms: int = 100, manifest_path=None, qc_root=None
) -> dict:
```

and in its return dict, before the closing brace:

```python
        "field": (
            field_figures(qc_root)
            if qc_root is not None
            else {"per_speaker": {}, "takes": 0, "parsable": 0, "agree": 0}
        ),
```

In `render_recordings_section`, directly before the closing `return re.sub(...)`:

```python
    if res.get("field", {}).get("takes"):
        out.append(render_field_section(res["field"]))
```

- [ ] **Step 5: Add the `--qc` flag**

In `main()`, next to the other `--recordings` arguments:

```python
    ap.add_argument(
        "--qc",
        default=None,
        help="QC output root (default data/recordings/qc) -> adds the Field section",
    )
```

and in the `if args.recordings:` branch, replace the `eval_recordings` call with:

```python
        qc_root = Path(args.qc) if args.qc else config.DATA_DIR / "recordings" / "qc"
        res = eval_recordings(
            approved_dir,
            predict_fn,
            manifest_path=manifest_path,
            qc_root=qc_root if qc_root.is_dir() else None,
        )
```

- [ ] **Step 6: Run the eval tests**

Run: `uv run pytest tests/test_eval_recordings.py -v`
Expected: PASS — all tests, including the three new ones.

- [ ] **Step 7: Run the full suite and the linters**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: all tests pass; `All checks passed!`; no reformatting.

- [ ] **Step 8: Commit**

```bash
git add kws_de/eval.py tests/test_eval_recordings.py
git commit -m "$(cat <<'EOF'
feat(eval): Field section in kws-eval --recordings

Per speaker: field takes, share parsable, and device-vs-Whisper agreement read
from qc.csv — the field accuracy of the model deployed at capture time, stated
as such and kept apart from the figures for the model being evaluated. The
field-derived clips themselves already appear in the in-training / held-out
figures, since QC files them through the same approved paths.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016YVjMuh5AT7hGvYf4EtfUM
EOF
)"
```

---

### Task 6: Docs — requirements, tests, prose, paper notes

**Files:**

- Modify: `docs/sphinx/requirements.rst` (two new `req` directives)
- Modify: `docs/sphinx/tests.rst` (four new `test` directives)
- Modify: `docs/sphinx/firmware.rst` ("Modes" → the Assistent entry) and `docs/sphinx/pipeline.rst` ("Quality control rules")
- Modify: `firmware/README.md` ("Manual test checklist")
- Modify: `docs/paper-notes.md`

**Interfaces:**

- Consumes: every name Tasks 1–5 produced. No code changes here.
- Produces: `REQ_FW_FIELD_CAPTURE`, `REQ_PIPE_FIELD_LABELS`, `TEST_FIELD_SPAN`, `TEST_QC_FIELD_MODE`, `TEST_EVAL_FIELD`, `TEST_MANUAL_FIELD_CAPTURE`.

- [ ] **Step 1: Add the two requirements**

In `docs/sphinx/requirements.rst`, after the `REQ_FW_ASSIST_GATE` block:

```rst
.. req:: Field capture is opt-in and never writes inside the window
   :id: REQ_FW_FIELD_CAPTURE
   :status: implemented

   In ``UI_MODE_ASSIST``, with capture switched **on**, every wake fire
   arms one *field take*: the 1.0 s in front of the fire plus the 2.5 s
   assist window (``FIELD_TAKE_MS`` = 3500), copied out of the audio ring
   and saved as ``storage_root()/field/<spkNN>/<boot-ms>.wav`` (16 kHz mono
   int16) with one row in ``storage_root()/field/<spkNN>/field.csv``
   (``file,fire_ms,wake_prob,device_intent,device_words,window_ms,ms,peak_dbfs``).
   The speaker id is the current NVS id and is never bumped per
   interaction, so one boot of one user is one field directory.

   Capture is **opt-in**: the toggle is off until the user turns it on once
   on the Assistent screen ("Aufnahme"), is persisted in NVS under the
   existing ``kws`` namespace and restored at boot, is settable over the
   serial console (``field on|off``), and while on the screen carries a
   "REC" badge — the only visible difference.

   **No file I/O happens while the recogniser is active.** The wake task
   only records the fire's ring position (``field.c``, pure C, host-tested);
   the copy and the FAT write happen on the record task, which is idle in
   Assistent mode, *after* the window has closed and the recogniser has
   been switched off. A FAT write costs 100-300 ms, which is more than a
   whole recognise step. Below ``STORAGE_MIN_FREE_BYTES`` the take is
   dropped with one log line (``field: dropped, storage low``) and a
   counter reported by ``status``.

   The audio ring must still hold the take when the copy starts; a
   ``_Static_assert`` in ``field.h`` checks ``AUDIO_RING_SAMPLES`` against
   the take plus 0.2 s of copy latency (the ring is 10 s today, so this
   guards a future shrink).

.. req:: Field takes are labelled by Whisper and the grammar, never by the device
   :id: REQ_PIPE_FIELD_LABELS
   :status: implemented

   A pulled field take (``set=field`` in ``sessions.csv``, with the device
   columns ``fire_ms,wake_prob,device_intent,device_words`` appended) is
   transcribed whole by the same Whisper model, prompt and padding as every
   other take. If the transcript's first two words match the wake regex
   ``(hey|hej|he|hei)(bus|buss|bos|boss)`` and Whisper's word timestamps put
   them inside the first 1.3 s, ``[0, end of "bus" + 0.15 s]`` is cut as a
   ``wake`` clip into ``approved/wake/<spkNN>/``. The remaining words are
   normalised, filtered to the command vocabulary and run through
   ``kws_de.grammar.parse`` — the same grammar the device uses:

   - a valid intent -> the label is ``kws_de.eval.intent_text(intent)`` and
     the take is filed exactly like an approved guided *sentence*
     (``approved/phrases/``, index row, word segmentation into
     ``approved/words/``);
   - no valid intent -> ``approved/negatives/<spkNN>/`` with the transcript
     itself in the index row's ``prompt`` column, so it still feeds
     ``_unknown_`` windows;
   - an empty transcript, or a failed audio gate, rejects the take as
     today.

   **Everything else is kept** — a field take is real usage, and speech the
   grammar cannot parse is exactly the negative material the model needs.

   The device's own intent is carried through to ``qc.csv`` as
   ``device_intent``, and ``agrees`` records whether it equals the
   Whisper-derived intent (``1``/``0``, empty when the take did not parse).
   That column is **provenance, never a label**: it is what
   ``kws-eval --recordings``'s Field section reports as the field accuracy
   of the model that was deployed at capture time, which is a different
   model from the one the rest of the report measures.
```

- [ ] **Step 2: Add the four test entries**

In `docs/sphinx/tests.rst`, add the host test after the `TEST_STREAM_DETECTOR` entry:

```rst
.. test:: Field-take window arithmetic and the capture toggle
   :id: TEST_FIELD_SPAN
   :status: passing
   :links: REQ_FW_FIELD_CAPTURE, REQ_FW_HOST_TESTS_NO_IDF

   ``firmware/test/test_field.c``: asserts that capture is off after
   ``field_reset`` (opt-in), that an enabled fire yields the pre-roll +
   window span ending exactly at the window's close, that a second fire
   inside an open window keeps the first fire's position (one interaction,
   one take), that ``field_disarm`` prevents a second copy, that a fire in
   the first second after boot shortens the take instead of reading in
   front of the ring, and that turning capture off drops a pending take.
```

and the two Python tests plus the manual check with the other Python/manual entries:

```rst
.. test:: QC field mode: wake split, grammar labels, agreement
   :id: TEST_QC_FIELD_MODE
   :status: passing
   :links: REQ_PIPE_FIELD_LABELS, REQ_PIPE_QC_CONTENT, REQ_PIPE_APPROVED_LAYOUT

   ``tests/test_qc.py``: ``field_wake_split`` cuts at the end of "Bus"
   + 0.15 s and returns the command tokens, and ignores a wake phrase that
   lands after 1.3 s or is absent; ``field_intent`` drops filler and returns
   the same ``Intent`` the device's grammar would; ``run_qc`` on a field
   session writes the wake clip, files the command as an approved phrase
   with the grammar-derived prompt (plus its word clips), records
   ``device_intent``/``agrees``, keeps an unparsable take as a negative with
   the transcript as its prompt, and rejects an empty transcript
   (``empty_transcript``). A disagreeing device intent changes ``agrees``
   and nothing else — the label still comes from Whisper.

.. test:: Field section of the recordings eval
   :id: TEST_EVAL_FIELD
   :status: passing
   :links: REQ_PIPE_FIELD_LABELS, REQ_PIPE_EVAL_LABELS

   ``tests/test_eval_recordings.py``: ``field_figures`` counts takes,
   parsable takes and agreements per speaker across several ``qc/<stamp>/``
   directories and ignores non-field rows; the rendered section carries the
   per-speaker table and says the agreement is the accuracy of the model
   deployed *at capture time*; with no field takes there is no Field
   section at all.

.. test:: On-device field capture in Assistent mode
   :id: TEST_MANUAL_FIELD_CAPTURE
   :status: manual
   :links: REQ_FW_FIELD_CAPTURE, REQ_FW_ASSIST_GATE, REQ_FW_STORAGE_MIN_FREE

   ``firmware/README.md`` "Manual test checklist": in Assistent mode the
   "Aufnahme" switch is off after a fresh flash and ``status`` reports
   ``field off takes 0 dropped 0``; turning it on shows the "REC" badge and
   survives a power cycle; one interaction ("Hey Bus" + a command) adds
   exactly one ``field/<spkNN>/<boot-ms>.wav`` and one ``field.csv`` row
   visible over USB-drive mode, and ``status`` counts it; the recognise
   step time logged inside the window shows no 100-300 ms outlier, i.e. no
   file write happened while the recogniser ran. Run by hand on real
   M5Stack CoreS3 hardware; not automated.
```

- [ ] **Step 3: Extend the firmware and pipeline prose**

In `docs/sphinx/firmware.rst`, in the "Modes" section's Assistent entry, append:

```rst
An **"Aufnahme"** switch on this screen turns *field capture* on
(:need:`REQ_FW_FIELD_CAPTURE`). It is off until switched on once, is
remembered across reboots, and while on the screen carries a small red
"REC" badge. With it on, every interaction — the second before the wake
fire plus the 2.5 s window — is saved to
``storage_root()/field/<spkNN>/`` together with what the device itself
recognised, *after* the window closes: the recogniser is already off when
the file is written, so the capture cannot slow a recognise step. The
serial console can set the same switch (``field on|off``) and ``status``
reports it along with the number of takes saved and dropped.
```

In `docs/sphinx/pipeline.rst`, at the end of "Quality control rules", append:

```rst
A **field take** (``set=field``, captured in Assistent mode) takes a
different route through the same rules
(:need:`REQ_PIPE_FIELD_LABELS`). It has no prompt to match against, so the
content gate approves anything that transcribed to something. The
transcript is then split: a "Hey Bus" in the first 1.3 s becomes a ``wake``
clip, and the rest is run through the *same* ``kws_de.grammar.parse`` the
device runs. A valid intent becomes the label and the take is filed as an
approved phrase — prompt, index row and word segmentation exactly as for a
guided sentence; anything else is kept as a negative with the transcript as
its prompt. Nothing is thrown away for failing to parse: unparsable real
speech is precisely the ``_unknown_`` material the model needs.

The device's own intent travels with the take into ``qc.csv``
(``device_intent``) next to an ``agrees`` flag, and
``kws-eval --recordings`` reports it as a separate **Field** section. It is
the accuracy of the model that was *deployed* when the recording happened —
a different model from the one being evaluated — and it is never used as a
label.
```

- [ ] **Step 4: Add the manual checklist entry**

In `firmware/README.md`, under "Manual test checklist", add:

```markdown
- **Field capture (Assistent mode).** After a fresh flash, open Assistent: the
  "Aufnahme" switch is off and no "REC" badge is shown; `status` over the
  console answers `field off takes 0 dropped 0`. Turn the switch on → the badge
  appears; power-cycle the device and re-open Assistent → the switch is still
  on. Say "Hey Bus" and then a command: the screen behaves exactly as before
  (green flash, beep, recognised word). `status` now reports `field on takes 1
  dropped 0`, and in USB-drive mode the drive holds
  `field/<spkNN>/<boot-ms>.wav` plus a `field.csv` row for it. In the serial
  log, the `recognise` step times inside the window stay in their usual range —
  a 100–300 ms outlier would mean a file was written while the recogniser ran.
```

- [ ] **Step 5: Write the paper-notes entry with the numbers from the run**

Append to `docs/paper-notes.md`, before the "## Open questions" section. Fill each italic slot from the run named next to it — this entry goes in **after** the first real field session, so every number is measured, not projected:

- `<takes>`, `<parsable>`, `<agreement>`: the "Field" line of `data/recordings/qc/<stamp>/report.md` from `kws-qc` on that session.
- `<wake clips>`: the same line.
- `<step range>`: the `recognise` `step <N> ms` lines inside the windows of the on-device check (Task 2, Step 9), as min–max.

```markdown
## Field capture: real interactions as training data (2026-09-03)

Assistent mode now optionally keeps what it hears. With the "Aufnahme" switch on,
every wake fire arms one *field take* — the second before the fire plus the 2.5 s
command window — which the record task copies out of the always-on audio ring and
writes **after** the window has closed, with the recogniser already switched off.
That ordering is the whole design: a FAT write on this device costs 100–300 ms,
more than a full recognise step, so capturing during the window would have changed
the very behaviour being captured. Measured on the device, the recognise step time
inside a capturing window stayed at <step range> ms — unchanged from a
non-capturing window.

The label never comes from the device. On the workstation the take is transcribed
by the same Whisper model as every guided take; a "Hey Bus" in the first 1.3 s is
cut off as a wake positive, and the remaining words are run through the *same*
`kws_de.grammar.parse` the firmware's vocabulary feeds. A valid intent becomes the
phrase label, and the take joins training exactly like a guided sentence take
(phrase clip, index row, per-word segmentation); anything that does not parse is
kept as `_unknown_` material rather than discarded — real speech that the grammar
rejects is the negative data this model is chronically short of.

What the device *did* recognise is kept beside the label and scored against it.
That gives a figure no synthetic evaluation can: **field accuracy**, the deployed
model's agreement with the transcript on real, unprompted interactions in the van.
First session: <takes> field takes, <parsable> parsable, <wake clips> wake clips,
device–Whisper agreement <agreement>.

Two honest caveats. The agreement figure belongs to whichever model was flashed at
capture time, not to the model a later report evaluates — `kws-eval --recordings`
prints it in its own **Field** section for that reason. And a field take is
self-selected: it exists only because the wake word fired, so it measures command
accuracy *given* a successful wake, and says nothing about missed wakes, for which
no trigger exists.
```

- [ ] **Step 6: Build the docs and check the requirement/test links resolve**

Run: `bash docs/sphinx/build.sh`
Expected: `build succeeded` with no `undefined label` / `need ... not found` warnings for `REQ_FW_FIELD_CAPTURE`, `REQ_PIPE_FIELD_LABELS` or the four new test ids.

- [ ] **Step 7: Lint the markdown that is not excluded**

Run: `npx markdownlint-cli@0.42.0 -c .markdownlint.json docs/paper-notes.md firmware/README.md`
Expected: no output (clean)

- [ ] **Step 8: Commit**

```bash
git add docs/sphinx/requirements.rst docs/sphinx/tests.rst docs/sphinx/firmware.rst \
        docs/sphinx/pipeline.rst firmware/README.md docs/paper-notes.md
git commit -m "$(cat <<'EOF'
docs: field capture requirements, tests, prose and first numbers

REQ_FW_FIELD_CAPTURE (opt-in, no I/O during the window, storage floor) and
REQ_PIPE_FIELD_LABELS (Whisper + grammar labels, everything kept, the device
prediction as provenance only), the four tests that cover them, the firmware and
pipeline prose, the manual checklist entry, and the paper-notes log entry with
the first real field session's numbers.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016YVjMuh5AT7hGvYf4EtfUM
EOF
)"
```

---

## Self-Review

### 1. Spec coverage

| Spec requirement | Task | Where |
|---|---|---|
| §2 Toggle: "Aufnahme" on the Assistent screen, off at boot, NVS `kws` namespace, restored at boot | 2 | Steps 4–5 (`wake_field_set/get`, `nvs_get_u8`, `ui_show_assist`) |
| §2 Toggle: "REC" badge while on | 2 | Step 5 (`l_rec`) |
| §2 Toggle: console `field on\|off`, `status` reports it | 2 | Step 6 |
| §2 Capture = copy after the window, never during it | 2 | Steps 3–4 (`post_field_take` on the closing edge; `save_field_take` on the record task) |
| §2 `fire_pos` kept; `start = fire_pos − 1.0 s`, `len = 3.5 s` | 1 | `field_on_wake`, `field_take_span`, `FIELD_TAKE_SAMPLES` |
| §2 Payload `{start, len, prob, device_intent, device_words}` posted as `REC_CMD_FIELD_TAKE` | 1, 2 | `field_take_t`; `record_post_field_take` |
| §2 Ring length checked against 4.0 s at compile time | 1 | `_Static_assert` in `field.h` (3.5 s take + 0.2 s latency; ring is 10 s, no raise needed — see Context) |
| §2 Pre-roll and window as single constants next to it | 1 | `FIELD_PREROLL_MS`, `ASSIST_WINDOW_MS` reused, `FIELD_TAKE_MS` |
| §2 Layout `storage_root()/field/spkNN/<boot-ms>.wav`, speaker id not bumped | 2 | Steps 1, 3 |
| §2 `field.csv` columns | 2 | Step 3 (header string, verbatim) |
| §2 `device_intent` / `device_words` semantics | 2 | Step 2 (`window_intent` / `window_words`) — see the deviation note below |
| §2 Storage floor: drop + `field: dropped, storage low` + counter in `status` | 2 | Steps 3, 6 |
| §2 Screens otherwise unchanged | 2 | Step 5 adds only the badge and the switch |
| §3 `pull-recordings.sh` pulls `field/`; each row becomes a `sessions.csv` row with `set=field`, `prompt=""`, device columns appended | 3 | Step 3 |
| §3 `ingest.sh` needs no change beyond the pass-through and the count verification | 3 | Step 5 asserts it (`ingested 2 takes`) |
| §4.1 Transcribe the whole take, same model/prompt/padding | 4 | unchanged `whisper_transcriber`; the field branch only consumes `tr` |
| §4.2 Wake part: regex, inside 1.3 s, cut `[0, end of "bus" + 0.15]`, next-free numbering, `written.txt` | 4 | `field_wake_split`, Step 9's wake block (`_next_no`, `written.append`) |
| §4.3 Command part: parses -> `intent_text` label, filed as a sentence + segmentation | 4 | Step 9's `Intent` branch |
| §4.3 Does not parse -> negatives with the transcript as `prompt` | 4 | Step 9's `else` branch |
| §4.3 Empty transcript / audio-gate failure -> rejected as today | 4 | `content_gate` field branch (`empty_transcript`); `audio_gate` unchanged |
| §4.4 `qc.csv` gains `device_intent` and `agrees` | 4 | Steps 7, 9 |
| §4.4 `report.md` gets a Field section: takes, parsable, wake clips, agreement | 4 | Step 10 |
| §5 `kws-eval --recordings` Field section: per speaker takes, share parsable, agreement | 5 | `field_figures`, `render_field_section` |
| §5 The agreement is independent of the model being evaluated | 5 | Stated in the docstring and in the rendered text; asserted by `test_recordings_section_carries_the_field_table` |
| §5 The usual recordings figures for field-derived clips under the manifest labels | 5 | Free: QC files field takes through the existing `approved/` paths, so `eval_recordings` already sees them; stated in the section text |
| §6 Files table | 1–6 | The File Structure table covers every row (`ui_assist.c` is the assist screen file) |
| §7 Order: firmware + host tests + on-device check; ingest + QC; eval + docs | 1–6 | Task order |
| §7 On-device check: toggle persists, take appears, no change in step time | 2 | Step 9's three numbered checks |

**Deviation, called out rather than papered over:** the spec's §2 says `device_intent` is "the intent the on-device grammar composed". There is no grammar on the device (see Context), so Task 2 writes the ordered fired command words and Task 4 runs *both* sides through `kws_de.grammar.parse` before comparing. The comparison is `Intent == Intent`, which is the answer a C grammar port would have produced, at no cost in firmware. If a device-side grammar ever lands, `device_intent` becomes its output and Task 4's `field_intent(normalise(t.device_intent))` still parses it correctly.

**Deliberately not built (spec non-goals, restated so nobody adds them):** capturing missed wakes, any cloud component, model changes. Also not built: a queue of pending field takes (one slot, with the reasoning as a `ponytail:` comment in `record.c`), and a C port of the grammar.

### 2. Placeholder scan

Searched the plan for `TBD`, `TODO`, `implement later`, `fill in details`, `add appropriate error handling`, `add validation`, `handle edge cases`, `write tests for the above`, `similar to Task`. None present. Every code step carries the actual code; every test step carries the actual test body; every run step names the command and the expected output. The only non-literal values are the five measured numbers in Task 6 Step 5 (`<takes>`, `<parsable>`, `<wake clips>`, `<agreement>`, `<step range>`), which cannot exist before the run — the step names the exact command and output line each one comes from, and the entry is written after that run, per the paper-notes rule "real numbers only".

### 3. Type consistency

- `field_state_t` / `field_take_t` field names are identical in Task 1's header, Task 1's test, and Tasks 2's `post_field_take` / `save_field_take` (`start`, `len`, `fire_ms`, `wake_prob`, `intent`, `words`).
- `FIELD_PREROLL_SAMPLES` / `FIELD_WINDOW_SAMPLES` / `FIELD_TAKE_SAMPLES` are used with exactly these names in `field.h`, `field.c` and `test_field.c`; the test's `start + len == fire + FIELD_WINDOW_SAMPLES` holds for both branches of `field_take_span`.
- `recognise_status_t.window_intent` / `window_words` (Task 2 Step 2) are read under those names in `post_field_take` (Step 4) and are the source of `field_take_t.intent` / `.words`.
- `record_status_t.field_takes` / `field_dropped` (Task 2 Step 3) are the names `console.c` prints (Step 6).
- The `field.csv` column order in `record.c`'s `fputs` header and `fprintf` row (Task 2 Step 3) matches the `$1..$8` positions the awk mapping in Task 3 Step 3 assumes, and matches the fixture in Task 3 Step 1.
- The `sessions.csv` 13-column header is identical in Task 3's script change, both Task 3 test fixtures and Task 4's `_field_session` fixture.
- `QcRow`'s new fields `device_intent` / `agrees` (Task 4 Step 7) are the exact keys Task 5's `field_figures` reads (`r.get("agrees")`, `r["speaker"]`, `r.get("set") == "field"`), and the exact column names Task 4's tests assert on.
- `run_qc`'s new return keys `field_takes` / `field_parsable` / `field_agree` are the same in the return dict, the Field-section computation, and both updated/new tests.
- `field_figures`'s result keys (`per_speaker`, `takes`, `parsable`, `agree`) are the ones `render_field_section` indexes and the ones `render_recordings_section`'s guard (`res.get("field", {}).get("takes")`) checks.
- `eval_recordings`'s new keyword is `qc_root` in the signature, in `main()`'s call, and in all three Task 5 tests.
