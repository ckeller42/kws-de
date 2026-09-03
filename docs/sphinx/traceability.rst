Traceability
============

How the ESP32 code is unit-tested
----------------------------------

Three tiers, cheapest/most-automated first:

1. **Host ``cc`` tests** (``firmware/test/*.c``, ``make -C firmware/test``)
   — for pure logic with no ESP-IDF dependency: MFCC, the streaming
   detector, the WAV header writer, prompt shuffling, the energy VAD. Runs
   on Mac, Pi, and CI without a Docker/IDF toolchain.
2. **pytest** (``tests/*.py``) — for the Python side that generates or
   gates what reaches the device: header generators (``kws-fwgen``), model
   export and the model-health regression gate, the TFLM op-set gate, and
   the host-side ``pull-recordings.sh`` wrapper script. These are also the
   reference implementations the C ports (MFCC, ``KeywordStream``) are
   checked against.
3. **Docker/CI build + manual on-hardware checklist** — for code that only
   makes sense wired to real silicon: the audio codec ring buffer, FAT
   storage mount/unmount, USB MSC, the LVGL UI, and TFLM inference on an
   actual CoreS3. CI's ``build`` job (pinned IDF, ``idf.py build``) proves
   it *compiles and links*; ``firmware/README.md``'s manual checklist is
   the only thing that proves it *works*, run by hand before merging
   changes that touch hardware-coupled code.

**Future (not yet built):** on-target Unity tests (ESP-IDF's built-in test
framework) would let VAD/stream/mfcc-adjacent hardware interactions run on
the actual device in CI via a test runner + JTAG/serial harness, closing
the gap between tier 1 and tier 3. Out of scope for phase 1 (see the design
spec's "Out of scope / follow-ups").

Requirements
------------

.. needtable::
   :types: req
   :columns: id, title, status, incoming
   :style: table

Tests
-----

.. needtable::
   :types: test
   :columns: id, title, status, outgoing
   :style: table

Requirement → test coverage
-----------------------------

Every requirement, and which tests verify it (blank = gap):

.. needtable::
   :types: req
   :columns: id, incoming
   :style: table

Gaps: requirements with no linked test
-----------------------------------------

.. needtable::
   :types: req
   :filter: len(links_back) == 0
   :columns: id, title, status
   :style: table

As of this writing, four requirements are open gaps:

- ``REQ_FW_RECORD_CLIP_REJECT`` — clip-and-redo is implemented but neither
  a host test nor the manual checklist currently exercises it (the
  checklist doesn't force a clipped take). Add a manual step, or a host
  test against a synthetic full-scale PCM buffer if the clip check is
  ever factored out of ``record.c`` into something host-testable.
- ``REQ_FW_STORAGE_MIN_FREE`` — the flash-full path (< 200 KB free →
  REC_FULL) has no automated or manual coverage; it's awkward to hit by
  hand (needs a near-full ``/rec``) and isn't in the current checklist.
- ``REQ_FW_DATA_NOT_COMMITTED`` — enforced only by ``.gitignore`` and code
  review, not by a CI check (e.g. a ``git ls-files`` assertion that none
  of the ignored paths are tracked).
- ``REQ_FW_ASSIST_GATE`` — the gate's logic is fully covered on the host
  (``firmware/test/test_assist_gate.c``), but the wiring around it — that a
  wake fire reaches the gate and that the recogniser actually stops — is
  tier 3, evidenced by the ``KWS_DUTY`` and ``assist: recogniser on/off``
  log lines under a console-injected fire (``wakefire``).
- ``REQ_FW_ARENA_PLACEMENT`` — tier 3 by nature: which heap an arena lands
  in, and what it costs, only exist on real silicon. The boot log is the
  evidence (an ``INFO`` line naming internal RAM and the free size, or a
  ``WARN`` naming PSRAM); an on-target Unity test could assert it once that
  tier exists.
