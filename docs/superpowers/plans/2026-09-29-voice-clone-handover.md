# Voice-clone experiment — handover (2026-09-29, branch `exp/voice-clone`)

Spec: `docs/superpowers/specs/2026-09-29-voice-clone-tts-design.md`. Spike results: paper E63.
This branch holds everything the experiment needs; nothing is merged yet (spike code is kept
only if the clone arm wins).

## State on the GPU box (Linux, RTX 3090 Ti)

- Data root `~/kws-data` (`~/.config/kws-de/config.toml` wires it; `kws-doctor` confirms).
- Experiment worktree `~/src/kws-de-main` (detached; `git fetch && git checkout exp/voice-clone`).
  The owner's checkout `~/src/kws-de` is theirs — never switch branches there.
- XTTS venv `~/xtts-spike/.venv` (coqui-tts 0.27 + `transformers<5` + torch CUDA 13). Keep it
  separate from the TF venv. Full synthesis output: `~/xtts-spike/out/full/` (372 clips, manifest,
  `tts_check.csv`); earlier variants `out/smoke,B,C,D,E,F,F2,G`.
- **Strict-passing command-word clones already materialised:**
  `~/kws-data/data/recordings/clone/words/<word>/spkNN_NNN.wav` — 139 clips, 21 words × spk01/spk02/spk22
  (zero for Küche/spk01, fünfzig/spk01, fünfzig/spk02, kälter/spk22). Move this dir aside to build
  the control arm.
- The GPU is shared with an LLM server (ollama) that holds ~21.5 GB when running — ask the owner
  to pause it for GPU work; never kill it.
- `kws-tts-check` on CUDA needs the venv's NVIDIA libs visible:
  `LD_LIBRARY_PATH=$(find ~/src/kws-de-main/.venv/lib/python3.11/site-packages/nvidia -maxdepth 2 -type d -name lib | tr '\n' ':')`
  (scope it to that command — it breaks the XTTS venv's torch). Without it the gate falls back to
  CPU (PR #114).

## Recipe that works (E63)

```bash
cd ~/xtts-spike && .venv/bin/python ~/src/kws-de-main/scripts/xtts_clone.py synth out/full --all --trim --takes 4
cd ~/src/kws-de-main && LD_LIBRARY_PATH=… uv run --no-sync kws-tts-check ~/xtts-spike/out/full
~/xtts-spike/.venv/bin/python scripts/xtts_clone.py score out/full      # strict transcript==text
~/xtts-spike/.venv/bin/python scripts/xtts_clone.py keep  out/full ~/kws-data/data/recordings/clone/words
```

Sampling knobs (temperature, repetition penalty, shorter conditioning) made it worse; the
first-utterance trim + strict gate + several takes is the recipe. Expect ~57 % of takes to pass.

## Next steps (the experiment, spec §Experiment)

1. `kws-doctor`; `nvidia-smi` (GPU must be free).
2. **Control arm:** `mv ~/kws-data/data/recordings/clone ~/kws-data/data/recordings/clone.off`;
   `uv run --no-sync kws-dataset build --prefix features_v3` → copy the npz set aside as
   `features_v3_control_*` (or build with `--prefix features_v3_control` if the build honours it
   end-to-end — check `kws-train --prefix`).
3. **Clone arm:** move the dir back; `kws-dataset build --prefix features_v3_clone`. The build
   log must show `[recordings] merged:` counts that include the clone clips (data.py `clone:` tree)
   and they must all land in train (`force_rec_to_train`).
4. Train both arms, seeds 0 and 1: `kws-train --v2 --width 48 --qat --qat-epochs 20 --real-weight 3 --epochs 40 --seed S --prefix <arm> --out <arm>_s<S>`.
   ~4 min each on the GPU; run them in tmux.
5. Score each model on the real scoreboard: `kws_de.eval.eval_recordings` `isolated` figure
   (guided-only `approved/words`), false accepts, and `scripts/recipe-grid.py`'s `passes()` rule
   (aggregate ≥ 0.785, 0 FA) against deployed `86b7105e`.
6. Pass = clone beats control in **both** seeds with 0 FA and beats `86b7105e` → open the PR to
   merge this branch (add the engine to `docs/dev-setup.md`). Fail → paper entry, drop the branch
   (keep only the paper text).
7. Paper: E64 with the four runs' numbers, either way.

## Not done / open

- Owner's ear check of the clones (6 samples were sent): if they do not sound like the speakers,
  the whole premise is off — ask before spending the training runs.
- Compounds/scene triggers were synthesised too (in `out/full`) but are not in the 23-class
  vocabulary; they matter only for the parked 26-class model (PR #99).
- F5-TTS fallback (spec) untested.
