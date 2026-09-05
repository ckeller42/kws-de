import importlib.util
import pathlib

_SPEC = importlib.util.spec_from_file_location(
    "drop_tts_clips", pathlib.Path(__file__).parent.parent / "scripts" / "drop-tts-clips.py"
)
drop_tts_clips_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(drop_tts_clips_mod)


def test_drops_legacy_and_explicit_say_keeps_other_engines_and_real_clips():
    clips = {
        "an": [
            (b"a", "tts:Anna:180"),  # legacy say format (no engine field): drop
            (b"b", "tts:say:Eddy"),  # explicit say engine: drop
            (b"c", "tts:piper:de_DE-thorsten-medium"),  # other engine: keep
            (b"d", "deadbeef"),  # MSWC real clip: keep
            (b"e", "rec:spk01"),  # device recording: keep
        ]
    }
    counts = drop_tts_clips_mod.drop_tts_clips(clips)
    assert counts == {"an": (5, 3)}
    kept_speakers = {s for _, s in clips["an"]}
    assert kept_speakers == {"tts:piper:de_DE-thorsten-medium", "deadbeef", "rec:spk01"}


def test_all_engines_drops_every_tts_clip():
    clips = {"zu": [(b"a", "tts:piper:x"), (b"b", "deadbeef")]}
    counts = drop_tts_clips_mod.drop_tts_clips(clips, all_engines=True)
    assert counts == {"zu": (2, 1)}
    assert [s for _, s in clips["zu"]] == ["deadbeef"]
