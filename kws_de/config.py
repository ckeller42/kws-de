import os
from pathlib import Path

SAMPLE_RATE = 16000
CLIP_MS = 1000
CLIP_SAMPLES = SAMPLE_RATE * CLIP_MS // 1000  # 16000
WIN_SAMPLES = 480  # 30 ms
HOP_SAMPLES = 320  # 20 ms
N_MELS = 40
N_MFCC = 10
N_FRAMES = (CLIP_SAMPLES - WIN_SAMPLES) // HOP_SAMPLES + 1  # 49

COMMANDS = ["Licht", "Kühlschrank", "Camping", "Heizung", "Wasser"]
LABELS = COMMANDS + ["_unknown_", "_silence_"]
NUM_CLASSES = len(LABELS)  # 7

# Training mini-batch. 128 (not 32) since E9: the nets are tiny (tens of k params on
# 49x10 MFCC), so per-step overhead dominated; 4x fewer steps, same Adam defaults.
BATCH_SIZE = 128

# On-device resource budgets (see spec Global Constraints).
MAX_MODEL_BYTES = 500_000
MAX_ARENA_BYTES = 300_000
MAX_MACS = 3_000_000
MAX_LATENCY_MS = 30

_REPO_ROOT = Path(__file__).resolve().parent.parent
# data/ and models/ are gitignored; their physical home is a local detail. Set
# KWS_DATA_ROOT to keep ONE root (<root>/data, <root>/models) outside every
# worktree — e.g. on an external SSD — shared by all checkouts and the ingest
# scripts; unset, they stay repo-relative. Never commit a machine path here.
_DATA_ROOT = Path(os.environ.get("KWS_DATA_ROOT", _REPO_ROOT))
DATA_DIR = _DATA_ROOT / "data"
MODELS_DIR = _DATA_ROOT / "models"


def label_index(label: str) -> int:
    return LABELS.index(label)


# --- v2: wake word + slot commands (additive; v1 constants above untouched) ---
WAKE_WORD = "Hey Bus"
WAKE_LABELS = ["wake", "_not_"]
DEVICES = ["Licht", "Kühlschrank", "Heizung", "Aufstelldach"]
ZONES = ["Küche", "Dach", "Außen", "Lesen"]  # light zones — apply to Licht only
# Light brightness levels (Licht only). Spoken German number words (single tokens for KWS);
# map to the app's 0-10 brightness scale: fünfundzwanzig≈3, fünfzig=5, fünfundsiebzig≈8, hundert=10.
LIGHT_LEVELS = ["fünfundzwanzig", "fünfzig", "fünfundsiebzig", "hundert"]
ACTIONS = [
    "an",
    "aus",
    "auf",
    "zu",
    "heller",
    "dunkler",
    "wärmer",
    "kälter",
    "leise",
    *LIGHT_LEVELS,
]
ZONED_DEVICES = ["Licht"]
# Per-device allowed actions — grounded in the real controllable functions.
DEVICE_ACTIONS = {
    "Licht": ["an", "aus", "heller", "dunkler", *LIGHT_LEVELS],
    "Kühlschrank": ["an", "aus", "leise"],
    "Heizung": ["an", "aus", "wärmer", "kälter"],
    "Aufstelldach": ["auf", "zu"],
}
COMMAND_LABELS = DEVICES + ZONES + ACTIONS + ["_unknown_", "_silence_"]


def command_index(label: str) -> int:
    return COMMAND_LABELS.index(label)


# Guided-recorder "negative" prompts: everyday German sentences that contain
# none of the command vocabulary. Used only for on-device recording; the
# recordings feed false-accept evaluation later.
NEGATIVE_PROMPTS = [
    "wie spät ist es",
    "wo sind wir gerade",
    "hast du den Schlüssel gesehen",
    "morgen wird es regnen",
    "ich habe Hunger",
    "wann fahren wir los",
    "das war ein schöner Tag",
    "kannst du mir helfen",
    "der Kaffee ist fertig",
    "wir brauchen noch Brot",
    "ich gehe kurz raus",
    "mach die Musik leiser",
    "wie weit ist es noch",
    "das Wetter ist super",
    "ich bin müde",
    "hast du gut geschlafen",
    "wir sind gleich da",
    "gib mir bitte das Handtuch",
    "die Kinder schlafen schon",
    "was gibt es heute zum Essen",
]
