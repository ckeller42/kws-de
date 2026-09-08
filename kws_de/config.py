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
# Raised from 3.0 M in E16: the width-48 command model spends 4,234,704 MACs, and the
# binding constraint is the 100 ms recognise step it has to fit in, not this figure.
MAX_MACS = 5_000_000
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
# Guided-recorder "Hey Bus"-only session: this many single-take reads, no
# doubled reads like the word/sentence/negative sets (real wake positives,
# not review pairs).
WAKE_PROMPT_REPEATS = 5
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
# Fused Licht+zone compounds a speaker naturally says instead of "Licht <zone>"
# (e.g. "Küchenlicht an" for "Licht Küche an"). Maps compound word -> the zone
# it stands for; kws_de.grammar.parse()/firmware/main/intent.c treat the
# compound as device=Licht + this zone in one token (see both for the wiring).
#
# NOT added to COMMAND_LABELS/LABELS: the trained command model only has
# KWS_MODEL_NUM_CLASSES=23 outputs (firmware/main/gen/model_config.h), and
# recognise.cc asserts KWS_NUM_LABELS == KWS_MODEL_NUM_CLASSES at compile time.
# Growing COMMAND_LABELS here without retraining would desync kws-fwgen's
# generated gen/labels.h from that model and break the firmware build. These
# compounds are grammar-ready but unrecognizable by the device until the next
# retrain adds them as their own classes (see docs/paper-notes.md E50).
#
# "Dach" (the roof-hatch light) is excluded: unlike Küche/Außen/Lesen, German
# speakers don't have a one-word compound for it -- "Dachlicht" never occurs
# in SITUATIONS' natural scene text below (grep confirms), where Küchenlicht/
# Außenlicht/Leselicht already do; the roof light is just "das Licht".
LIGHT_COMPOUNDS = {"Küchenlicht": "Küche", "Außenlicht": "Außen", "Leselicht": "Lesen"}
# Guided/elicit prompts for the above -- an/aus only, the idiom's natural
# form ("Küchenlicht heller" is not how this gets said; brightness stays
# "Licht Küche heller"). Same status as LIGHT_COMPOUNDS: not yet wired into
# prompt_sets()/SITUATIONS (kws_de.qc.vocab()/label_for_token only know
# DEVICES+ZONES+ACTIONS, so they cannot segment or label a token that isn't
# one of those -- wiring this in now would silently break word-cutting/QC for
# every take that reads it). Wire in at the retrain that adds LIGHT_COMPOUNDS
# to COMMAND_LABELS, the same way heller/dunkler etc. are single word classes.
LIGHT_COMPOUND_PROMPTS = [
    f"{word} {action}" for word in LIGHT_COMPOUNDS for action in ("an", "aus")
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


# Guided-recorder "Situationen" (elicitation) prompts: a scene or a question the
# device asks, paired with the EXPECTED INTENT it is meant to elicit — not the
# words to read. The speaker answers in their own phrasing, wake word included
# ("Hey Bus, mach das Licht in der Küche an"), so this collects natural command
# speech instead of read sentences (E30: read speech scored 0.27 on natural
# field clips). Weighted to real commands across DEVICES/ZONES/ACTIONS, plus a
# few question forms; deliberately no number-word drills (a scene that just
# asks for a number teaches nothing about natural phrasing).
SITUATIONS: list[tuple[str, str]] = [
    ("Es ist dunkel in der Küche.", "Licht Küche an"),
    ("Ihr geht schlafen, das Küchenlicht brennt noch.", "Licht Küche aus"),
    ("Draußen ist es stockfinster und ihr wollt raus.", "Licht Außen an"),
    ("Ihr seid wieder drin, das Außenlicht kann aus.", "Licht Außen aus"),
    ("Zum Lesen ist es zu dunkel.", "Licht Lesen an"),
    ("Du bist fertig mit Lesen und willst das Leselicht aus.", "Licht Lesen aus"),
    ("Wo soll das Licht angehen?", "Licht Dach an"),
    ("Wo soll das Licht ausgehen?", "Licht Dach aus"),
    ("Das Leselicht blendet dich beim Lesen.", "Licht Lesen dunkler"),
    ("Die Küche ist zu düster zum Kochen.", "Licht Küche heller"),
    ("Ein sanftes Licht zum Einschlafen in der Küche.", "Licht Küche fünfundzwanzig"),
    ("Du willst die Küche auf volle Helligkeit.", "Licht Küche hundert"),
    ("Wie hell soll das Licht in der Küche sein?", "Licht Küche fünfzig"),
    ("Wo soll die Heizung an?", "Heizung an"),
    ("Dir ist kalt im Bus.", "Heizung an"),
    ("Es ist drinnen zu warm geworden.", "Heizung aus"),
    ("Die Heizung läuft, aber dir ist immer noch kalt.", "Heizung wärmer"),
    ("Es wird euch langsam zu warm.", "Heizung kälter"),
    ("Was soll mit der Heizung passieren?", "Heizung aus"),
    ("Ihr kommt am Stellplatz an und wollt kühlen.", "Kühlschrank an"),
    ("Ihr packt ab und der Kühlschrank kann aus.", "Kühlschrank aus"),
    ("Der Kühlschrank brummt euch nachts wach.", "Kühlschrank leise"),
    ("Was soll mit dem Kühlschrank passieren?", "Kühlschrank an"),
    ("Ihr wollt das Dach für die Nacht öffnen.", "Aufstelldach auf"),
    ("Ihr fahrt los, das Dach muss zu.", "Aufstelldach zu"),
    ("Was soll mit dem Dach passieren?", "Aufstelldach auf"),
    ("Ihr kommt spät abends am Stellplatz an, es ist dunkel draußen.", "Licht Außen an"),
    ("Beim Kochen wird es dir zu warm in der Küche.", "Heizung kälter"),
    ("Das Leselicht ist dir zu schwach zum Lesen.", "Licht Lesen heller"),
    ("Wo soll es heller werden?", "Licht Dach heller"),
]

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
