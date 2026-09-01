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

# On-device resource budgets (see spec Global Constraints).
MAX_MODEL_BYTES = 500_000
MAX_ARENA_BYTES = 300_000
MAX_MACS = 3_000_000
MAX_LATENCY_MS = 30

_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _REPO_ROOT / "data"  # gitignored; where it physically lives is a local detail
MODELS_DIR = _REPO_ROOT / "models"  # gitignored


def label_index(label: str) -> int:
    return LABELS.index(label)


# --- v2: wake word + slot commands (additive; v1 constants above untouched) ---
WAKE_WORD = "Hey Bus"
WAKE_LABELS = ["wake", "_not_"]
DEVICES = ["Licht", "Heizung", "Kühlschrank", "Wasser"]
ZONES = ["Küche", "Bad", "Decke", "Außen"]
ACTIONS = ["an", "aus"]
COMMAND_LABELS = DEVICES + ZONES + ACTIONS + ["_unknown_", "_silence_"]


def command_index(label: str) -> int:
    return COMMAND_LABELS.index(label)
