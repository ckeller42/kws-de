"""Mine an extracted MSWC-de tarball (https://mlcommons.org/datasets/multilingual-spoken-words/,
CC-BY 4.0) by keyword folder, instead of streaming the HuggingFace mirror
alphabetically (`kws_de.data._fetch_mswc`), which never reached most of our
command words before its scan cap.

Expected layout under `root` (gitignored, e.g. `data/mswc/de/`):

    clips/<keyword>/<clip>.opus      1 s, 16 kHz mono
    de_splits.csv                    SET,LINK,WORD,VALID,SPEAKER,GENDER

Returns the same `{label: [(np.ndarray float32, speaker_id)]}` dict as
`_fetch_mswc`, so split/augment/manifest code is unchanged.
"""

from pathlib import Path

import numpy as np


def _folder_index(root: Path) -> dict[str, Path]:
    """Lower-cased keyword folder name -> folder path (case-insensitive lookup:
    config spells `Küche`, MSWC folders are lower-case)."""
    clips = Path(root) / "clips"
    return {p.name.lower(): p for p in sorted(clips.iterdir()) if p.is_dir()}


def _pick(items: list, n: int, rng: np.random.Generator) -> list:
    """Seeded shuffle, first `n`. Does not mutate `items`."""
    order = rng.permutation(len(items))
    return [items[i] for i in order[:n]]
