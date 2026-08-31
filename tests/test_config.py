from kws_de import config


def test_labels_are_commands_plus_aux():
    assert config.COMMANDS == ["Licht", "Kühlschrank", "Camping", "Heizung", "Wasser"]
    assert config.LABELS == config.COMMANDS + ["_unknown_", "_silence_"]
    assert config.NUM_CLASSES == 7


def test_label_index_roundtrip():
    for i, lbl in enumerate(config.LABELS):
        assert config.label_index(lbl) == i


def test_frame_count_matches_audio_geometry():
    # (CLIP_SAMPLES - WIN) // HOP + 1
    expected = (config.CLIP_SAMPLES - config.WIN_SAMPLES) // config.HOP_SAMPLES + 1
    assert config.N_FRAMES == expected


def test_data_dirs_are_under_repo_root():
    # Paths are repo-relative — no machine-specific absolute locations.
    assert config.DATA_DIR.name == "data"
    assert config.MODELS_DIR.name == "models"
