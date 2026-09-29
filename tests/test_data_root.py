"""KWS_DATA_ROOT relocates the gitignored data/ and models/ dirs to one shared
root (e.g. an external SSD used by every worktree); unset, they stay
repo-relative. The module is reloaded around the env change and restored at the
end so the rest of the suite sees the real configuration."""

import importlib
import os

from kws_de import config


def test_data_root_env_relocates_data_and_models(monkeypatch, tmp_path):
    original = os.environ.get("KWS_DATA_ROOT")
    try:
        # Point XDG at an empty tmp dir so no real ~/.config/kws-de/config.toml
        # leaks into the "unset" branch below.
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        monkeypatch.setenv("KWS_DATA_ROOT", str(tmp_path))
        importlib.reload(config)
        assert config.DATA_DIR == tmp_path / "data"
        assert config.MODELS_DIR == tmp_path / "models"

        monkeypatch.delenv("KWS_DATA_ROOT")
        importlib.reload(config)
        # Unset falls back to the per-OS default (repo root on macOS, ~/kws-data
        # on Linux) — a computed default, never a committed machine path.
        assert config.DATA_DIR == config._default_data_root() / "data"
        assert config.MODELS_DIR == config._default_data_root() / "models"
    finally:
        if original is None:
            monkeypatch.delenv("KWS_DATA_ROOT", raising=False)
        else:
            monkeypatch.setenv("KWS_DATA_ROOT", original)
        importlib.reload(config)
