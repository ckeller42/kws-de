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
        monkeypatch.setenv("KWS_DATA_ROOT", str(tmp_path))
        importlib.reload(config)
        assert config.DATA_DIR == tmp_path / "data"
        assert config.MODELS_DIR == tmp_path / "models"

        monkeypatch.delenv("KWS_DATA_ROOT")
        importlib.reload(config)
        assert config.DATA_DIR == config._REPO_ROOT / "data"
        assert config.MODELS_DIR == config._REPO_ROOT / "models"
    finally:
        if original is None:
            monkeypatch.delenv("KWS_DATA_ROOT", raising=False)
        else:
            monkeypatch.setenv("KWS_DATA_ROOT", original)
        importlib.reload(config)
