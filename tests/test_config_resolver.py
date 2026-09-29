"""Storage resolver precedence (design §1): explicit env var > per-machine
config.toml > per-OS default, for the data root and each aux dir. Every case is
hermetic — XDG_CONFIG_HOME points at a tmp dir so the developer's real
~/.config/kws-de/config.toml never leaks in — and config is reloaded back to the
real environment at the end so the rest of the suite is unaffected."""

import importlib

import pytest

from kws_de import config


@pytest.fixture
def load_config(monkeypatch, tmp_path):
    """Reload kws_de.config under a controlled env + config file, then restore."""

    def _load(*, env=None, toml=None, xdg=None):
        for var in ("KWS_DATA_ROOT", "KWS_NOISE_DIR", "KWS_RIR_DIR"):
            monkeypatch.delenv(var, raising=False)
        xdg_home = xdg or (tmp_path / "xdg")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_home))
        if toml is not None:
            cfg_dir = xdg_home / "kws-de"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            (cfg_dir / "config.toml").write_text(toml)
        for key, val in (env or {}).items():
            monkeypatch.setenv(key, val)
        return importlib.reload(config)

    yield _load
    # Undo the monkeypatched env (automatic) then reload to the real config.
    monkeypatch.undo()
    importlib.reload(config)


def test_default_when_nothing_set(load_config):
    cfg = load_config()
    assert cfg._resolve_data_root() == cfg._default_data_root()
    assert cfg.DATA_DIR == cfg._default_data_root() / "data"
    assert cfg.MODELS_DIR == cfg._default_data_root() / "models"
    assert cfg.noise_dir() is None
    assert cfg.rir_dir() is None


def test_config_file_beats_default(load_config, tmp_path):
    root = tmp_path / "fromfile"
    noise = tmp_path / "n"
    rir = tmp_path / "r"
    cfg = load_config(toml=f'data_root = "{root}"\nnoise_dir = "{noise}"\nrir_dir = "{rir}"\n')
    assert cfg.DATA_DIR == root / "data"
    assert cfg.MODELS_DIR == root / "models"
    assert cfg.noise_dir() == noise
    assert cfg.rir_dir() == rir


def test_env_beats_config_file(load_config, tmp_path):
    from_file = tmp_path / "fromfile"
    from_env = tmp_path / "fromenv"
    noise_file, noise_env = tmp_path / "nf", tmp_path / "ne"
    rir_file, rir_env = tmp_path / "rf", tmp_path / "re"
    cfg = load_config(
        toml=(f'data_root = "{from_file}"\nnoise_dir = "{noise_file}"\nrir_dir = "{rir_file}"\n'),
        env={
            "KWS_DATA_ROOT": str(from_env),
            "KWS_NOISE_DIR": str(noise_env),
            "KWS_RIR_DIR": str(rir_env),
        },
    )
    assert cfg.DATA_DIR == from_env / "data"
    assert cfg.noise_dir() == noise_env
    assert cfg.rir_dir() == rir_env


def test_env_data_root_reproduces_legacy_behaviour(load_config, tmp_path):
    # Backward-compat contract: KWS_DATA_ROOT set == exactly the old behaviour
    # (<root>/data, <root>/models), regardless of platform or config file.
    cfg = load_config(env={"KWS_DATA_ROOT": str(tmp_path)})
    assert cfg.DATA_DIR == tmp_path / "data"
    assert cfg.MODELS_DIR == tmp_path / "models"


def test_malformed_config_file_is_ignored(load_config):
    cfg = load_config(toml="this is not = valid toml [[[")
    # A broken config resolves to {} rather than crashing: config is optional.
    assert cfg.DATA_DIR == cfg._default_data_root() / "data"
    assert cfg.noise_dir() is None
