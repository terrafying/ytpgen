"""Config loading: config/ytp.yaml overrides built-in defaults."""

from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULTS: dict = {
    "workspace": "/Volumes/evol/ytpgen",
    "model_size": "base",
    "compute_type": "int8",
    "language": "en",
    "scene": {
        "threshold": 27.0,
        "min_scene_len_s": 1.0,
    },
    "download": {
        "height": 720,
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
    },
    "compose": {
        "fps": 30,
        "width": 1280,
        "height": 720,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load YAML/JSON config over defaults. Missing file returns defaults."""
    if path is None:
        return copy.deepcopy(DEFAULTS)
    p = Path(path)
    if not p.exists():
        return copy.deepcopy(DEFAULTS)
    text = p.read_text(encoding="utf-8")
    data: dict
    if p.suffix in (".yaml", ".yml"):
        import yaml  # type: ignore[import]

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {p}")
    return _deep_merge(DEFAULTS, data)


def workspace_dir(cfg: dict, create: bool = True) -> Path:
    """Resolve the workspace dir (bulk media + db + outputs live here)."""
    ws = Path(cfg["workspace"]).expanduser()
    if create:
        ws.mkdir(parents=True, exist_ok=True)
    return ws
