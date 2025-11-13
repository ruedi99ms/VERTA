# route_analyzer/config.py
from __future__ import annotations
from typing import Any, Dict, Optional, Set
import json
import os

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_config_file(path: Optional[str]) -> Dict[str, Any]:
    """
    Load YAML or JSON into a dict. Returns {} if path is None.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    text = _load_text(path)
    lower = path.lower()
    if lower.endswith((".yml", ".yaml")):
        if not _HAS_YAML:
            raise RuntimeError("pyyaml is not installed but a YAML config was provided.")
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping/object.")
        return data
    if lower.endswith(".json"):
        data = json.loads(text) or {}
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON must be an object.")
        return data
    # fallback: try JSON first, then YAML
    try:
        data = json.loads(text) or {}
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    if _HAS_YAML:
        data = yaml.safe_load(text) or {}
        if isinstance(data, dict):
            return data
    raise ValueError("Unsupported config format (use .json, .yml, or .yaml).")


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dicts. override wins on conflicts.
    """
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def parse_columns(value: Any) -> Dict[str, str]:
    """
    Accept either:
      - a string like "x=X,z=Z,t=time"
      - a dict like {"x": "X", "z": "Z", "t": "time"}
    Return a dict.
    """
    if value is None:
        return {"x": "x", "z": "z", "t": "t"}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, str):
        out: Dict[str, str] = {}
        for kv in value.split(","):
            if not kv.strip():
                continue
            k, v = kv.split("=")
            out[k.strip()] = v.strip()
        return out
    raise ValueError("columns must be a dict or 'x=...,z=...,t=...' string")


def overlay_config_on_namespace(ns, cfg: Dict[str, Any], subcommand: str, provided_keys: Optional[Set[str]] = None) -> None:
    defaults = cfg.get("defaults", {})
    subcfg   = cfg.get(subcommand, {})
    flat     = deep_update(defaults, subcfg)

    if "columns" in flat:
        flat["columns"] = parse_columns(flat["columns"])

    for k, v in flat.items():
        if not hasattr(ns, k):
            continue
        # If user explicitly passed it on CLI, don't touch it.
        if provided_keys and k in provided_keys:
            continue

        cur = getattr(ns, k)
        if isinstance(cur, bool) and isinstance(v, bool):
            # Config may raise False->True (common pattern for flags); leave True->False untouched
            if cur is False and v is True:
                setattr(ns, k, True)
        else:
            # Not provided on CLI → config should override parser default
            setattr(ns, k, v)
