"""Tiny JSON state per run, so a crash or Ctrl-C never loses a paid training job."""

import json
from pathlib import Path
from typing import Any

from .config import run_dir


def _path(run: str) -> Path:
    return run_dir(run) / "state.json"


def load(run: str) -> dict[str, Any]:
    p = _path(run)
    return json.loads(p.read_text()) if p.exists() else {}


def save(run: str, data: dict[str, Any]) -> None:
    _path(run).write_text(json.dumps(data, indent=2))


def update(run: str, **kv: Any) -> dict[str, Any]:
    data = load(run)
    data.update(kv)
    save(run, data)
    return data
