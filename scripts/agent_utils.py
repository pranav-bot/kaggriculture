"""Load submission agents for local scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ROOT / "submissions"


def load_agent(name: str, *, module_suffix: str = "") -> Any:
    """Resolve a built-in opponent name or load a submission main.py."""
    if name in ("random", "pass", "starter"):
        return name
    path = Path(name)
    if path.is_dir() and (path / "main.py").is_file():
        main_py = path / "main.py"
    elif (SUBMISSIONS / name / "main.py").is_file():
        main_py = SUBMISSIONS / name / "main.py"
    elif path.is_file():
        main_py = path
    else:
        from kaggriculture.actions.controller import ActionController
        from kaggriculture.env.items import Plants

        return ActionController(target_crop=Plants.WHEAT)

    mod_name = f"submission_{main_py.parent.name}{module_suffix}"
    spec = importlib.util.spec_from_file_location(mod_name, main_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load agent from {main_py}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(main_py.parent))
    spec.loader.exec_module(module)
    if hasattr(module, "controller"):
        return module.controller
    if hasattr(module, "agent"):
        return module.agent
    raise RuntimeError(f"No agent or controller export in {main_py}")
