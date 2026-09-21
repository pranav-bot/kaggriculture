"""Architecture Alpha: modular champion heuristic (Kaggle entrypoint)."""

import os
import sys
from typing import Any, Dict

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from orchestrator import agent as _agent


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return _agent(obs)
