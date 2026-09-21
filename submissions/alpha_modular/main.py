"""Architecture Alpha: modular champion heuristic (Kaggle entrypoint)."""

import os
import sys
from typing import Any, Dict

if "__file__" in globals():
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_here))
else:
    sys.path.insert(0, os.getcwd())

from alpha_modular.orchestrator import agent as _agent


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return _agent(obs)
