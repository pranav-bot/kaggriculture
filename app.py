"""Root Streamlit App entry point for Kaggriculture Telemetry Dashboard."""

import os
import sys

# Ensure workspace root and dashboard packages are discoverable
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dashboard.app import main

if __name__ == "__main__":
    main()
