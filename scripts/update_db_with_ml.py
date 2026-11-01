"""Compatibility entry point for refreshing stored priorities from the current rules."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.rescore_with_rules import rescore_all


if __name__ == "__main__":
    rescore_all()
