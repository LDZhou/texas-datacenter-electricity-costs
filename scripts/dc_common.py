"""Compatibility exports for the revision experiment utilities."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_pipeline.scripts.dc_common import *  # noqa: F401,F403
