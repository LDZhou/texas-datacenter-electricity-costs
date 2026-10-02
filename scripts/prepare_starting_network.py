"""Compatibility entry point for freezing a historical optimum."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_pipeline.scripts.prepare_network import *  # noqa: F401,F403
from paper_pipeline.scripts.prepare_network import main


if __name__ == "__main__":
    main()
