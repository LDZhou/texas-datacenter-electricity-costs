"""Compatibility entry point for the revision experiment runner."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_pipeline.scripts.run_experiment import *  # noqa: F401,F403
from paper_pipeline.scripts.run_experiment import main


if __name__ == "__main__":
    main()
