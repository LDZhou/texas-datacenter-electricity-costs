"""Compatibility entry point for matched capped-adder decomposition."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_pipeline.scripts.compute_adder_factor_decomposition import *  # noqa: F401,F403
from paper_pipeline.scripts.compute_adder_factor_decomposition import main


if __name__ == "__main__":
    main()
