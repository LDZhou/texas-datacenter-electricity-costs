"""Public entry point for the paper's retailer analysis."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_pipeline.scripts.analyze_results import *  # noqa: F401,F403
from paper_pipeline.scripts.analyze_results import main

if __name__ == "__main__":
    sys.argv = ["--out-dir" if arg == "--output-dir" else arg for arg in sys.argv]
    main()
