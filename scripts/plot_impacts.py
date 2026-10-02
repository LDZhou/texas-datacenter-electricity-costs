"""Public entry point for the customer-cost and planning figures."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_pipeline.scripts.make_revision_figures import main

if __name__ == "__main__":
    main()
