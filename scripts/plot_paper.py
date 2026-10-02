"""Generate the paper figures after the planning and customer analyses."""
import sys
from run_paper import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "figures", *sys.argv[1:]]
    main()
