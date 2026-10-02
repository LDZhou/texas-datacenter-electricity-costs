"""Shared figure style for the NC submission.

Every figure script imports from here so that type, sizes, panel letters and -
most importantly - the meaning of each colour are identical across the paper.

Colour semantics
----------------
Two independent encodings are used in the paper, and they are kept disjoint so
that a colour never means two things:

  cost channels (what the customer pays)
      marginal generation   blue
      congestion            sky        (a component of wholesale, so a lighter
                                        shade of the same hue)
      regulated transmission vermillion
      retailer buffer       amber
      suppressed scarcity   purple     (never billed, so visually apart)

  expansion pathways (how the load is served)
      resource only         grey       (the counterfactual)
      coordinated           blue       (the case carried through the paper)

Technology mixes keep their own carrier palette, which appears only in stacked
capacity charts and never beside the two encodings above.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

SINGLE_COL = 3.50   # 89 mm
DOUBLE_COL = 7.20   # 183 mm

# Okabe-Ito colour-blind-safe palette
C = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "amber": "#E69F00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "grey": "#7F7F7F",
    "dgrey": "#595959",
    "lgrey": "#D9D9D9",
}
# backwards-compatible alias: older scripts spelled vermillion "orange"
C["orange"] = C["vermillion"]

# cost channels
CH_ENERGY = C["blue"]
CH_CONGESTION = C["sky"]
CH_REGULATED = C["vermillion"]
CH_BUFFER = C["amber"]
CH_SCARCITY = C["purple"]

# expansion pathways
PW_RESOURCE = C["dgrey"]
PW_COORDINATED = C["blue"]

CARRIER_COLORS = {
    "Gas OCGT": "#8A8A8A",
    "Gas CCGT": "#3B3B3B",
    "Solar": C["amber"],
    "Wind": C["green"],
    "Battery": C["sky"],
    "Transmission": C["blue"],
}

RC = {
    "font.family": "sans-serif",
    # Arial/Helvetica are not installed on the build host; listing DejaVu first
    # makes the rendered type deterministic instead of silently falling back.
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 450,
    # Nature requires embedded, editable type in vector artwork.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply() -> None:
    plt.rcParams.update(RC)


def panel_letter(ax, letter: str, dx: float = -0.16, dy: float = 1.10) -> None:
    """Bold lowercase panel letter, top-left, in axes coordinates."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="top", ha="left")
