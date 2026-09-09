"""Original-paper investment/capacity plotting helpers."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nc_style import C, CARRIER_COLORS, apply
apply()
OUT = Path(__file__).resolve().parents[1] / "results/nc_paper_figures"
YEARS = list(range(2019,2024))

CARRIER_KEYS = [
    ("Gas OCGT", "new_OCGT_mw"),
    ("Solar", "new_solar_mw"),
    ("Wind", "new_onwind_mw"),
    ("Battery", "new_battery_mw"),
    ("Transmission", "new_transmission_mw"),
]
def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {name}")

def make_fig5(metrics: dict) -> None:
    fig, ax = plt.subplots(
        figsize=(5.2, 2.9),
        gridspec_kw={"left": 0.11, "right": 0.985, "top": 0.97, "bottom": 0.16},
    )
    width = 0.38
    xpos = np.arange(len(YEARS))
    gs_inv = [metrics[("generation_storage", y)]["total_annual_investment"] / 1e9 for y in YEARS]
    ft_inv = [metrics[("full_tx", y)]["total_annual_investment"] / 1e9 for y in YEARS]
    ax.bar(xpos - width / 2, gs_inv, width, color=C["orange"], hatch="///",
           edgecolor="white", linewidth=0.4, label="Generation + storage")
    ax.bar(xpos + width / 2, ft_inv, width, color=C["blue"],
           label="Full expansion")
    for x, v in zip(xpos - width / 2, gs_inv):
        ax.text(x, v + 0.4, f"{v:.1f}", ha="center", fontsize=5.8)
    for x, v in zip(xpos + width / 2, ft_inv):
        ax.text(x, v + 0.4, f"{v:.1f}", ha="center", fontsize=5.8)
    ax.set_xticks(xpos, YEARS)
    ax.set_xlabel("Weather year")
    ax.set_ylabel("Annualized new investment ($B per year)")
    ax.set_ylim(0, max(gs_inv) * 1.30)
    ax.legend(frameon=False, loc="upper right", borderaxespad=0.0, fontsize=6)
    save(fig, "fig5_all2030_annual_investment_by_year")

def _mix_bars(ax, metrics: dict, year=None) -> None:
    cases = ["generation_storage", "full_tx"]
    case_labels = ["Generation +\nstorage", "Full\nexpansion"]
    bottoms = np.zeros(len(cases))
    for label, key in CARRIER_KEYS:
        if year is None:
            vals = np.array([np.mean([metrics[(m, y)][key] for y in YEARS]) for m in cases]) / 1000
        else:
            vals = np.array([metrics[(m, year)][key] for m in cases]) / 1000
        ax.bar(case_labels, vals, 0.55, bottom=bottoms,
               color=CARRIER_COLORS[label], label=label)
        bottoms += vals
    for i, total in enumerate(bottoms):
        ax.text(i, total + 2.5, f"{total:.1f} GW", ha="center", fontsize=6.5)
    ax.set_ylim(0, max(bottoms) * 1.17)
    ax.legend(frameon=False, ncol=2, loc="upper right", borderaxespad=0.0,
              columnspacing=0.9, handletextpad=0.45, fontsize=6)

def make_fig6_fig7(metrics: dict) -> None:
    fig, ax = plt.subplots(
        figsize=(4.6, 3.1),
        gridspec_kw={"left": 0.12, "right": 0.98, "top": 0.97, "bottom": 0.13},
    )
    _mix_bars(ax, metrics, year=None)
    ax.set_ylabel("Mean new capacity, 2019–2023 (GW)")
    save(fig, "fig6_all2030_mean_capacity_mix")

    fig, ax = plt.subplots(
        figsize=(4.6, 3.1),
        gridspec_kw={"left": 0.12, "right": 0.98, "top": 0.97, "bottom": 0.13},
    )
    _mix_bars(ax, metrics, year=2022)
    ax.set_ylabel("New capacity, 2022 weather year (GW)")
    save(fig, "fig7_2022_all2030_capacity_mix")
