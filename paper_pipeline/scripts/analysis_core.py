"""
analysis_core.py
================

Stylised retail-electricity-provider (REP / LSE) risk model for PyPSA-USA
ERCOT data-center simulations.

This module is intentionally a *risk translation layer*, not a full
regulatory/accounting model of a real REP. It converts modeled wholesale
LMP paths into balance-sheet stress metrics for a stylised fixed-price
load-serving entity.

Core economic model
-------------------
A representative REP sells electricity at an energy retail price G and
procures energy through a mixture of forward hedges and spot purchases:

    margin_t = G_eff - [h * F + (1 - h) * LMP_t] - OpEx

where:
    G_eff  = retail energy price after optional pass-through
    h      = hedge ratio
    F      = forward hedge price
    LMP_t  = spot wholesale price
    OpEx   = non-energy REP operating cost

Important interpretation
------------------------
The model reports *capital breach / solvency stress*, not literal REP
bankruptcy probability. A single scenario path either breaches the chosen
capital buffer or does not. When averaged over years / locations, this
becomes a capital-breach frequency across scenario-years.

Forward-pricing cases
---------------------
1. scenario-repriced hedge:
       F = mean(LMP_scenario) + premium
   Represents long-run market adaptation where forward prices already
   incorporate the data-center scenario.

2. base-forward hedge:
       F = mean(LMP_base) + premium
   Represents under-anticipated data-center growth where a REP hedged
   based on a no-DC baseline expectation before the wholesale shock.

The second case is usually more policy-relevant for unexpected or
under-planned data-center growth.

Backwards compatibility
-----------------------
For compatibility with older plotting code, the returned metrics still
include aliases such as:
    bankruptcy_prob
    annual_profit_dollar
    max_collateral_call_dollar
    collateral_to_capital_ratio

Prefer the new names in papers:
    capital_breach
    horizon_profit_dollar
    max_48h_procurement_requirement_dollar
    liquidity_requirement_to_capital_ratio
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =====================================================================
# 1. Texas retail price data
# =====================================================================

# EIA monthly average residential retail price for Texas, cents/kWh.
# Keep this as a fallback only. Prefer EIA API when an API key is available.
TEXAS_RETAIL_PRICE_MONTHLY = {
    (2019, 1): 11.20, (2019, 2): 11.31, (2019, 3): 11.22, (2019, 4): 11.17,
    (2019, 5): 11.24, (2019, 6): 11.41, (2019, 7): 11.52, (2019, 8): 11.83,
    (2019, 9): 11.94, (2019, 10): 11.46, (2019, 11): 11.02, (2019, 12): 10.98,

    (2020, 1): 11.54, (2020, 2): 11.42, (2020, 3): 11.31, (2020, 4): 11.02,
    (2020, 5): 11.48, (2020, 6): 11.95, (2020, 7): 12.01, (2020, 8): 12.10,
    (2020, 9): 12.32, (2020, 10): 12.19, (2020, 11): 11.74, (2020, 12): 11.52,

    (2021, 1): 11.35, (2021, 2): 12.69, (2021, 3): 11.70, (2021, 4): 12.08,
    (2021, 5): 12.09, (2021, 6): 12.04, (2021, 7): 11.73, (2021, 8): 11.94,
    (2021, 9): 12.45, (2021, 10): 12.56, (2021, 11): 12.23, (2021, 12): 12.10,

    (2022, 1): 12.45, (2022, 2): 12.48, (2022, 3): 13.01, (2022, 4): 13.29,
    (2022, 5): 13.34, (2022, 6): 13.49, (2022, 7): 13.71, (2022, 8): 14.12,
    (2022, 9): 14.75, (2022, 10): 14.94, (2022, 11): 15.01, (2022, 12): 14.49,

    (2023, 1): 14.36, (2023, 2): 14.72, (2023, 3): 14.61, (2023, 4): 14.36,
    (2023, 5): 14.65, (2023, 6): 14.35, (2023, 7): 13.95, (2023, 8): 14.17,
    (2023, 9): 14.68, (2023, 10): 14.89, (2023, 11): 14.79, (2023, 12): 14.64,
}

# Approximate regulated delivery share of all-in Texas residential price.
# This is a stylised decomposition. Use scenario/sensitivity checks.
TDU_FRACTION_BY_YEAR = {
    2019: 0.38,
    2020: 0.37,
    2021: 0.36,
    2022: 0.34,
    2023: 0.35,
}


def fetch_eia_retail_prices(
    api_key: str,
    state: str = "TX",
    sectors: Optional[List[str]] = None,
    start: str = "2019-01",
    end: str = "2023-12",
) -> pd.DataFrame:
    """
    Fetch monthly retail electricity prices from EIA API v2.

    Returns a DataFrame with EIA rows. If the API call fails, returns an
    empty DataFrame and the caller can use fallback data.
    """
    import requests

    if sectors is None:
        sectors = ["RES", "COM", "IND"]

    url = "https://api.eia.gov/v2/electricity/retail-sales/data"
    all_rows = []

    for sector in sectors:
        params = {
            "api_key": api_key,
            "data[]": ["price", "revenue", "sales"],
            "facets[stateid][]": state,
            "facets[sectorid][]": sector,
            "frequency": "monthly",
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000,
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if "response" in data and "data" in data["response"]:
                all_rows.extend(data["response"]["data"])
        except Exception as exc:
            print(f"  [WARN] EIA API call failed for {sector}: {exc}")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["period"] = pd.to_datetime(df["period"], format="%Y-%m")
    for col in ["price", "revenue", "sales"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_retail_energy_price(
    year: int,
    api_key: Optional[str] = None,
    sector: str = "RES",
    tdu_fraction_override: Optional[float] = None,
) -> Dict:
    """
    Get the stylised energy component of Texas retail price for one year.

    Returns:
        total_retail_cents_kwh
        tdu_fraction
        energy_cents_kwh
        energy_dollar_mwh
        monthly_energy_dollar_mwh
        monthly_total_cents_kwh
    """
    monthly_total: Dict[int, float] = {}

    if api_key:
        try:
            df = fetch_eia_retail_prices(
                api_key,
                start=f"{year}-01",
                end=f"{year}-12",
                sectors=[sector],
            )
            if not df.empty and "price" in df.columns:
                for _, row in df.iterrows():
                    monthly_total[int(row["period"].month)] = float(row["price"])
                print(f"  [EIA API] fetched {len(monthly_total)} months for TX {sector} {year}")
        except Exception as exc:
            print(f"  [WARN] EIA API failed, using fallback: {exc}")

    if not monthly_total:
        for (y, m), price in TEXAS_RETAIL_PRICE_MONTHLY.items():
            if y == year:
                monthly_total[m] = price

        if monthly_total:
            print(f"  [Fallback] Using hardcoded EIA data for TX {sector} {year}")
        else:
            nearest_year = min(
                {ym[0] for ym in TEXAS_RETAIL_PRICE_MONTHLY},
                key=lambda y: abs(y - year),
            )
            print(f"  [WARN] No fallback data for {year}, using {nearest_year}")
            for (y, m), price in TEXAS_RETAIL_PRICE_MONTHLY.items():
                if y == nearest_year:
                    monthly_total[m] = price

    tdu_frac = (
        tdu_fraction_override
        if tdu_fraction_override is not None
        else TDU_FRACTION_BY_YEAR.get(year, 0.36)
    )

    avg_total = float(np.mean(list(monthly_total.values())))
    monthly_energy = {
        m: float(total * (1.0 - tdu_frac) * 10.0)
        for m, total in monthly_total.items()
    }
    avg_energy_cents = avg_total * (1.0 - tdu_frac)

    return {
        "year": year,
        "sector": sector,
        "total_retail_cents_kwh": round(avg_total, 2),
        "tdu_fraction": float(tdu_frac),
        "energy_cents_kwh": round(avg_energy_cents, 2),
        "energy_dollar_mwh": round(avg_energy_cents * 10.0, 2),
        "monthly_energy_dollar_mwh": monthly_energy,
        "monthly_total_cents_kwh": monthly_total,
    }


# =====================================================================
# 2. Stylised REP financial model
# =====================================================================


@dataclass
class REPConfig:
    """
    Configuration for a stylised fixed-price load-serving entity.

    Args:
        served_load_mw:
            Average served load represented by the REP. This is a
            normalisation choice, not a claim about a specific company.

        capital_mw_dollar:
            Capital / liquidity buffer. A capital breach means cumulative
            modelled losses exceed this buffer.

        retail_price_mwh:
            Baseline retail energy component, excluding regulated delivery.

        operating_cost_mwh:
            Non-energy operating cost.

        hedge_ratio:
            Share of load hedged via a forward contract.

        forward_premium_mwh:
            Risk premium added to the expected LMP to form forward price.

        retail_pass_through:
            Fraction of cost_adder_mwh passed through into retail price.
            0 means fixed-price retail exposure.
            1 means full recovery of the modeled adder.

        cost_adder_mwh:
            Additional per-MWh cost added to OpEx unless passed through.
            This can represent TCOS or all-in system cost recovery.

        hedge_ratios_scan:
            Candidate hedge ratios for simple grid-search optimisation.
    """
    name: str = "Stylized_Fixed_Price_LSE"
    served_load_mw: float = 500.0
    capital_mw_dollar: float = 5e6
    retail_price_mwh: float = 70.0
    operating_cost_mwh: float = 3.0
    hedge_ratio: float = 0.70
    forward_premium_mwh: float = 5.0
    retail_pass_through: float = 0.0
    cost_adder_mwh: float = 0.0
    hedge_ratios_scan: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    )


class REPFinancialModel:
    """
    Stylised REP/LSE P&L model for an LMP time series.

    The model is deliberately transparent:
        margin_t = G_eff - [h * F + (1-h) * LMP_t] - OpEx_eff

    It should be described as a balance-sheet stress proxy, not a literal
    bankruptcy model.
    """

    def __init__(
        self,
        lmp_series: pd.Series,
        config: REPConfig,
        hours_per_step: float = 3.0,
    ):
        if lmp_series is None or len(lmp_series) == 0:
            raise ValueError("lmp_series must be non-empty")

        self.lmp = pd.Series(lmp_series).astype(float).copy()
        self.cfg = config
        self.hours_per_step = float(hours_per_step)
        self.n_steps = len(self.lmp)
        self.total_hours = self.n_steps * self.hours_per_step
        self.lmp_mean = float(self.lmp.mean())
        self.lmp_std = float(self.lmp.std())

    # -----------------------------------------------------------------
    # Price / cost construction
    # -----------------------------------------------------------------

    def compute_forward_price(
        self,
        premium: Optional[float] = None,
        reference_lmp_mean: Optional[float] = None,
    ) -> float:
        """
        Compute forward price.

        If reference_lmp_mean is None:
            F = mean(LMP_scenario) + premium
            This is the scenario-repriced / market-adapted case.

        If reference_lmp_mean is provided:
            F = reference_lmp_mean + premium
            This is the base-forward / ex-ante hedge case.
        """
        if premium is None:
            premium = self.cfg.forward_premium_mwh
        ref = self.lmp_mean if reference_lmp_mean is None else float(reference_lmp_mean)
        return float(ref + premium)

    def compute_effective_retail_price(
        self,
        cost_adder_mwh: Optional[float] = None,
        pass_through: Optional[float] = None,
    ) -> float:
        """
        Retail energy price after optional pass-through.

        G_eff = G + pass_through * cost_adder
        """
        adder = self.cfg.cost_adder_mwh if cost_adder_mwh is None else float(cost_adder_mwh)
        pt = self.cfg.retail_pass_through if pass_through is None else float(pass_through)
        return float(self.cfg.retail_price_mwh + pt * adder)

    def compute_effective_opex(
        self,
        cost_adder_mwh: Optional[float] = None,
        pass_through: Optional[float] = None,
    ) -> float:
        """
        Operating cost net of pass-through.

        OpEx_eff = OpEx + (1 - pass_through) * cost_adder
        """
        adder = self.cfg.cost_adder_mwh if cost_adder_mwh is None else float(cost_adder_mwh)
        pt = self.cfg.retail_pass_through if pass_through is None else float(pass_through)
        return float(self.cfg.operating_cost_mwh + (1.0 - pt) * adder)

    # -----------------------------------------------------------------
    # P&L
    # -----------------------------------------------------------------

    def compute_pnl_series(
        self,
        hedge_ratio: Optional[float] = None,
        forward_price: Optional[float] = None,
        reference_lmp_mean: Optional[float] = None,
        cost_adder_mwh: Optional[float] = None,
        pass_through: Optional[float] = None,
    ) -> pd.Series:
        """
        Per-MWh margin at each timestep.

        Use forward_price for an explicit hedge price.
        Use reference_lmp_mean for a base-forward hedge.
        Use neither for scenario-repriced hedge.
        """
        h = self.cfg.hedge_ratio if hedge_ratio is None else float(hedge_ratio)
        h = min(max(h, 0.0), 1.0)

        if forward_price is None:
            F = self.compute_forward_price(reference_lmp_mean=reference_lmp_mean)
            forward_case = "base_forward" if reference_lmp_mean is not None else "scenario_repriced"
        else:
            F = float(forward_price)
            forward_case = "explicit_forward"

        G_eff = self.compute_effective_retail_price(cost_adder_mwh, pass_through)
        opex_eff = self.compute_effective_opex(cost_adder_mwh, pass_through)

        procurement = h * F + (1.0 - h) * self.lmp
        margin = G_eff - procurement - opex_eff
        margin.name = f"margin_{forward_case}"
        return margin

    def compute_total_pnl(
        self,
        hedge_ratio: Optional[float] = None,
        forward_price: Optional[float] = None,
        reference_lmp_mean: Optional[float] = None,
        cost_adder_mwh: Optional[float] = None,
        pass_through: Optional[float] = None,
    ) -> pd.Series:
        """Dollar P&L per timestep."""
        margin = self.compute_pnl_series(
            hedge_ratio=hedge_ratio,
            forward_price=forward_price,
            reference_lmp_mean=reference_lmp_mean,
            cost_adder_mwh=cost_adder_mwh,
            pass_through=pass_through,
        )
        pnl = margin * self.cfg.served_load_mw * self.hours_per_step
        pnl.name = "pnl_dollar"
        return pnl

    def compute_cumulative_pnl(
        self,
        hedge_ratio: Optional[float] = None,
        forward_price: Optional[float] = None,
        reference_lmp_mean: Optional[float] = None,
        cost_adder_mwh: Optional[float] = None,
        pass_through: Optional[float] = None,
    ) -> pd.Series:
        """Cumulative P&L over the modeled horizon."""
        return self.compute_total_pnl(
            hedge_ratio=hedge_ratio,
            forward_price=forward_price,
            reference_lmp_mean=reference_lmp_mean,
            cost_adder_mwh=cost_adder_mwh,
            pass_through=pass_through,
        ).cumsum()

    # -----------------------------------------------------------------
    # Risk metrics
    # -----------------------------------------------------------------

    def risk_metrics(
        self,
        hedge_ratio: Optional[float] = None,
        forward_price: Optional[float] = None,
        reference_lmp_mean: Optional[float] = None,
        cost_adder_mwh: Optional[float] = None,
        pass_through: Optional[float] = None,
        alpha: float = 0.05,
        annualize_profit: bool = False,
    ) -> Dict:
        """
        Compute stylised REP risk metrics.

        New preferred fields:
            horizon_profit_dollar
            annualized_profit_dollar
            capital_breach
            capital_breach_frequency
            max_48h_procurement_requirement_dollar
            liquidity_requirement_to_capital_ratio

        Backward-compatible aliases are also returned.
        """
        h = self.cfg.hedge_ratio if hedge_ratio is None else float(hedge_ratio)
        h = min(max(h, 0.0), 1.0)

        if forward_price is None:
            F = self.compute_forward_price(reference_lmp_mean=reference_lmp_mean)
            forward_pricing_case = (
                "base_forward" if reference_lmp_mean is not None else "scenario_repriced"
            )
        else:
            F = float(forward_price)
            forward_pricing_case = "explicit_forward"

        adder = self.cfg.cost_adder_mwh if cost_adder_mwh is None else float(cost_adder_mwh)
        pt = self.cfg.retail_pass_through if pass_through is None else float(pass_through)
        G_eff = self.compute_effective_retail_price(adder, pt)
        opex_eff = self.compute_effective_opex(adder, pt)

        pnl_mwh = self.compute_pnl_series(
            hedge_ratio=h,
            forward_price=F,
            cost_adder_mwh=adder,
            pass_through=pt,
        )
        pnl_dollar = pnl_mwh * self.cfg.served_load_mw * self.hours_per_step
        cum_pnl = pnl_dollar.cumsum()

        horizon_profit = float(pnl_dollar.sum())
        annual_factor = 8760.0 / self.total_hours if self.total_hours > 0 else np.nan
        annualized_profit = horizon_profit * annual_factor if annualize_profit else np.nan

        mean_margin = float(pnl_mwh.mean())
        std_margin = float(pnl_mwh.std())
        sharpe = mean_margin / std_margin if std_margin > 0 else np.inf

        sorted_pnl = np.sort(pnl_dollar.values)
        var_idx = int(np.floor(alpha * len(sorted_pnl)))
        var_idx = min(max(var_idx, 0), len(sorted_pnl) - 1)
        var_value = float(sorted_pnl[var_idx])
        cvar_value = float(sorted_pnl[: max(var_idx, 1)].mean())

        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        max_drawdown = float(drawdown.min())

        capital = float(self.cfg.capital_mw_dollar)
        running_min = cum_pnl.cummin()
        capital_breach = bool((running_min < -capital).any())
        distress_fraction = float((cum_pnl < -capital).sum() / len(cum_pnl))

        window_h = 48.0
        window_steps = max(1, int(window_h / self.hours_per_step))
        procurement_cash_outflow = (h * F + (1.0 - h) * self.lmp) * self.cfg.served_load_mw * self.hours_per_step
        rolling_outflow = procurement_cash_outflow.rolling(window_steps, min_periods=1).sum()
        max_48h_procurement = float(rolling_outflow.max())

        hedging_cost = h * (F - self.lmp_mean) * self.cfg.served_load_mw * self.total_hours
        loss_hours = float((pnl_mwh < 0).sum() * self.hours_per_step)
        loss_fraction = loss_hours / self.total_hours if self.total_hours > 0 else np.nan

        metrics = {
            "rep_model": "stylized_fixed_price_lse",
            "forward_pricing_case": forward_pricing_case,
            "hedge_ratio": h,
            "forward_price": round(F, 4),
            "reference_lmp_mean": (
                round(float(reference_lmp_mean), 4)
                if reference_lmp_mean is not None
                else np.nan
            ),
            "retail_price_G": round(float(self.cfg.retail_price_mwh), 4),
            "effective_retail_price_G": round(G_eff, 4),
            "base_opex_mwh": round(float(self.cfg.operating_cost_mwh), 4),
            "effective_opex_mwh": round(opex_eff, 4),
            "cost_adder_mwh": round(adder, 4),
            "retail_pass_through": round(pt, 4),

            "mean_lmp": round(float(self.lmp_mean), 4),
            "std_lmp": round(float(self.lmp_std), 4),
            "skew_lmp": round(float(self.lmp.skew()), 4),
            "kurt_lmp": round(float(self.lmp.kurtosis()), 4),
            "p95_lmp": round(float(np.percentile(self.lmp, 95)), 4),
            "p99_lmp": round(float(np.percentile(self.lmp, 99)), 4),

            "mean_margin_mwh": round(mean_margin, 4),
            "std_margin_mwh": round(std_margin, 4),
            "sharpe_ratio": round(float(sharpe), 4) if np.isfinite(sharpe) else np.inf,

            "horizon_profit_dollar": round(horizon_profit, 0),
            "annualized_profit_dollar": (
                round(float(annualized_profit), 0)
                if np.isfinite(annualized_profit)
                else np.nan
            ),
            "var_5pct_dollar": round(var_value, 0),
            "cvar_5pct_dollar": round(cvar_value, 0),
            "max_drawdown_dollar": round(max_drawdown, 0),

            "capital_buffer_dollar": round(capital, 0),
            "capital_breach": float(capital_breach),
            "capital_breach_frequency": float(capital_breach),
            "distress_fraction": round(distress_fraction, 4),

            "max_48h_procurement_requirement_dollar": round(max_48h_procurement, 0),
            "liquidity_requirement_to_capital_ratio": (
                round(max_48h_procurement / capital, 4) if capital > 0 else np.inf
            ),

            "hedging_cost_dollar": round(float(hedging_cost), 0),
            "loss_hours": round(loss_hours, 2),
            "loss_fraction": round(loss_fraction, 4),
            "total_hours": round(float(self.total_hours), 2),
        }

        # Backward-compatible aliases. Prefer the new names in paper text.
        metrics["bankruptcy_prob"] = metrics["capital_breach"]
        metrics["annual_profit_dollar"] = metrics["horizon_profit_dollar"]
        metrics["max_collateral_call_dollar"] = metrics["max_48h_procurement_requirement_dollar"]
        metrics["collateral_to_capital_ratio"] = metrics["liquidity_requirement_to_capital_ratio"]

        return metrics

    def optimal_hedge_ratio(
        self,
        ratios: Optional[List[float]] = None,
        objective: str = "cvar",
        forward_price_override: Optional[float] = None,
        reference_lmp_mean: Optional[float] = None,
        cost_adder_mwh: Optional[float] = None,
        pass_through: Optional[float] = None,
    ) -> Dict:
        """
        Find a simple grid-search hedge ratio.

        objective:
            cvar    -> maximize 5% CVaR, least negative is best
            var     -> maximize 5% VaR
            sharpe  -> maximize Sharpe ratio
            breach  -> minimize capital breach, then maximize CVaR
        """
        if ratios is None:
            ratios = self.cfg.hedge_ratios_scan

        rows = []
        for h in ratios:
            m = self.risk_metrics(
                hedge_ratio=h,
                forward_price=forward_price_override,
                reference_lmp_mean=reference_lmp_mean,
                cost_adder_mwh=cost_adder_mwh,
                pass_through=pass_through,
            )
            rows.append(m)

        df = pd.DataFrame(rows)

        if objective == "cvar":
            best_idx = df["cvar_5pct_dollar"].idxmax()
        elif objective == "var":
            best_idx = df["var_5pct_dollar"].idxmax()
        elif objective == "sharpe":
            best_idx = df["sharpe_ratio"].replace([np.inf, -np.inf], np.nan).idxmax()
        elif objective == "breach":
            tmp = df.sort_values(
                ["capital_breach", "cvar_5pct_dollar"],
                ascending=[True, False],
            )
            best_idx = tmp.index[0]
        else:
            raise ValueError(f"Unknown objective: {objective}")

        return {
            "scan_results": df,
            "optimal_hedge_ratio": float(df.loc[best_idx, "hedge_ratio"]),
            "optimal_metrics": df.loc[best_idx].to_dict(),
            "objective": objective,
        }


# =====================================================================
# 3. LMP extraction from PyPSA networks
# =====================================================================


def _assign_ercot_zone_from_xy(x: float, y: float) -> str:
    """Simple ERCOT zone heuristic used consistently across experiments."""
    if x < -100.0:
        return "WEST"
    if y >= 32.0:
        return "NORTH"
    if x >= -96.5 and y < 30.5:
        return "HOUSTON"
    return "SOUTH"


def _load_weights_by_bus(net, candidate_buses: pd.Index, exclude_dc: bool = True) -> Optional[pd.Series]:
    """
    Demand weights for LMP aggregation.

    Uses mean loads_t.p_set by bus. If unavailable, returns None.
    """
    if not hasattr(net, "loads_t") or "p_set" not in net.loads_t:
        return None
    if net.loads_t.p_set.empty or net.loads.empty:
        return None

    load_ids = net.loads_t.p_set.columns.intersection(net.loads.index)
    if len(load_ids) == 0:
        return None

    if exclude_dc:
        load_ids = [lid for lid in load_ids if not str(lid).startswith("DC_")]
        if len(load_ids) == 0:
            return None

    mean_load = net.loads_t.p_set[load_ids].mean()
    load_buses = net.loads.loc[mean_load.index, "bus"]
    bus_load = mean_load.groupby(load_buses).sum()

    common = pd.Index(candidate_buses).intersection(bus_load.index)
    if len(common) == 0:
        return None

    w = bus_load.loc[common].astype(float).clip(lower=0)
    if w.sum() <= 0:
        return None
    return w / w.sum()


def extract_zonal_lmp(
    network_path: str,
    zone: str = "system",
    clip_lower: float = -251.0,
    clip_upper: float = 9000.0,
    weight_by_load: bool = True,
    exclude_dc_from_weights: bool = True,
) -> pd.Series:
    """
    Extract system or zone LMP time series from a PyPSA network.

    Args:
        zone:
            "system", "HOUSTON", "NORTH", "SOUTH", or "WEST".

        clip_lower / clip_upper:
            ERCOT-like offer-cap clipping for stylised financial stress
            analysis. This prevents numerical artifacts from dominating
            REP P&L.

        weight_by_load:
            If True, aggregate buses by base-load weights when available.

        exclude_dc_from_weights:
            If True, use non-DC loads for system/zone weighting so the REP
            represents ordinary retail load, not the data center itself.
    """
    import pypsa

    net = pypsa.Network(str(network_path))
    lmp = net.buses_t.marginal_price.copy()

    if isinstance(lmp.index, pd.MultiIndex):
        lmp.index = pd.DatetimeIndex(lmp.index.get_level_values("timestep"))

    lmp = lmp.clip(lower=clip_lower, upper=clip_upper)

    zone_upper = zone.upper()
    buses = net.buses.copy()

    if zone_upper == "SYSTEM":
        candidate_buses = lmp.columns.intersection(buses.index)
    else:
        zones = buses.apply(
            lambda row: _assign_ercot_zone_from_xy(row["x"], row["y"]),
            axis=1,
        )
        candidate_buses = zones[zones == zone_upper].index.intersection(lmp.columns)

    if len(candidate_buses) == 0:
        print(f"  [WARN] no buses for zone={zone}; using simple system mean")
        return lmp.mean(axis=1)

    if weight_by_load:
        weights = _load_weights_by_bus(net, candidate_buses, exclude_dc=exclude_dc_from_weights)
        if weights is not None and len(weights) > 0:
            common = weights.index.intersection(lmp.columns)
            return (lmp[common] * weights.loc[common]).sum(axis=1)

    return lmp[candidate_buses].mean(axis=1)


# =====================================================================
# 4. Scenario comparison helpers
# =====================================================================


def run_scenario_comparison(
    base_lmp: pd.Series,
    dc_lmps: Dict[str, pd.Series],
    year: int,
    rep_config: REPConfig,
    api_key: Optional[str] = None,
    output_dir: Optional[Path] = None,
    hours_per_step: float = 3.0,
    forward_case: str = "base_forward",
    cost_adder_by_scenario: Optional[Dict[str, float]] = None,
    pass_through: Optional[float] = None,
) -> pd.DataFrame:
    """
    Run REP risk comparison across base and DC LMP paths.

    forward_case:
        "base_forward"       -> all scenarios use base mean LMP + premium.
        "scenario_repriced"  -> each scenario uses its own mean LMP + premium.

    cost_adder_by_scenario:
        Optional dict mapping scenario label to $/MWh cost adder.

    pass_through:
        Optional override for retail pass-through fraction.
    """
    if output_dir is None:
        raise ValueError("output_dir is required")
    output_dir.mkdir(parents=True, exist_ok=True)

    price_info = get_retail_energy_price(year, api_key)
    rep_config.retail_price_mwh = price_info["energy_dollar_mwh"]

    base_model = REPFinancialModel(base_lmp, rep_config, hours_per_step)
    base_reference = base_model.lmp_mean if forward_case == "base_forward" else None

    scenarios = {"Base": base_lmp}
    scenarios.update(dc_lmps)

    rows = []
    for label, lmp in scenarios.items():
        model = REPFinancialModel(lmp, rep_config, hours_per_step)

        adder = 0.0
        if cost_adder_by_scenario:
            adder = float(cost_adder_by_scenario.get(label, 0.0))

        ref = base_reference if forward_case == "base_forward" else None
        metrics = model.risk_metrics(
            reference_lmp_mean=ref,
            cost_adder_mwh=adder,
            pass_through=pass_through,
        )
        opt = model.optimal_hedge_ratio(
            reference_lmp_mean=ref,
            cost_adder_mwh=adder,
            pass_through=pass_through,
        )

        metrics.update({
            "scenario": label,
            "year": year,
            "optimal_h": opt["optimal_hedge_ratio"],
        })
        rows.append(metrics)

    summary = pd.DataFrame(rows)

    if "Base" in summary["scenario"].values:
        base_row = summary[summary["scenario"] == "Base"].iloc[0]
        for col in [
            "mean_lmp",
            "mean_margin_mwh",
            "horizon_profit_dollar",
            "cvar_5pct_dollar",
            "capital_breach",
            "loss_fraction",
        ]:
            if col in summary.columns:
                summary[f"delta_{col}"] = summary[col] - base_row[col]

    out = output_dir / f"rep_risk_summary_{year}_{forward_case}.csv"
    summary.to_csv(out, index=False)
    return summary


# =====================================================================
# 5. Lightweight visualisation helpers
# =====================================================================


def plot_pnl_distributions(
    base_lmp: pd.Series,
    dc_lmps: Dict[str, pd.Series],
    rep_config: REPConfig,
    year: int,
    hours_per_step: float,
    output_dir: Path,
    forward_case: str = "base_forward",
) -> None:
    """
    Plot per-MWh margin distributions and cumulative P&L paths.

    This function is kept lightweight so downstream notebooks/scripts can
    still use it. Heavy paper figures should live in analyze_rep_risk.py.
    """
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = REPFinancialModel(base_lmp, rep_config, hours_per_step)
    ref = base_model.lmp_mean if forward_case == "base_forward" else None

    scenarios = {"Base": base_lmp}
    scenarios.update(dc_lmps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, lmp in scenarios.items():
        model = REPFinancialModel(lmp, rep_config, hours_per_step)
        margin = model.compute_pnl_series(reference_lmp_mean=ref)
        cum = model.compute_cumulative_pnl(reference_lmp_mean=ref) / 1e6

        axes[0].hist(margin, bins=80, alpha=0.35, density=True, label=label)
        axes[1].plot(cum.values, lw=1.5, label=label)

    axes[0].axvline(0, color="red", ls="--", lw=1)
    axes[0].set_xlabel("Per-MWh margin ($/MWh)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Stylized LSE margin distribution — {year}")

    capital_floor = -rep_config.capital_mw_dollar / 1e6
    axes[1].axhline(0, color="gray", ls="--", lw=0.8)
    axes[1].axhline(capital_floor, color="red", ls=":", lw=1.0)
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Cumulative P&L ($M)")
    axes[1].set_title(f"Cumulative stylized LSE P&L — {year}")

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out = output_dir / f"pnl_distributions_{year}_{forward_case}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_hedge_optimization(
    base_lmp: pd.Series,
    dc_lmps: Dict[str, pd.Series],
    rep_config: REPConfig,
    year: int,
    hours_per_step: float,
    output_dir: Path,
    forward_case: str = "base_forward",
) -> None:
    """Plot selected risk metrics vs hedge ratio."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = REPFinancialModel(base_lmp, rep_config, hours_per_step)
    ref = base_model.lmp_mean if forward_case == "base_forward" else None

    scenarios = {"Base": base_lmp}
    scenarios.update(dc_lmps)

    ratios = np.arange(0, 1.01, 0.05)
    metric_keys = [
        ("mean_margin_mwh", "Mean margin ($/MWh)"),
        ("cvar_5pct_dollar", "5% CVaR ($)"),
        ("capital_breach", "Capital breach"),
        ("liquidity_requirement_to_capital_ratio", "48h liquidity / capital"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, ylabel) in zip(axes.flat, metric_keys):
        for label, lmp in scenarios.items():
            model = REPFinancialModel(lmp, rep_config, hours_per_step)
            vals = []
            for h in ratios:
                m = model.risk_metrics(hedge_ratio=h, reference_lmp_mean=ref)
                vals.append(m[key])
            ax.plot(ratios, vals, marker="o", ms=3, label=label)
        ax.set_xlabel("Hedge ratio")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Hedge sensitivity — {year}, {forward_case}", fontweight="bold")
    fig.tight_layout()

    out = output_dir / f"hedge_optimization_{year}_{forward_case}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# =====================================================================
# 6. Minimal CLI demo
# =====================================================================


def _demo(output_dir: Path) -> None:
    """Small synthetic demo for smoke testing."""
    rng = np.random.default_rng(123)
    idx = pd.date_range("2022-05-01", periods=1000, freq="3h")
    base = pd.Series(rng.normal(40, 20, len(idx)), index=idx).clip(-50, 500)
    dc = base + rng.gamma(2.0, 10.0, len(idx))

    cfg = REPConfig(
        served_load_mw=1000.0,
        capital_mw_dollar=50e6,
        retail_price_mwh=85.0,
        operating_cost_mwh=3.0,
        hedge_ratio=0.7,
        forward_premium_mwh=5.0,
    )

    out = run_scenario_comparison(
        base_lmp=base,
        dc_lmps={"DC": dc},
        year=2022,
        rep_config=cfg,
        output_dir=output_dir,
        forward_case="base_forward",
    )
    print(out[[
        "scenario",
        "forward_pricing_case",
        "mean_lmp",
        "mean_margin_mwh",
        "capital_breach",
        "max_drawdown_dollar",
        "liquidity_requirement_to_capital_ratio",
    ]])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stylized REP/LSE risk core smoke test")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for optional demo outputs.",
    )
    args = parser.parse_args()

    if args.demo:
        if args.output_dir is None:
            parser.error("--demo requires --output-dir")
        args.output_dir = args.output_dir.expanduser().resolve()
        _demo(args.output_dir)
    else:
        print("analysis_core.py is a library module. Run with --demo for a smoke test.")
