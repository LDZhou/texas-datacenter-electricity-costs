import importlib
import unittest
from types import SimpleNamespace

import pandas as pd


class FullYearComparabilityTests(unittest.TestCase):
    def test_static_report_flags_changed_line_capacities(self):
        module = importlib.import_module(
            "paper_pipeline.scripts.build_comparable_full_year"
        )
        baseline = SimpleNamespace(
            snapshots=pd.date_range("2023-01-01", periods=1464, freq="3h"),
            snapshot_weightings=pd.DataFrame({"objective": [3.0] * 1464}),
            lines=pd.DataFrame(
                {"s_nom": [100.0, 200.0], "s_nom_extendable": [False, False]},
                index=["a", "b"],
            ),
            generators=pd.DataFrame(
                {"p_nom": [50.0], "p_nom_extendable": [False]}, index=["g"]
            ),
            storage_units=pd.DataFrame(
                {"p_nom": [25.0], "p_nom_extendable": [False]}, index=["s"]
            ),
            stores=pd.DataFrame({"e_nom": []}),
            links=pd.DataFrame({"p_nom": []}),
        )
        full_year = SimpleNamespace(
            snapshots=pd.date_range("2023-01-01", periods=2920, freq="3h"),
            snapshot_weightings=pd.DataFrame({"objective": [3.0] * 2920}),
            lines=pd.DataFrame(
                {"s_nom": [100.0, 210.0], "s_nom_extendable": [False, False]},
                index=["a", "b"],
            ),
            generators=baseline.generators.copy(),
            storage_units=baseline.storage_units.copy(),
            stores=baseline.stores.copy(),
            links=baseline.links.copy(),
        )

        report = module.static_comparability_report(baseline, full_year)

        self.assertEqual(report["full_year_snapshots"], 2920)
        self.assertEqual(report["full_year_weighted_hours"], 8760.0)
        self.assertFalse(report["static_grid_comparable"])
        self.assertEqual(report["components"]["lines"]["changed_capacity_count"], 1)
        self.assertEqual(report["components"]["lines"]["baseline_total_nominal"], 300.0)
        self.assertEqual(report["components"]["lines"]["full_year_total_nominal"], 310.0)

    def test_persists_full_year_profile_at_the_full_horizon_path(self):
        module = importlib.import_module(
            "paper_pipeline.scripts.build_comparable_full_year"
        )
        received = {}
        network = SimpleNamespace(snapshots="full-year-snapshots")

        module.persist_full_year_profiles(
            network, "/run-root", 2023, "/data/TX_datacenters.xlsx",
            writer=lambda *args, **kwargs: received.update(args=args, kwargs=kwargs),
            validator=lambda path, snapshots: pd.DataFrame(
                {"DataCenter": [1.0] * 2920}
            ),
        )

        self.assertEqual(received["args"], (network, "/run-root", 2023,
                                             "/data/TX_datacenters.xlsx"))
        self.assertEqual(received["kwargs"], {"horizon": "full_year", "siting": True})

    def test_leap_year_is_not_longer_because_the_workflow_drops_february_29(self):
        module = importlib.import_module(
            "paper_pipeline.scripts.build_comparable_full_year"
        )

        self.assertEqual(module.expected_snapshots(2023), (2920, 8760.0))
        self.assertEqual(module.expected_snapshots(2020), (2920, 8760.0))


if __name__ == "__main__":
    unittest.main()
