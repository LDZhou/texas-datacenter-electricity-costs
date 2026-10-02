"""Exercise the public paper command assembly without opening networks."""
import sys
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_paper


class EntrypointTests(unittest.TestCase):
    def test_analysis_writes_the_canonical_analysis_view_and_attribution_tables(self):
        commands = run_paper.build_commands(
            "analysis", run_root=Path("results/paper"), years=[2019, 2023]
        )

        rendered = [" ".join(map(str, command)) for command in commands]
        self.assertIn("paper_pipeline.scripts.summarize_revision", rendered[0])
        self.assertIn("results/paper/analysis_view", rendered[0])
        self.assertIn("--groups seasonal siting", rendered[0])
        self.assertIn("results/paper/results/seasonal/{y}/none_full_tx/network.nc", rendered[1])
        self.assertIn("--results-root results/paper/results", rendered[2])
        self.assertIn("--results-root results/paper/results", rendered[3])
        self.assertTrue(any("paper_pipeline.scripts.compute_capped_lmp_adders" in line for line in rendered))
        self.assertTrue(any("paper_pipeline.scripts.compute_adder_factor_decomposition" in line for line in rendered))
        self.assertTrue(any("paper_pipeline.scripts.compute_graded_unserved" in line for line in rendered))
        self.assertTrue(all("--years 2019 2023" in line for line in rendered[1:]))

    def test_case_command_uses_a_frozen_common_baseline_and_persisted_profile(self):
        command = run_paper.build_commands(
            "run-case", run_root=Path("results/paper"), years=[2023], group="seasonal", index=4
        )[0]

        self.assertEqual(command[:3], [sys.executable, "-m", "paper_pipeline.scripts.run_revision_case"])
        self.assertIn("--run-root", command)
        self.assertIn("--historical-root", command)
        self.assertEqual(command[command.index("--group") + 1], "seasonal")

    def test_matrix_command_has_no_solver_side_effect(self):
        command = run_paper.build_commands("matrix", run_root=Path("results/paper"), years=[])[0]

        self.assertEqual(command[:3], [sys.executable, "-m", "paper_pipeline.scripts.revision_matrix"])
        self.assertEqual(command[-1], "results/paper/manifests/experiment_matrix.json")

    def test_matrix_storage_group_matches_analysis_layout(self):
        from paper_pipeline.scripts.revision_matrix import build_matrix

        matrix = build_matrix()
        self.assertEqual({case["group"] for case in matrix["central"] + matrix["weather"]}, {"seasonal"})
        self.assertTrue(all(case["output"].startswith("results/seasonal/")
                            for case in matrix["central"] + matrix["weather"]))

    def test_rep_command_uses_the_authoritative_module_and_cli(self):
        commands = run_paper.build_commands("rep", run_root=Path("results/paper"), years=[2023])
        command = commands[1]
        self.assertEqual(command[:3], [sys.executable, "-m", "paper_pipeline.scripts.analyze_results"])
        help_output = subprocess.run(command[:3] + ["--help"], text=True,
                                     capture_output=True, check=True).stdout
        for argument in command[3:]:
            if argument.startswith("--"):
                self.assertIn(argument, help_output)
        self.assertEqual(command[command.index("--n-sims") + 1], "5000")
        self.assertEqual(command[command.index("--seed") + 1], "123")

    def test_retail_view_contains_only_the_published_three_case_groups(self):
        import tempfile
        from collections import defaultdict
        from paper_pipeline.scripts.prepare_retail_inputs import prepare_view
        from paper_pipeline.scripts.analyze_results import enumerate_experiments
        years = [2019, 2020, 2021, 2022, 2023]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for year in years:
                for name in ("none_dispatch", "none_full_tx", "none_generation_storage",
                             "all2030_dispatch", "all2030_full_tx", "all2030_generation_storage"):
                    (root / "results" / "seasonal" / str(year) / name).mkdir(parents=True)
            view = prepare_view(root, years)
            self.assertEqual(prepare_view(root, years), view)
            groups = defaultdict(list)
            for experiment in enumerate_experiments(view, years, ["none", "all2030"]):
                groups[(experiment["scenario"], experiment["mode"])].append(experiment["year"])
            self.assertEqual(set(groups), {("none", "dispatch"), ("all2030", "full_tx"),
                                           ("all2030", "generation_storage")})
            self.assertTrue(all(candidates == years for candidates in groups.values()))

if __name__=='__main__':unittest.main()
