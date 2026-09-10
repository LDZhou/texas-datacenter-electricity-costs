# Texas2k Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a public, self-contained Texas2k engineering module that consumes normalized PyPSA expansion assets and reproduces the collaborator's documented DC N-0/N-1 upgrade-cost workflow.

**Architecture:** `powerworld.texas2k` is an importable package within the repository, with focused modules for case parsing, input normalization, dispatch, contingencies, upgrades, exports, and the CLI. Immutable public inputs and reference outputs live beside the package; runtime outputs stay under ignored `results/powerworld/texas2k` or an explicit user path.

**Tech Stack:** Python 3.11, NumPy 1.26, pandas 2.2, SciPy 1.11, PYPOWER 5.1.19, pytest, uv

**Spec:** `docs/superpowers/specs/2026-09-10-texas2k-integration-design.md`

## Global Constraints

- Preserve the original PyPSA experiment behavior and technology scope.
- Do not commit the source ZIP, XLSX workbooks, credentials, licenses, environment files, personal absolute paths, or private data.
- Use semantic module and file names; record the supplied date-prefixed names only in provenance documentation.
- Use a stable zero-based MATPOWER `branch_id` for every upgrade and export join.
- Record and warn on non-convergent, islanded, non-finite, and residual cases; allow runs to continue.
- Describe outputs as a DC engineering screening heuristic with fixed upgrade rules, not an AC or optimized transmission plan.
- Run production commands from the repository root with Python 3.11.
- Write tests before production behavior and observe every new test fail for the intended reason.

---

### Task 1: Public Inputs and Dependency Lock

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `powerworld/__init__.py`
- Create: `powerworld/texas2k/__init__.py`
- Create: `powerworld/texas2k/inputs/case/texas2k_series25_summer_peak.m`
- Create: `powerworld/texas2k/inputs/baseline/buses.csv`
- Create: `powerworld/texas2k/inputs/datacenters/projects.csv`
- Create: `powerworld/texas2k/inputs/datacenters/bus_matches.csv`
- Create: `powerworld/texas2k/inputs/pypsa/generation_only_2023.csv`
- Create: `powerworld/texas2k/inputs/pypsa/generation_storage_2023.csv`

**Interfaces:**
- Consumes: the locally normalized, private collaborator source tree outside Git
- Produces: immutable CSV/MATPOWER inputs and importable `powerworld.texas2k`

- [ ] **Step 1: Add package import test**

```python
def test_texas2k_package_imports():
    import powerworld.texas2k

    assert powerworld.texas2k.__version__ == "1.0.0"
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest powerworld/texas2k/tests/test_package.py -q`
Expected: collection fails because `powerworld.texas2k` does not exist.

- [ ] **Step 3: Add package files, PYPOWER pin, and sanitized inputs**

Set `__version__ = "1.0.0"`, add `pypower==5.1.19` to root dependencies, and refresh the lock with `uv lock`. Convert each required workbook sheet to plain UTF-8 CSV, filter the two PyPSA tables to `year=2023`, `case=all2030_full_tx`, `location=ALL_2030`, and required asset types, then copy the already-corrected bus table, bus matches, and MATPOWER case. Preserve source numeric precision and column names.

- [ ] **Step 4: Verify package and dependency**

Run: `uv sync --frozen --no-install-project && uv run pytest powerworld/texas2k/tests/test_package.py -q`
Expected: one passing test and `uv run python -c "from pypower.api import rundcpf"` exits 0.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock powerworld/__init__.py powerworld/texas2k/__init__.py powerworld/texas2k/inputs powerworld/texas2k/tests/test_package.py
git commit -m "build: add Texas2k inputs and PYPOWER dependency"
```

### Task 2: MATPOWER Case Parsing and Branch Identity

**Files:**
- Create: `powerworld/texas2k/case.py`
- Create: `powerworld/texas2k/tests/test_case.py`
- Create: `powerworld/texas2k/tests/fixtures/tiny_case.m`

**Interfaces:**
- Produces: `load_matpower_case(path: Path) -> dict[str, object]`
- Produces: `run_dc_power_flow(case: dict[str, object]) -> tuple[dict[str, object], bool]`
- Produces: `branch_loading(result: dict[str, object], ratings: ndarray | None = None) -> ndarray`

- [ ] **Step 1: Write parser and branch-loading tests**

```python
def test_load_case_preserves_matrix_rows_and_base_mva(tmp_path):
    case = load_matpower_case(FIXTURES / "tiny_case.m")
    assert case["baseMVA"] == 100.0
    assert case["bus"].shape == (3, 13)
    assert case["gen"].shape == (1, 21)
    assert case["branch"].shape == (3, 13)

def test_branch_loading_uses_absolute_pf_and_zero_for_unrated_branch():
    result = {"branch": np.array([[1, 2, 0, 0, 0, 100, 0, 0, 0, 0, 1, -30, 30], [2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 20, -20]], dtype=float)}
    np.testing.assert_allclose(branch_loading(result), [30.0, 0.0])
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest powerworld/texas2k/tests/test_case.py -q`
Expected: import fails because `powerworld.texas2k.case` is absent.

- [ ] **Step 3: Implement strict MATPOWER parsing and DC wrapper**

Parse `mpc.baseMVA`, `bus`, `gen`, `branch`, and optional `gencost`; raise `ValueError` with the missing matrix name for malformed input. Deep-copy before `rundcpf`, suppress solver table output through `ppoption(VERBOSE=0, OUT_ALL=0)`, and calculate loading from `abs(PF) / RATE_A` with unrated branches set to zero.

- [ ] **Step 4: Verify unit and bundled-case contracts**

Run: `uv run pytest powerworld/texas2k/tests/test_case.py -q`
Expected: tests pass, including bundled dimensions of 2,751 buses, 1,099 generators, and 5,344 branches.

- [ ] **Step 5: Commit**

```bash
git add powerworld/texas2k/case.py powerworld/texas2k/tests/test_case.py powerworld/texas2k/tests/fixtures/tiny_case.m
git commit -m "feat: parse and solve Texas2k MATPOWER cases"
```

### Task 3: Input Validation and PyPSA Normalization

**Files:**
- Create: `powerworld/texas2k/data.py`
- Create: `powerworld/texas2k/tests/test_inputs.py`

**Interfaces:**
- Produces: `load_assets(path: Path) -> pandas.DataFrame`
- Produces: `normalize_pypsa_assets(source: Path, destination: Path, *, year: int, case: str, location: str, include_storage: bool) -> pandas.DataFrame`
- Produces: `load_datacenter_injections(projects_path: Path, matches_path: Path) -> tuple[pandas.DataFrame, dict[str, float | int]]`

- [ ] **Step 1: Write tests for schema, filtering, and load totals**

```python
def test_load_assets_rejects_missing_coordinates(tmp_path):
    path = tmp_path / "assets.csv"
    pd.DataFrame([{"year": 2023, "case": "c", "location": "l", "asset_type": "generation", "carrier": "OCGT", "bus": "b", "bus_x": np.nan, "bus_y": 31.0, "added_mw": 1.0}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite bus_x and bus_y"):
        load_assets(path)

def test_datacenter_injections_keep_positive_matched_rows(fixture_paths):
    rows, stats = load_datacenter_injections(*fixture_paths)
    assert rows["total_mw"].sum() == pytest.approx(15.0)
    assert stats == {"source_rows": 4, "matched_rows": 3, "injected_rows": 2, "injected_mw": 15.0}
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest powerworld/texas2k/tests/test_inputs.py -q`
Expected: import fails because input functions are absent.

- [ ] **Step 3: Implement canonical input boundary**

Require the nine canonical columns, coerce year and numeric columns, reject non-finite coordinates and negative capacities, and allow only generation/storage assets. Join project rows and match rows by stable row order only after checking equal length; calculate `total_mw`; keep `rescue_status == "RESCUABLE"` or `match_status == "MATCHED"` with positive capacity; return source, matched, injected row counts and MW.

- [ ] **Step 4: Verify tests and public input totals**

Run: `uv run pytest powerworld/texas2k/tests/test_inputs.py -q`
Expected: tests pass and integration assertions report 40,293.907 MW source projects, 39,864.607 MW injected, 32,942.9 MW generation-only assets, and 39,992.7 MW generation-plus-storage assets.

- [ ] **Step 5: Commit**

```bash
git add powerworld/texas2k/data.py powerworld/texas2k/tests/test_inputs.py
git commit -m "feat: validate Texas2k and PyPSA input tables"
```

### Task 4: Scenario Assembly and Dispatch

**Files:**
- Create: `powerworld/texas2k/dispatch.py`
- Create: `powerworld/texas2k/tests/test_dispatch.py`

**Interfaces:**
- Consumes: validated case, bus geography, asset table, and DC injection table
- Produces: `assemble_scenario(base_case: dict[str, object], *, assets: DataFrame, bus_geography: DataFrame, dc_loads: DataFrame | None, portfolio: str) -> tuple[dict[str, object], dict[str, float | int]]`
- Produces: `redispatch_generation(case: dict[str, object], *, max_utilization: float = 0.98) -> tuple[dict[str, object], dict[str, float]]`

- [ ] **Step 1: Write scenario behavior tests**

```python
def test_assemble_scenario_adds_each_dc_load_once(tiny_inputs):
    scenario, info = assemble_scenario(**tiny_inputs, portfolio="generation")
    assert scenario["bus"][:, PD].sum() == pytest.approx(tiny_inputs["base_case"]["bus"][:, PD].sum() + 15.0)
    assert info["dc_applied_mw"] == pytest.approx(15.0)

def test_redispatch_keeps_non_slack_generation_within_capacity(tiny_dispatched_case):
    info = redispatch_generation(tiny_dispatched_case, max_utilization=0.98)
    assert info["unserved_generation_mw"] == pytest.approx(0.0)
    assert np.all(tiny_dispatched_case["gen"][:, PG] <= tiny_dispatched_case["gen"][:, PMAX] + 1e-9)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest powerworld/texas2k/tests/test_dispatch.py -q`
Expected: import fails because dispatch functions are absent.

- [ ] **Step 3: Port the two supplied portfolio rules**

Implement generation-only nearest-generator spreading for OCGT and new rows for renewable assets. Implement generation-plus-storage zone-aware gas matching and capacity-factor dispatch from the supplied `0731_10_genstorage_base.py` behavior. Apply each data-center row once at its selected Texas2k bus, preserve reactive-load calculations for metadata, and keep DC power-flow semantics.

- [ ] **Step 4: Verify scenario tests**

Run: `uv run pytest powerworld/texas2k/tests/test_dispatch.py -q`
Expected: tests pass for no-data-center and injected scenarios, capacity bounds, duplicate-address handling, and portfolio selection.

- [ ] **Step 5: Commit**

```bash
git add powerworld/texas2k/dispatch.py powerworld/texas2k/tests/test_dispatch.py
git commit -m "feat: assemble Texas2k generation and storage scenarios"
```

### Task 5: N-0 and N-1 Screening with Warning-Only Failures

**Files:**
- Create: `powerworld/texas2k/contingency.py`
- Create: `powerworld/texas2k/tests/test_contingency.py`

**Interfaces:**
- Produces: `screen_n0(case: dict[str, object]) -> ScreeningResult`
- Produces: `scan_n1(case: dict[str, object], *, solver: Callable = run_dc_power_flow, progress_every: int = 1000) -> ScreeningResult`
- `ScreeningResult` contains `n0_loading`, `worst_loading`, `worst_contingency`, and `skipped` records.

- [ ] **Step 1: Write finite-flow and skipped-contingency tests**

```python
def test_scan_n1_records_nonfinite_solver_output_and_continues(tiny_case):
    calls = iter([(finite_result, True), (nan_result, True), (higher_result, True), (finite_result, False)])
    result = scan_n1(tiny_case, solver=lambda _: next(calls), progress_every=0)
    assert [row.reason for row in result.skipped] == ["non_finite_flow", "not_converged"]
    np.testing.assert_allclose(result.worst_loading, [40.0, 25.0, 0.0])

def test_scan_n1_warns_once_when_any_contingency_is_skipped(tiny_case):
    with pytest.warns(RuntimeWarning, match="2 N-1 contingencies skipped"):
        scan_n1(tiny_case, solver=configured_solver, progress_every=0)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest powerworld/texas2k/tests/test_contingency.py -q`
Expected: import fails because screening functions are absent.

- [ ] **Step 3: Implement screening**

Run one intact solve, then one copied-case solve per active branch outage. Skip and record exceptions, false convergence status, missing branch arrays, and any non-finite active-branch PF value. Exclude the open branch from its contingency loading, update worst values only from finite results, preserve the triggering branch ID, and issue one aggregate warning after the scan.

- [ ] **Step 4: Verify tests**

Run: `uv run pytest powerworld/texas2k/tests/test_contingency.py -q`
Expected: tests pass with warnings captured and no global warning suppression.

- [ ] **Step 5: Commit**

```bash
git add powerworld/texas2k/contingency.py powerworld/texas2k/tests/test_contingency.py
git commit -m "feat: add resilient Texas2k N-0 and N-1 screening"
```

### Task 6: Upgrade Rules and Parallel-Circuit-Safe Exports

**Files:**
- Create: `powerworld/texas2k/upgrades.py`
- Create: `powerworld/texas2k/exports.py`
- Create: `powerworld/texas2k/tests/test_upgrades.py`
- Create: `powerworld/texas2k/tests/test_exports.py`

**Interfaces:**
- Produces: `classify_loading(percent: float) -> str`
- Produces: `estimate_upgrade_cost(row: Mapping[str, object]) -> tuple[float, str]`
- Produces: `apply_upgrades(case: dict[str, object], targets: DataFrame, *, tier_column: str) -> tuple[dict[str, object], DataFrame]`
- Produces: `build_normalized_exports(upgrades: DataFrame, branches: DataFrame, *, portfolio: str, criterion: str, recovery: str) -> tuple[DataFrame, DataFrame]`

- [ ] **Step 1: Write cost and stable-identity regression tests**

```python
def test_line_cost_uses_voltage_distance_and_detour():
    cost, kind = estimate_upgrade_cost({"tier": "T4_light", "component": "line", "high_kv": 230.0, "distance_mile": 10.0})
    assert kind == "line"
    assert cost == pytest.approx(17.55)

def test_allbranch_export_marks_only_selected_parallel_circuit():
    branches = pd.DataFrame([{"branch_id": 7, "from_bus": 1, "to_bus": 2}, {"branch_id": 8, "from_bus": 1, "to_bus": 2}])
    upgrades = pd.DataFrame([{"branch_id": 8, "from_bus": 1, "to_bus": 2, "capex_musd": 4.0}])
    _, all_branches = build_normalized_exports(upgrades, branches, portfolio="generation-storage", criterion="n1", recovery="NEW")
    assert all_branches.set_index("branch_id")["is_upgraded"].to_dict() == {7: False, 8: True}
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest powerworld/texas2k/tests/test_upgrades.py powerworld/texas2k/tests/test_exports.py -q`
Expected: imports fail because upgrade and export modules are absent.

- [ ] **Step 3: Port fixed upgrade/cost rules and implement safe exports**

Port the exact tier edges, rating factors, impedance factors, three-times-rating cap, 1.30 detour factor, voltage buckets, line costs, transformer costs, zone mapping, and labels. Use `branch_id` for all indexing and joins. Export `cost_musd` only on the exact upgraded branch and derive summary cost exclusively from upgrade detail.

- [ ] **Step 4: Verify tests and supplied-result reconciliation**

Run: `uv run pytest powerworld/texas2k/tests/test_upgrades.py powerworld/texas2k/tests/test_exports.py -q`
Expected: tests pass, including duplicate endpoint pairs and detail-to-summary cost differences below 0.05 million USD.

- [ ] **Step 5: Commit**

```bash
git add powerworld/texas2k/upgrades.py powerworld/texas2k/exports.py powerworld/texas2k/tests/test_upgrades.py powerworld/texas2k/tests/test_exports.py
git commit -m "fix: export Texas2k upgrades by stable branch identity"
```

### Task 7: Unified CLI, Reference Outputs, and Smoke Run

**Files:**
- Create: `powerworld/texas2k/cli.py`
- Create: `powerworld/texas2k/workflow.py`
- Create: `powerworld/texas2k/tests/test_cli.py`
- Create: `powerworld/texas2k/reference_outputs/engineering_summary.csv`
- Create: `powerworld/texas2k/reference_outputs/generation/*`
- Create: `powerworld/texas2k/reference_outputs/generation_storage/*`

**Interfaces:**
- Produces: `run_workflow(config: RunConfig) -> dict[str, Path]`
- Produces CLI subcommands `validate-inputs`, `normalize-assets`, `run`, and `export`

- [ ] **Step 1: Write CLI isolation and smoke tests**

```python
def test_cli_run_writes_only_to_requested_output(tmp_path):
    result = runner.invoke(app, ["run", "--input-dir", str(FIXTURES), "--output-dir", str(tmp_path), "--portfolio", "generation", "--criterion", "n0"])
    assert result.exit_code == 0
    assert (tmp_path / "generation" / "n0" / "summary_n0.csv").exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["generation"]

def test_cli_reports_residuals_without_failing(tmp_path):
    result = runner.invoke(app, configured_residual_arguments(tmp_path))
    assert result.exit_code == 0
    assert "residual violations" in result.output
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest powerworld/texas2k/tests/test_cli.py -q`
Expected: import fails because CLI and workflow modules are absent.

- [ ] **Step 3: Implement orchestration and copy reference outputs**

Use `argparse` so no additional runtime dependency is needed. Resolve default inputs relative to `powerworld/texas2k`, create a scenario-specific output directory, write files atomically within it, never unlink unrelated content, warn when residuals remain, and return exit zero when inputs and solves are otherwise valid. Copy the four supplied scenario summary/detail/branch/skipped tables into portfolio/criterion reference directories and create the eight-row engineering summary for NEW and ALL recovery modes.

- [ ] **Step 4: Verify CLI and synthetic end-to-end flow**

Run: `uv run pytest powerworld/texas2k/tests/test_cli.py -q && uv run python -m powerworld.texas2k.cli validate-inputs`
Expected: tests pass; validation reports all required files, exact row/capacity totals, case dimensions, and no private paths.

- [ ] **Step 5: Commit**

```bash
git add powerworld/texas2k/cli.py powerworld/texas2k/workflow.py powerworld/texas2k/tests/test_cli.py powerworld/texas2k/reference_outputs
git commit -m "feat: add reproducible Texas2k command-line workflow"
```

### Task 8: Documentation, Attribution, and Public-Tree Guard

**Files:**
- Modify: `README.md`
- Modify: `powerworld/README.md`
- Create: `powerworld/texas2k/README.md`
- Create: `powerworld/texas2k/NOTICE.md`
- Modify: `PROVENANCE.md`
- Modify: `scripts/check_public_tree.py`
- Create: `tests/test_public_tree.py`

**Interfaces:**
- Consumes: all module commands and reference values
- Produces: public reproduction guide and scanner coverage for CSV inputs

- [ ] **Step 1: Write public-tree behavior tests**

```python
def test_scanner_checks_csv_without_printing_secret_value(tmp_repo):
    secret = "A" * 32
    (tmp_repo / "sample.csv").write_text(f"api_key,{secret}\n")
    completed = run_scanner(tmp_repo)
    assert completed.returncode == 1
    assert "sample.csv" in completed.stdout
    assert secret not in completed.stdout
```

- [ ] **Step 2: Run test and verify RED**

Run: `uv run pytest tests/test_public_tree.py -q`
Expected: scanner returns zero because CSV files are not scanned.

- [ ] **Step 3: Extend scanner and write final documentation**

Include `.csv` in text scanning, make the root overrideable only for tests, and keep findings limited to path, reason, and line number. Update the root flow to `PyPSA -> canonical asset CSV -> Texas2k DC screening -> engineering cost -> residential adder`. Replace the PowerWorld placeholder with an accurate method description. Document input schemas, all commands, scenario mapping, runtime expectations, output columns, validation targets, residual warnings, data-count definitions, Texas A&M sources, source-script mapping, and collaborator-level attribution.

- [ ] **Step 4: Run full verification**

Run:

```bash
uv run pytest tests powerworld/texas2k/tests -q
uv run python scripts/check_public_tree.py
uv run python -m powerworld.texas2k.cli validate-inputs
uv run python -m powerworld.texas2k.cli run --input-dir powerworld/texas2k/tests/fixtures/smoke_inputs --output-dir /tmp/texas2k-smoke --portfolio generation-storage --criterion n1
git diff --check
```

Expected: all tests pass, scanner reports zero findings, input validation matches documented totals, smoke run exits zero while recording expected warning-only skipped/residual cases, and Git reports no whitespace errors.

- [ ] **Step 5: Commit**

```bash
git add README.md powerworld/README.md powerworld/texas2k/README.md powerworld/texas2k/NOTICE.md PROVENANCE.md scripts/check_public_tree.py tests/test_public_tree.py
git commit -m "docs: connect PyPSA and Texas2k reproduction workflows"
```
