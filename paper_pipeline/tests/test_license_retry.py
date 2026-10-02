import subprocess
from types import SimpleNamespace

from paper_pipeline.scripts.run_revision_case import is_license_failure, run_with_license_retry


def _runner(outcomes):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        code, text = outcomes.pop(0)
        return SimpleNamespace(returncode=code, stdout=text)
    run.calls = calls
    return run


def test_license_overage_is_retried_then_succeeds():
    slept = []
    runner = _runner([(1, "gurobipy.GurobiError: Overage for too long, 3 active sessions"), (0, "solver status: ok")])
    attempts = run_with_license_retry(["x"], retries=3, wait_seconds=7, runner=runner, sleep=slept.append)
    assert attempts == 2 and slept == [7] and len(runner.calls) == 2


def test_modelling_failure_is_not_retried():
    runner = _runner([(1, "RuntimeError: solver did not prove optimality: status=ok, condition=suboptimal")])
    assert run_with_license_retry(["x"], retries=3, wait_seconds=1, runner=runner, sleep=lambda s: None) == -1
    assert len(runner.calls) == 1


def test_gives_up_after_the_retry_budget():
    runner = _runner([(1, "Overage for too long")] * 3)
    assert run_with_license_retry(["x"], retries=2, wait_seconds=1, runner=runner, sleep=lambda s: None) == -1
    assert len(runner.calls) == 3


def test_marker_detection():
    assert is_license_failure("gurobipy.GurobiError: Overage for too long, 3 active sessions")
    assert not is_license_failure("Time limit reached")
