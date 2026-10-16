"""CI reuses pip downloads while still resolving and installing every job's stack.

The cache inputs include the project metadata and every committed constraints file,
including the compatibility floor. Checkout must precede setup-python so those inputs
can be hashed. A cache hit must never bypass installation: nightly jobs still resolve
the newest allowed versions, and the floor job still installs its constrained stack.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).absolute().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
CACHED_JOBS = ("test", "mlx-smoke", "torchao-contract", "pytorch-smoke", "transformers-floor")


def _steps(job_name: str) -> list[dict[str, Any]]:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))["jobs"][job_name]["steps"]


def _python_setup(job_name: str) -> tuple[int, dict[str, Any]]:
    setups = [
        (index, step)
        for index, step in enumerate(_steps(job_name))
        if str(step.get("uses", "")).startswith("actions/setup-python@")
    ]
    assert len(setups) == 1, f"{job_name}: expected one Python setup, got {setups}"
    return setups[0]


def _project_installs(job_name: str) -> list[tuple[int, dict[str, Any]]]:
    installs = []
    for index, step in enumerate(_steps(job_name)):
        command = str(step.get("run", ""))
        if re.search(r"\bpip\s+install\b", command):
            arguments = shlex.split(command)
            if any(argument == "." or argument.startswith(".[") for argument in arguments):
                installs.append((index, step))
    assert installs, f"{job_name}: missing project installation"
    return installs


@pytest.mark.parametrize("job_name", CACHED_JOBS)
class TestPipDownloadCache:
    def test_project_jobs_enable_pip_download_caching(self, job_name: str) -> None:
        _, setup = _python_setup(job_name)
        assert setup.get("with", {}).get("cache") == "pip", (
            f"{job_name}: setup-python must cache pip downloads"
        )

    def test_cache_inputs_cover_project_and_all_constraints(self, job_name: str) -> None:
        _, setup = _python_setup(job_name)
        dependency_paths = setup.get("with", {}).get("cache-dependency-path", "")
        assert isinstance(dependency_paths, str), f"{job_name}: cache inputs must be paths"
        patterns = [line.strip() for line in dependency_paths.splitlines() if line.strip()]
        matched = {path for pattern in patterns for path in ROOT.glob(pattern) if path.is_file()}
        constraints = set((ROOT / ".github" / "constraints").glob("*.txt"))
        assert constraints, "the constraint check must cover committed constraints files"
        required = {ROOT / "pyproject.toml", *constraints}
        missing = sorted(str(path.relative_to(ROOT)) for path in required - matched)
        assert not missing, f"{job_name}: pip cache does not hash dependency inputs: {missing}"

    def test_checkout_precedes_python_cache_restore(self, job_name: str) -> None:
        setup_index, _ = _python_setup(job_name)
        checkouts = [
            index
            for index, step in enumerate(_steps(job_name))
            if str(step.get("uses", "")).startswith("actions/checkout@")
        ]
        assert checkouts and min(checkouts) < setup_index, (
            f"{job_name}: checkout must make cache dependency inputs available before setup-python"
        )

    def test_project_install_follows_cache_restore(self, job_name: str) -> None:
        setup_index, _ = _python_setup(job_name)
        for install_index, step in _project_installs(job_name):
            assert setup_index < install_index, (
                f"{job_name}: {step.get('name', 'project install')} must follow setup-python"
            )

    def test_cache_hit_cannot_bypass_project_install(self, job_name: str) -> None:
        for _, step in _project_installs(job_name):
            assert "if" not in step, f"{job_name}: project installation must be unconditional"
            assert "cache-hit" not in str(step["run"]), (
                f"{job_name}: cache hits must not gate installation inside the shell"
            )
