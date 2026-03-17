"""
Integration test session fixtures.

Ray is initialized ONCE per pytest session here.  Individual test files
must NOT call ray.init() themselves.  They declare a dependency on
the session-scoped ray_session fixture.

The Rust wheel (theseo_core) is built via `maturin develop` in
pytest_configure so that it is importable before test collection begins.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _wheel_is_current() -> bool:
    """Return True if the installed theseo_core wheel has all required symbols."""
    try:
        import theseo_core
        return hasattr(theseo_core.PyVoxelEnv, "filled_voxels")
    except ImportError:
        return False


def pytest_configure(config: "pytest.Config") -> None:  # noqa: F821
    """
    Build the theseo_core PyO3 wheel before any test module is collected.
    Rebuilds if the wheel is missing OR if it is outdated (missing required
    symbols such as filled_voxels).  If maturin is not available or the build
    fails, a warning is printed and tests relying on theseo_core will be
    skipped by their own importorskip / skipif guards.
    """
    if _wheel_is_current():
        return  # already up-to-date

    core_dir = Path(__file__).parent.parent.parent / "theseo_anysearch" / "core"
    result = subprocess.run(
        [sys.executable, "-m", "maturin", "develop"],
        cwd=str(core_dir),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        import warnings

        warnings.warn(
            "maturin develop failed — theseo_core integration tests will be skipped.\n"
            f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
        )


@pytest.fixture(scope="session")
def ray_session(tmp_path_factory):
    """
    Initialize Ray once for the whole test session and shut it down at the end.
    Kills any stale Ray instance (from previous crashed runs) before starting.
    """
    import ray as _ray
    # Ensure we start clean regardless of zombie instances from prior runs.
    try:
        _ray.shutdown()
    except Exception:
        pass
    ray_root = tmp_path_factory.mktemp("ray_session")
    _ray.init(
        num_cpus=1,
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        _temp_dir=str(Path(ray_root).joinpath("runtime")),
    )
    yield
    try:
        _ray.shutdown()
    except Exception:
        pass
