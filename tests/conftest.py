"""Shared pytest configuration, temp-path isolation, and repository-wide fixtures.

This module provides the common pytest plumbing used across both unit and
integration tests:

- repository-local temporary directories
- isolated `.anysearch` registry state
- optional `theseo_core` wheel bootstrap for Rust-backed integration tests
- a session-scoped Ray fixture for integration suites that need it
"""

import os
import re
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Any

import pytest
import _pytest.pathlib as pytest_pathlib
import _pytest.tmpdir as pytest_tmpdir


_ORIGINAL_CLEANUP_DEAD_SYMLINKS = pytest_pathlib.cleanup_dead_symlinks


def _safe_cleanup_dead_symlinks(root: Path) -> None:
    try:
        _ORIGINAL_CLEANUP_DEAD_SYMLINKS(root)
    except PermissionError:
        # Some Windows workspaces expose transient temp directories that cannot
        # be enumerated during pytest session cleanup.
        return


pytest_pathlib.cleanup_dead_symlinks = _safe_cleanup_dead_symlinks
pytest_tmpdir.cleanup_dead_symlinks = _safe_cleanup_dead_symlinks

_TEMP_ROOT = Path.cwd().joinpath("pytest_tmp_root")


class LocalTmpPathFactory:
    """Lightweight tmp-path factory rooted inside the repository workspace.

    Parameters
    ----------
    base_root : Path
        Root directory under which per-test temporary directories are created.
    """
    def __init__(self, base_root: Path) -> None:
        self._base_root = base_root
        self._base_root.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def mktemp(self, basename: str, numbered: bool = True) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", basename).strip("._") or "tmp"
        if numbered:
            self._counter += 1
            path = self._base_root.joinpath(f"{safe_name}_{self._counter}")
            path.mkdir(parents=True, exist_ok=False)
            return path

        path = self._base_root.joinpath(safe_name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def getbasetemp(self) -> Path:
        return self._base_root


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest temp roots to stay inside the repository workspace.

    Parameters
    ----------
    config : pytest.Config
        Active pytest configuration object.

    Returns
    -------
    None
        This function mutates pytest and process temp-path settings in place.
    """
    temp_root = _TEMP_ROOT
    temp_root.mkdir(parents=True, exist_ok=True)

    session_root = temp_root.joinpath(
        f"session_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    config.option.basetemp = str(session_root)

    os.environ["TMP"] = str(temp_root.resolve())
    os.environ["TEMP"] = str(temp_root.resolve())

    if _wheel_is_current():
        return

    core_dir = Path(__file__).resolve().parent.parent.joinpath("theseo_anysearch", "core")
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
            "maturin develop failed; theseo_core integration tests may be skipped.\n"
            f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
        )


def _wheel_is_current() -> bool:
    """Return whether the installed ``theseo_core`` module has required symbols."""
    try:
        import theseo_core

        return hasattr(theseo_core.PyVoxelEnv, "filled_voxels")
    except ImportError:
        return False


@pytest.fixture(scope="session")
def tmp_path_factory() -> LocalTmpPathFactory:
    """Return a repository-local temporary path factory for tests.

    Returns
    -------
    LocalTmpPathFactory
        Session-scoped temporary path factory.
    """
    return LocalTmpPathFactory(
        _TEMP_ROOT.joinpath(f"fixture_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    )


@pytest.fixture()
def tmp_path(tmp_path_factory: LocalTmpPathFactory, request: pytest.FixtureRequest) -> Path:
    """Return a per-test temporary directory path.

    Parameters
    ----------
    tmp_path_factory : LocalTmpPathFactory
        Session-scoped temp path factory.
    request : pytest.FixtureRequest
        Active fixture request describing the current test node.

    Returns
    -------
    Path
        Unique temporary directory for the requesting test.
    """
    return tmp_path_factory.mktemp(request.node.name)


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the registry to a per-test temp path.

    Parameters
    ----------
    tmp_path : Path
        Per-test temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to update process environment variables.

    Returns
    -------
    Path
        The isolated registry file path for the active test.
    """
    tmp_registry = tmp_path.joinpath(".anysearch", "registry.yaml")
    tmp_registry.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ANYSEARCH_REGISTRY", str(tmp_registry))
    return tmp_registry


@pytest.fixture(scope="session")
def ray_session() -> Any:
    """Initialize Ray once for the test session and shut it down afterwards.

    Returns
    -------
    Any
        The active Ray session handle from ``ray.init()``.
    """
    import tempfile

    import ray as _ray

    try:
        _ray.shutdown()
    except Exception:
        pass

    ray_tmp = os.path.join(tempfile.gettempdir(), "anysearch_ray_session")
    os.makedirs(ray_tmp, exist_ok=True)
    session = _ray.init(
        num_cpus=4,
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        _temp_dir=ray_tmp,
    )
    yield session
    try:
        _ray.shutdown()
    except Exception:
        pass


MINIMAL_YAML = textwrap.dedent("""\
    env:
      stl_path: /tmp/test.stl
      scale: 1.0
      agent_count: 1
      max_steps: 100
      seed: 7

    training:
      algorithm: ppo
      model: voxel_encoder
      runner: local
      iterations: 5
      checkpoint_interval: 1
      output_dir: /tmp/out
      video_every: 2
      require_gpu: false

    anyscale:
      cluster_env: test-cluster
      compute_config: test-compute
      project: test-project

    algorithm_config:
      lr: 0.001
      gamma: 0.95
      train_batch_size: 512

    model_config:
      hidden_sizes: [128]
      activation: relu
""")


@pytest.fixture()
def minimal_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid YAML settings file and return its path."""
    p = tmp_path.joinpath("settings.yaml")
    p.write_text(MINIMAL_YAML)
    return p


# ---------------------------------------------------------------------------
# Trainer fixtures
# ---------------------------------------------------------------------------

class FakeAlgorithm:
    """
    Duck-typed stand-in for a ray.rllib Algorithm.

    train() returns a minimal RLlib-style result dict so that Trainer.train()
    can parse episode_reward_mean, etc.  save() and restore() write/read a
    small sentinel file so checkpoint round-trips can be verified.
    """

    def __init__(self) -> None:
        self._call_count = 0
        self._saved_path: str | None = None

    def train(self) -> dict[str, Any]:
        self._call_count += 1
        return {
            "episode_reward_mean": float(self._call_count),
            "episode_len_mean": 10.0,
            "episodes_total": self._call_count * 5,
        }

    def save(self, path: str) -> str:
        Path(path).mkdir(parents=True, exist_ok=True)
        sentinel = Path(path).joinpath("weights.sentinel")
        sentinel.write_text("ok")
        self._saved_path = path
        return path

    def restore(self, path: str) -> None:
        sentinel = Path(path).joinpath("weights.sentinel")
        if not sentinel.exists():
            raise FileNotFoundError(f"No weights at {path}")
        self._saved_path = path


@pytest.fixture()
def trainer_settings(tmp_path: Path) -> "Settings":  # noqa: F821
    """A Settings object pointing output_dir at a tmp directory."""
    import yaml
    from theseo_anysearch.settings import load_settings

    yaml_text = textwrap.dedent(f"""\
        env:
          stl_path: /tmp/test.stl
          scale: 1.0
          agent_count: 1
          max_steps: 50
          seed: 0

        training:
          algorithm: ppo
          model: voxel_encoder
          runner: local
          iterations: 4
          checkpoint_interval: 2
          output_dir: {tmp_path}
          video_every: 10
          require_gpu: false

        anyscale:
          cluster_env: x
          compute_config: y
          project: z

        algorithm_config:
          lr: 0.001
          gamma: 0.99
          train_batch_size: 64
          clip_param: 0.2
          num_sgd_iter: 4
          lambda_: 0.95
          kl_coeff: 0.2

        model_config:
          hidden_sizes: [64]
          activation: relu
    """)
    p = tmp_path.joinpath("trainer_settings.yaml")
    p.write_text(yaml_text)
    return load_settings(p)


@pytest.fixture()
def fake_ppo_trainer(trainer_settings: Any) -> "PPOTrainer":  # noqa: F821
    """
    A PPOTrainer wired with a FakeAlgorithm so no Ray installation is needed.
    """
    from theseo_anysearch.rllib.algorithms.ppo import PPOTrainer

    class _FakePPOTrainer(PPOTrainer):
        def _build_algorithm(self) -> FakeAlgorithm:
            return FakeAlgorithm()

    return _FakePPOTrainer(trainer_settings)
