import os
import re
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
    temp_root = _TEMP_ROOT
    temp_root.mkdir(parents=True, exist_ok=True)

    session_root = temp_root.joinpath(
        f"session_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    config.option.basetemp = str(session_root)

    os.environ["TMP"] = str(temp_root.resolve())
    os.environ["TEMP"] = str(temp_root.resolve())


@pytest.fixture(scope="session")
def tmp_path_factory() -> LocalTmpPathFactory:
    return LocalTmpPathFactory(
        _TEMP_ROOT.joinpath(f"fixture_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    )


@pytest.fixture()
def tmp_path(tmp_path_factory: LocalTmpPathFactory, request: pytest.FixtureRequest) -> Path:
    return tmp_path_factory.mktemp(request.node.name)


MINIMAL_YAML = textwrap.dedent("""\
    env:
      stl_path: /tmp/test.stl
      scale: 1.0
      agent_count: 2
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
    p = tmp_path / "settings.yaml"
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
        sentinel = Path(path) / "weights.sentinel"
        sentinel.write_text("ok")
        self._saved_path = path
        return path

    def restore(self, path: str) -> None:
        sentinel = Path(path) / "weights.sentinel"
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
          agent_count: 2
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
    p = tmp_path / "trainer_settings.yaml"
    p.write_text(yaml_text)
    return load_settings(p)


@pytest.fixture()
def fake_ppo_trainer(trainer_settings: Any) -> "PPOTrainer":  # noqa: F821
    """
    A PPOTrainer wired with a FakeAlgorithm so no Ray installation is needed.
    """
    from theseo_anysearch.rllib.trainer.ppo import PPOTrainer

    class _FakePPOTrainer(PPOTrainer):
        def _build_algorithm(self) -> FakeAlgorithm:
            return FakeAlgorithm()

    return _FakePPOTrainer(trainer_settings)
