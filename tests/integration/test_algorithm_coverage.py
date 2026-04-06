"""
Integration tests verifying that every active algorithm is both trainable and
tunable through TuneRunner.

Active algorithms (theseo_core required for all):
  ppo                 — single-agent PPO (legacy RLlib stack)
  sac                 — single-agent discrete SAC
  dqn                 — single-agent DQN
  rainbow             — single-agent Rainbow DQN
  multi_agent_voxel_ppo — multi-agent PPO (PettingZoo parallel API)

Inactive (stub — require continuous Box action space, not yet supported):
  td3, ddpg

Bugs covered:
  - _experiment_trainable applied a hardcoded _PPO_FIELDS filter that
    silently dropped algorithm-specific search-space params (SAC: tau, n_step;
    DQN: n_step, num_atoms) and raised Pydantic ValidationError when PPO-only
    fields were passed to a non-PPO config via model_copy().
    Fix: filter by model_fields of the actual config instance.  See spec/bugs.md.

  - All sweep tests use num_env_runners=0 (inline rollouts) to avoid the
    SIGSEGV-kills-raylet placement-group bug.  See spec/bugs.md §2.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Per-algorithm YAML templates
# ---------------------------------------------------------------------------
# All use geometry_boxes so no STL file is needed.
# train_batch_size is kept small (32) and iterations=1 so tests are fast.
# search_space uses only `lr` (present on every AlgorithmConfig) unless the
# test is specifically exercising algorithm-specific field sweeping.
# ---------------------------------------------------------------------------

_BASE_ENV = textwrap.dedent("""\
    env:
      agent_count: 1
      max_steps: 5
      geometry_boxes:
        - [10, 10, 10, 22, 22, 22]
""")

_BASE_TRAINING = textwrap.dedent("""\
    training:
      runner: local
      iterations: 1
      checkpoint_interval: 100
      trajectory_every: 100
      video_every: 100
      require_gpu: false
      num_env_runners: 0
""")

_BASE_TUNE = textwrap.dedent("""\
    tune_config:
      scheduler: asha
      num_samples: 2
      metric: episode_reward_mean
      mode: max
      max_concurrent: 1
      search_space:
        algorithm_config:
          lr: {{choice: [1.0e-4, 3.0e-4]}}

    mlflow: {{}}
""")

_BASE_MODEL = textwrap.dedent("""\
    model_config:
      hidden_sizes: [16]
""")

_YAML_PPO = textwrap.dedent("""\
    experiment:
      name: cov-ppo
      output_dir: {output_dir}
      seed: 0
""") + _BASE_ENV + textwrap.dedent("""\
    training:
      algorithm: ppo
      runner: local
      iterations: 1
      checkpoint_interval: 100
      trajectory_every: 100
      video_every: 100
      require_gpu: false
      num_env_runners: 0

    algorithm_config:
      lr: 1.0e-3
      gamma: 0.99
      train_batch_size: 32
      clip_param: 0.2
      num_sgd_iter: 1
      lambda_: 0.95
      kl_coeff: 0.2
      minibatch_size: 16

""") + _BASE_MODEL + _BASE_TUNE

_YAML_SAC = textwrap.dedent("""\
    experiment:
      name: cov-sac
      output_dir: {output_dir}
      seed: 0
""") + _BASE_ENV + textwrap.dedent("""\
    training:
      algorithm: sac
      runner: local
      iterations: 1
      checkpoint_interval: 100
      trajectory_every: 100
      video_every: 100
      require_gpu: false
      num_env_runners: 0

    algorithm_config:
      lr: 1.0e-3
      gamma: 0.99
      train_batch_size: 32
      tau: 0.005
      n_step: 1
      warmup_steps: 0

""") + _BASE_MODEL + _BASE_TUNE

_YAML_DQN = textwrap.dedent("""\
    experiment:
      name: cov-dqn
      output_dir: {output_dir}
      seed: 0
""") + _BASE_ENV + textwrap.dedent("""\
    training:
      algorithm: dqn
      runner: local
      iterations: 1
      checkpoint_interval: 100
      trajectory_every: 100
      video_every: 100
      require_gpu: false
      num_env_runners: 0

    algorithm_config:
      lr: 1.0e-3
      gamma: 0.99
      train_batch_size: 32
      n_step: 1
      num_atoms: 1
      dueling: true
      double_q: true
      noisy: false
      replay_buffer_capacity: 1000
      warmup_steps: 0

""") + _BASE_MODEL + _BASE_TUNE

_YAML_RAINBOW = textwrap.dedent("""\
    experiment:
      name: cov-rainbow
      output_dir: {output_dir}
      seed: 0
""") + _BASE_ENV + textwrap.dedent("""\
    training:
      algorithm: rainbow
      runner: local
      iterations: 1
      checkpoint_interval: 100
      trajectory_every: 100
      video_every: 100
      require_gpu: false
      num_env_runners: 0

    algorithm_config:
      lr: 1.0e-3
      gamma: 0.99
      train_batch_size: 32
      n_step: 3
      num_atoms: 11
      noisy: true
      dueling: true
      double_q: true
      replay_buffer_capacity: 1000
      warmup_steps: 0
      v_min: -5.0
      v_max: 5.0
      prioritized_replay_alpha: 0.6
      prioritized_replay_beta: 0.4

""") + _BASE_MODEL + _BASE_TUNE

_YAML_MULTI_AGENT_PPO = textwrap.dedent("""\
    experiment:
      name: cov-mappo
      output_dir: {output_dir}
      seed: 0

    env:
      agent_count: 2
      max_steps: 5
      geometry_boxes:
        - [10, 10, 10, 22, 22, 22]

    training:
      algorithm: multi_agent_voxel_ppo
      runner: local
      iterations: 1
      checkpoint_interval: 100
      trajectory_every: 100
      video_every: 100
      require_gpu: false
      num_env_runners: 0

    algorithm_config:
      lr: 1.0e-3
      gamma: 0.99
      train_batch_size: 32
      clip_param: 0.2
      num_sgd_iter: 1
      lambda_: 0.95
      kl_coeff: 0.2
      minibatch_size: 16

""") + _BASE_MODEL + _BASE_TUNE

# YAML for regression: SAC sweep that explicitly tunes tau and n_step — these
# are SAC-specific fields that were silently dropped by the old _PPO_FIELDS filter.
_YAML_SAC_SPECIFIC_SEARCH = textwrap.dedent("""\
    experiment:
      name: cov-sac-specific
      output_dir: {output_dir}
      seed: 0
""") + _BASE_ENV + textwrap.dedent("""\
    training:
      algorithm: sac
      runner: local
      iterations: 1
      checkpoint_interval: 100
      trajectory_every: 100
      video_every: 100
      require_gpu: false
      num_env_runners: 0

    algorithm_config:
      lr: 1.0e-3
      gamma: 0.99
      train_batch_size: 32
      tau: 0.005
      n_step: 1
      warmup_steps: 0

""") + _BASE_MODEL + textwrap.dedent("""\
    tune_config:
      scheduler: asha
      num_samples: 2
      metric: episode_reward_mean
      mode: max
      max_concurrent: 1
      search_space:
        algorithm_config:
          lr:    {{choice: [1.0e-4, 3.0e-4]}}
          tau:   {{choice: [0.005, 0.01]}}
          n_step: {{choice: [1, 2]}}

    mlflow: {{}}
""")

# YAML for regression: DQN sweep that tunes n_step — DQN-specific.
_YAML_DQN_SPECIFIC_SEARCH = textwrap.dedent("""\
    experiment:
      name: cov-dqn-specific
      output_dir: {output_dir}
      seed: 0
""") + _BASE_ENV + textwrap.dedent("""\
    training:
      algorithm: dqn
      runner: local
      iterations: 1
      checkpoint_interval: 100
      trajectory_every: 100
      video_every: 100
      require_gpu: false
      num_env_runners: 0

    algorithm_config:
      lr: 1.0e-3
      gamma: 0.99
      train_batch_size: 32
      n_step: 1
      num_atoms: 1
      dueling: true
      double_q: true
      noisy: false
      replay_buffer_capacity: 1000
      warmup_steps: 0

""") + _BASE_MODEL + textwrap.dedent("""\
    tune_config:
      scheduler: asha
      num_samples: 2
      metric: episode_reward_mean
      mode: max
      max_concurrent: 1
      search_space:
        algorithm_config:
          lr:     {{choice: [1.0e-4, 3.0e-4]}}
          n_step: {{choice: [1, 2]}}

    mlflow: {{}}
""")


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _load(yaml_template: str, tmp_path: Path):
    output_dir = tmp_path / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    text = yaml_template.format(output_dir=str(output_dir).replace("\\", "/"))
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(text)
    from theseo_anysearch.experiments.loader import load_experiment
    return load_experiment(cfg_file), cfg_file


# ---------------------------------------------------------------------------
# A. Every algorithm completes a TuneRunner sweep (2 samples, 1 iteration)
# ---------------------------------------------------------------------------

class TestTuneRunnerAllAlgorithms:
    """
    Each active algorithm must complete a 2-sample ASHA sweep via TuneRunner
    without raising an exception.  The result dict must contain the standard
    keys (best_config, best_result, experiment_dir, run_tag).

    num_env_runners=0 is used for all tests to avoid the SIGSEGV-kills-raylet
    placement-group bug (spec/bugs.md §2).
    """

    @pytest.mark.usefixtures("ray_session")
    def test_ppo_sweep_completes(self, tmp_path):
        """Single-agent PPO: 2-trial ASHA sweep via TuneRunner."""
        pytest.importorskip("theseo_core")
        cfg, cfg_file = _load(_YAML_PPO, tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner
        result = TuneRunner(cfg, config_path=cfg_file, tag="cov-ppo").run()
        for key in ("best_config", "best_result", "experiment_dir", "run_tag"):
            assert key in result, f"Missing key '{key}' in ppo sweep result"

    @pytest.mark.usefixtures("ray_session")
    def test_sac_sweep_completes(self, tmp_path):
        """Single-agent SAC: 2-trial ASHA sweep via TuneRunner."""
        pytest.importorskip("theseo_core")
        cfg, cfg_file = _load(_YAML_SAC, tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner
        result = TuneRunner(cfg, config_path=cfg_file, tag="cov-sac").run()
        for key in ("best_config", "best_result", "experiment_dir", "run_tag"):
            assert key in result, f"Missing key '{key}' in sac sweep result"

    @pytest.mark.usefixtures("ray_session")
    def test_dqn_sweep_completes(self, tmp_path):
        """Single-agent DQN: 2-trial ASHA sweep via TuneRunner."""
        pytest.importorskip("theseo_core")
        cfg, cfg_file = _load(_YAML_DQN, tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner
        result = TuneRunner(cfg, config_path=cfg_file, tag="cov-dqn").run()
        for key in ("best_config", "best_result", "experiment_dir", "run_tag"):
            assert key in result, f"Missing key '{key}' in dqn sweep result"

    @pytest.mark.usefixtures("ray_session")
    def test_rainbow_sweep_completes(self, tmp_path):
        """Single-agent Rainbow: 2-trial ASHA sweep via TuneRunner."""
        pytest.importorskip("theseo_core")
        cfg, cfg_file = _load(_YAML_RAINBOW, tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner
        result = TuneRunner(cfg, config_path=cfg_file, tag="cov-rainbow").run()
        for key in ("best_config", "best_result", "experiment_dir", "run_tag"):
            assert key in result, f"Missing key '{key}' in rainbow sweep result"

    @pytest.mark.usefixtures("ray_session")
    def test_multi_agent_ppo_sweep_completes(self, tmp_path):
        """Multi-agent PPO: 2-trial ASHA sweep via TuneRunner."""
        pytest.importorskip("theseo_core")
        cfg, cfg_file = _load(_YAML_MULTI_AGENT_PPO, tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner
        result = TuneRunner(cfg, config_path=cfg_file, tag="cov-mappo").run()
        for key in ("best_config", "best_result", "experiment_dir", "run_tag"):
            assert key in result, f"Missing key '{key}' in multi_agent_ppo sweep result"


# ---------------------------------------------------------------------------
# B. Algorithm-specific search space fields are applied (regression)
# ---------------------------------------------------------------------------

class TestAlgorithmSpecificSearchSpace:
    """
    Regression for the _PPO_FIELDS bug in _experiment_trainable.

    Before the fix, _experiment_trainable used a hardcoded frozenset of PPO
    field names to filter the search-space config dict before applying it to
    algorithm_config.  This had two failure modes:

      1. SAC/DQN-specific params (tau, n_step) were silently absent from
         algo_updates even when explicitly included in search_space, so the
         sampled values were never applied to the trial's config.

      2. If PPO-only fields (clip_param, num_sgd_iter) were passed to a
         SACConfig or DQNConfig via model_copy(), Pydantic raised
         ValidationError because those fields don't exist on those classes.

    Fix: filter algo_updates by exp_config.algorithm_config.model_fields so
    only fields that actually exist on the concrete config class are applied.

    These tests sweep SAC- and DQN-specific fields and assert the sweep
    completes without error (ValidationError would propagate as a failed trial
    and missing best_result key).
    """

    @pytest.mark.usefixtures("ray_session")
    def test_sac_specific_fields_do_not_raise(self, tmp_path):
        """
        Sweeping SAC's tau and n_step must not raise ValidationError.

        With the old _PPO_FIELDS filter, tau and n_step were in the filter
        (they weren't) or were included but not in SACConfig, causing
        model_copy() to fail.  The corrected filter uses model_fields so only
        SACConfig-native fields are applied.
        """
        pytest.importorskip("theseo_core")
        cfg, cfg_file = _load(_YAML_SAC_SPECIFIC_SEARCH, tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner
        # If ValidationError fires, TuneRunner propagates it — test fails clearly.
        result = TuneRunner(cfg, config_path=cfg_file, tag="sac-specific").run()
        assert "best_config" in result
        # The search space included tau and n_step; best_config must contain them.
        assert "tau" in result["best_config"], (
            "tau was in search_space but missing from best_config — "
            "field was silently dropped by the config filter."
        )
        assert "n_step" in result["best_config"], (
            "n_step was in search_space but missing from best_config — "
            "field was silently dropped by the config filter."
        )

    @pytest.mark.usefixtures("ray_session")
    def test_dqn_specific_fields_do_not_raise(self, tmp_path):
        """
        Sweeping DQN's n_step must not raise ValidationError.

        n_step is a DQNConfig field; it is not in PPOConfig.  The old filter
        would include it in algo_updates for DQN (if it happened to be in
        _PPO_FIELDS — it wasn't, so it was silently dropped instead).  Either
        way the sampled value was never applied.  The corrected filter passes
        n_step through for DQN because it is in DQNConfig.model_fields.
        """
        pytest.importorskip("theseo_core")
        cfg, cfg_file = _load(_YAML_DQN_SPECIFIC_SEARCH, tmp_path)
        from theseo_anysearch.experiments.tune_runner import TuneRunner
        result = TuneRunner(cfg, config_path=cfg_file, tag="dqn-specific").run()
        assert "best_config" in result
        assert "n_step" in result["best_config"], (
            "n_step was in search_space but missing from best_config — "
            "field was silently dropped by the config filter."
        )


# ---------------------------------------------------------------------------
# C. Trainer registry completeness (no theseo_core needed)
# ---------------------------------------------------------------------------

class TestTrainerRegistry:
    """
    Verify that every algorithm key expected in YAML resolves to a Trainer
    subclass in the registry.  These tests do not train — they only check
    that the registry is wired correctly.  No theseo_core required.
    """

    TUNABLE_ALGORITHMS = ["ppo", "sac", "dqn", "rainbow", "multi_agent_voxel_ppo"]
    STUB_ALGORITHMS = ["td3", "ddpg"]  # Box action space only — not usable with voxel env

    def test_tunable_algorithms_are_registered(self):
        from theseo_anysearch.rllib.trainer.base import Trainer
        for algo in self.TUNABLE_ALGORITHMS:
            assert algo in Trainer._registry, (
                f"Algorithm '{algo}' not in Trainer._registry. "
                "Was its trainer module imported?"
            )

    def test_stub_algorithms_are_registered(self):
        """TD3 and DDPG register themselves even though they raise NotImplementedError."""
        from theseo_anysearch.rllib.trainer.base import Trainer
        for algo in self.STUB_ALGORITHMS:
            assert algo in Trainer._registry, (
                f"Stub algorithm '{algo}' not in Trainer._registry."
            )

    def test_algorithm_config_classes_resolve_for_all_active(self):
        """get_algorithm_config_class returns a concrete subclass for each active algo."""
        from theseo_anysearch.rllib.algorithms.models import (
            get_algorithm_config_class,
            AlgorithmConfig,
        )
        from theseo_anysearch.models import AlgorithmConfig as BaseAlgorithmConfig
        for algo in self.TUNABLE_ALGORITHMS:
            cls = get_algorithm_config_class(algo)
            assert issubclass(cls, BaseAlgorithmConfig), (
                f"Config class for '{algo}' is not an AlgorithmConfig subclass."
            )
            # Must have at least lr, gamma, train_batch_size from base
            assert "lr" in cls.model_fields
            assert "gamma" in cls.model_fields
            assert "train_batch_size" in cls.model_fields

    def test_model_fields_filter_covers_algo_specific_fields(self):
        """
        Regression: verify that using model_fields to filter search-space
        params correctly includes algorithm-specific fields.

        This is the unit-level check for the _PPO_FIELDS → model_fields fix
        in _experiment_trainable.
        """
        from theseo_anysearch.rllib.algorithms.models import (
            PPOConfig, SACConfig, DQNConfig, RainbowConfig,
        )
        # PPO-specific fields must be in PPOConfig but not SACConfig
        assert "clip_param" in PPOConfig.model_fields
        assert "clip_param" not in SACConfig.model_fields

        # SAC-specific fields must be in SACConfig but not PPOConfig
        assert "tau" in SACConfig.model_fields
        assert "tau" not in PPOConfig.model_fields
        assert "n_step" in SACConfig.model_fields
        assert "n_step" not in PPOConfig.model_fields

        # DQN-specific fields must be in DQNConfig
        assert "n_step" in DQNConfig.model_fields
        assert "num_atoms" in DQNConfig.model_fields
        assert "dueling" in DQNConfig.model_fields

        # Rainbow inherits DQN fields
        assert "n_step" in RainbowConfig.model_fields
        assert "prioritized_replay_alpha" in RainbowConfig.model_fields

        # Common fields are on all
        for cls in (PPOConfig, SACConfig, DQNConfig, RainbowConfig):
            assert "lr" in cls.model_fields
            assert "gamma" in cls.model_fields
