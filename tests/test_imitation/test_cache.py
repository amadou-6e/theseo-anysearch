from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Lock, Thread

import pytest
import torch

from theseo_anysearch.imitation import cache as cache_module
from theseo_anysearch.imitation.cache import (
    cache_key_lock,
    load_cached_pretraining,
    pretraining_cache_key,
    publish_cached_pretraining,
)
from theseo_anysearch.imitation.models import (
    DemonstrationManifest,
    ImitationConfig,
    ImitationResult,
)


def _manifest() -> DemonstrationManifest:
    return DemonstrationManifest(
        fingerprint="dataset-contract",
        generation_provider_name="astar",
        generation_provider_parameters={},
        requested_episodes=2,
        successful_episodes=2,
        accepted_episodes=2,
        attempted_episodes=2,
        training_episodes=1,
        validation_episodes=1,
        training_samples=3,
        validation_samples=2,
        observation_size=4,
        action_count=2,
        seeds=[1, 2],
    )


def _result(checkpoint: Path) -> ImitationResult:
    return ImitationResult(
        epochs_completed=2,
        best_validation_loss=0.1,
        validation_accuracy=1.0,
        training_samples=3,
        validation_samples=2,
        checkpoint_path=str(checkpoint),
    )


def test_cache_key_ignores_rl_only_settings_and_tracks_model_contract() -> None:
    imitation = ImitationConfig(enabled=True)
    first = torch.nn.Linear(4, 2)
    same = torch.nn.Linear(4, 2)
    different = torch.nn.Linear(4, 3)

    first_key, _ = pretraining_cache_key(
        first, _manifest(), imitation, policy_id="default_policy"
    )
    same_key, _ = pretraining_cache_key(
        same, _manifest(), imitation, policy_id="default_policy"
    )
    different_key, _ = pretraining_cache_key(
        different, _manifest(), imitation, policy_id="default_policy"
    )

    assert first_key == same_key
    assert first_key != different_key


@pytest.mark.parametrize(
    "update",
    [
        {"world_schema_version": 2},
        {"coordinate_type": "u16"},
        {"coordinate_convention": "zero_based"},
        {"storage_coordinate_convention": "one_based"},
        {"source_origin": (1, 0, 0)},
        {"world_extent": (64, 32, 32)},
        {"world_identity_sha256": "a" * 64},
    ],
)
def test_cache_key_changes_with_world_contract(update: dict[str, object]) -> None:
    model = torch.nn.Linear(4, 2)
    imitation = ImitationConfig(enabled=True)
    current = _manifest()
    legacy = current.model_copy(update=update)

    current_key, current_contract = pretraining_cache_key(
        model, current, imitation, policy_id="default_policy"
    )
    legacy_key, legacy_contract = pretraining_cache_key(
        model, legacy, imitation, policy_id="default_policy"
    )

    assert current_key != legacy_key
    assert current_contract["world"] != legacy_contract["world"]


def test_cache_key_changes_with_sampling_provider() -> None:
    model = torch.nn.Linear(4, 2)
    manifest = _manifest()

    transition_key, _ = pretraining_cache_key(
        model,
        manifest,
        ImitationConfig(sampling={"provider": "uniform_transition"}),
        policy_id="default_policy",
    )
    episode_key, _ = pretraining_cache_key(
        model,
        manifest,
        ImitationConfig(sampling={"provider": "uniform_episode"}),
        policy_id="default_policy",
    )

    assert transition_key != episode_key


def test_heterogeneous_policy_ids_receive_distinct_cache_entries() -> None:
    model = torch.nn.Linear(4, 2)
    imitation = ImitationConfig(enabled=True)

    hunter, _ = pretraining_cache_key(
        model, _manifest(), imitation, policy_id="hunter"
    )
    hunted, _ = pretraining_cache_key(
        model, _manifest(), imitation, policy_id="hunted"
    )

    assert hunter != hunted


def test_cache_publish_and_load_materializes_checkpoint(tmp_path: Path) -> None:
    source_model = torch.nn.Linear(4, 2)
    imitation = ImitationConfig(enabled=True)
    cache_key, contract = pretraining_cache_key(
        source_model, _manifest(), imitation, policy_id="default_policy"
    )
    checkpoint = tmp_path.joinpath("source.pt")
    torch.save({"model_state": source_model.state_dict()}, checkpoint)
    cache_dir = tmp_path.joinpath("cache")
    cache_dir.mkdir()
    publish_cached_pretraining(
        cache_dir,
        cache_key,
        contract,
        _result(checkpoint),
    )
    restored_model = torch.nn.Linear(4, 2)

    result = load_cached_pretraining(
        restored_model,
        cache_dir,
        cache_key,
        contract,
        tmp_path.joinpath("trial"),
    )

    assert result is not None
    assert result.cache_hit is True
    assert result.cache_key == cache_key
    assert Path(result.checkpoint_path).is_file()
    for expected, actual in zip(source_model.parameters(), restored_model.parameters()):
        assert torch.equal(expected, actual)


def test_cache_load_rejects_changed_world_contract(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 2)
    imitation = ImitationConfig(enabled=True)
    cache_key, contract = pretraining_cache_key(
        model, _manifest(), imitation, policy_id="default_policy"
    )
    checkpoint = tmp_path.joinpath("source.pt")
    torch.save({"model_state": model.state_dict()}, checkpoint)
    cache_dir = tmp_path.joinpath("cache")
    cache_dir.mkdir()
    publish_cached_pretraining(cache_dir, cache_key, contract, _result(checkpoint))
    incompatible = {**contract, "world": {**contract["world"], "extent": [64, 32, 32]}}

    assert (
        load_cached_pretraining(
            model,
            cache_dir,
            cache_key,
            incompatible,
            tmp_path.joinpath("trial"),
        )
        is None
    )


def test_cache_key_lock_serializes_concurrent_publishers(tmp_path: Path) -> None:
    active = 0
    maximum_active = 0
    state_lock = Lock()

    def publisher() -> None:
        nonlocal active, maximum_active
        with cache_key_lock(tmp_path, "shared", 2.0):
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.05)
            with state_lock:
                active -= 1

    threads = [Thread(target=publisher), Thread(target=publisher)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert maximum_active == 1


def test_cache_key_lock_heartbeat_protects_a_slow_holder_past_its_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    """A holder still working past ``timeout_seconds`` must not be evicted.

    Regresses a bug where staleness was judged by each waiter's own elapsed
    wait time: a holder whose work legitimately outlived the timeout (e.g. a
    long behavior-cloning run) would be forcibly evicted by a waiter, and
    both would then believe they held the lock simultaneously.
    """
    monkeypatch.setattr(cache_module, "_LOCK_HEARTBEAT_INTERVAL_SECONDS", 0.02)
    events: list[str] = []
    events_lock = Lock()

    def slow_holder() -> None:
        with cache_key_lock(tmp_path, "slow", 0.2):
            time.sleep(0.5)
            with events_lock:
                events.append("holder-done")

    def impatient_waiter() -> None:
        time.sleep(0.05)
        try:
            with cache_key_lock(tmp_path, "slow", 0.2):
                with events_lock:
                    events.append("waiter-entered")
        except TimeoutError:
            with events_lock:
                events.append("waiter-timed-out")

    holder = Thread(target=slow_holder)
    waiter = Thread(target=impatient_waiter)
    holder.start()
    waiter.start()
    holder.join()
    waiter.join()

    assert "waiter-timed-out" in events
    assert "waiter-entered" not in events
    assert events.index("holder-done") < len(events)


def test_cache_key_lock_reclaims_a_genuinely_abandoned_lock(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("abandoned.lock")
    lock_path.write_text("dead-token", encoding="ascii")
    stale_time = time.time() - 10.0
    os.utime(lock_path, (stale_time, stale_time))

    entered = False
    with cache_key_lock(tmp_path, "abandoned", 0.2):
        entered = True

    assert entered
