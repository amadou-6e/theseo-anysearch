from __future__ import annotations

import json

import pytest

from theseo_anysearch.worlds.candidates import (
    CANDIDATE_DATA_FILE,
    CandidateBudgetExceeded,
    CandidateIndexHandle,
    CandidateQueryBudget,
    CandidateRecord,
    write_candidate_index,
)

IDENTITY = "a" * 64


def records() -> list[CandidateRecord]:
    return [
        CandidateRecord(
            position=(index + 1, 2, 3),
            kind="spawn" if index < 4 else "goal",
            quality=index / 5,
            region=(index // 2, 0, 0),
        )
        for index in range(6)
    ]


def test_sampling_is_deterministic_across_file_and_query_order(tmp_path) -> None:
    write_candidate_index(tmp_path, IDENTITY, list(reversed(records())))
    inferred = CandidateIndexHandle(tmp_path)
    assert inferred.world_identity == IDENTITY
    first = inferred.sample(3, "spawn", seed=41, stream=7)
    second_handle = CandidateIndexHandle(tmp_path, world_identity=IDENTITY)
    second_handle.sample(1, "goal", seed=2, stream=1)
    second = second_handle.sample(3, "spawn", seed=41, stream=7)
    assert first == second
    second_handle.sample(1, "spawn", seed=3, stream=9)
    assert second_handle.statistics["cache_hits"] > 0
    assert second_handle.statistics["latency_ns"] > 0


def test_filters_empty_ranges_and_limits(tmp_path) -> None:
    write_candidate_index(tmp_path, IDENTITY, records())
    handle = CandidateIndexHandle(tmp_path, world_identity=IDENTITY)
    assert handle.sample(4, "portal", seed=1, stream=1) == ()
    selected = handle.sample(
        4,
        "spawn",
        seed=1,
        stream=2,
        region=(1, 0, 0),
        near=(3, 2, 3),
        radius=1,
        minimum_quality=0.4,
    )
    assert {item.position for item in selected} == {(3, 2, 3), (4, 2, 3)}


def test_query_and_result_budgets_are_per_handle(tmp_path) -> None:
    write_candidate_index(tmp_path, IDENTITY, records())
    handle = CandidateIndexHandle(
        tmp_path,
        world_identity=IDENTITY,
        budget=CandidateQueryBudget(maximum_queries=2, maximum_results=2),
    )
    assert len(handle.sample(2, "spawn", seed=1, stream=1)) == 2
    with pytest.raises(CandidateBudgetExceeded, match="result"):
        handle.sample(1, "spawn", seed=1, stream=2)
    with pytest.raises(CandidateBudgetExceeded, match="query"):
        handle.sample(0, "spawn", seed=1, stream=3)


def test_identity_checksum_and_short_reads_are_rejected(tmp_path) -> None:
    write_candidate_index(tmp_path, IDENTITY, records())
    with pytest.raises(ValueError, match="different world"):
        CandidateIndexHandle(tmp_path, world_identity="b" * 64)
    data = tmp_path.joinpath(CANDIDATE_DATA_FILE)
    data.write_bytes(data.read_bytes()[:-1])
    with pytest.raises(ValueError, match="checksum"):
        CandidateIndexHandle(tmp_path, world_identity=IDENTITY)


def test_malformed_index_is_rejected(tmp_path) -> None:
    tmp_path.joinpath("candidates.idx").write_text("{", encoding="utf-8")
    tmp_path.joinpath(CANDIDATE_DATA_FILE).write_bytes(b"")
    with pytest.raises(json.JSONDecodeError):
        CandidateIndexHandle(tmp_path, world_identity=IDENTITY)
