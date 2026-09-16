from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

from theseo_anysearch.environments.task import TaskConfig
from theseo_anysearch.settings.training import TrainingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGURATION_MODULES = (
    PROJECT_ROOT / "theseo_anysearch/settings/training.py",
    PROJECT_ROOT / "theseo_anysearch/environments/task.py",
)


@pytest.mark.parametrize("module_path", CONFIGURATION_MODULES)
def test_configuration_classes_do_not_redeclare_annotated_fields(
    module_path: Path,
) -> None:
    module = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    duplicates: dict[str, list[str]] = {}

    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        field_names = [
            statement.target.id
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        ]
        repeated = sorted(
            name for name, count in Counter(field_names).items() if count > 1
        )
        if repeated:
            duplicates[node.name] = repeated

    assert duplicates == {}


def test_authoritative_configuration_defaults() -> None:
    assert TrainingConfig(algorithm="ppo").max_requests_in_flight_per_env_runner == 2
    assert TaskConfig().max_consecutive_collisions is None