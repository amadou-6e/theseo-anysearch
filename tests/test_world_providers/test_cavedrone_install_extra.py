from pathlib import Path
import tomllib

from packaging.requirements import Requirement


def test_cavedrone_extra_selects_separate_compatible_provider():
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    requirements = project["optional-dependencies"]["cavedrone"]
    assert len(requirements) == 1
    provider = Requirement(requirements[0])
    assert provider.name == "theseo-anysearch-cavedrone"
    assert "0.1.0" in provider.specifier
    assert "0.2.0" not in provider.specifier
    assert all(
        Requirement(value).name != provider.name
        for value in project["dependencies"]
    )
