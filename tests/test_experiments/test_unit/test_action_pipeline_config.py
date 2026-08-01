from theseo_anysearch.models import ActionConfig, EnvConfig


def test_legacy_action_behavior_resolves_trail_mode():
    trail = EnvConfig(agent_count=1, trail_mode=True)
    cursor = EnvConfig(agent_count=1, trail_mode=False)

    assert [item["name"] for item in trail.to_runtime_dict()["action_outcomes"]] == [
        "cursor_movement",
        "trail_placement",
    ]
    assert [item["name"] for item in cursor.to_runtime_dict()["action_outcomes"]] == [
        "cursor_movement",
    ]


def test_explicit_predicates_and_outcomes_override_behavior_preset():
    config = EnvConfig.model_validate({
        "agent_count": 1,
        "action": {
            "mode": "discrete_18",
            "behavior": "trail_navigation",
            "predicates": [
                "bounds",
                {"name": "turn_radius", "parameters": {"minimum": 3}},
            ],
            "outcomes": ["cursor_movement", "custom_fill"],
            "history_length": 12,
        },
    })

    runtime = config.to_runtime_dict()
    assert runtime["action_predicates"] == [
        {"name": "bounds", "parameters": {}},
        {"name": "turn_radius", "parameters": {"minimum": 3}},
    ]
    assert runtime["action_outcomes"][-1]["name"] == "custom_fill"
    assert runtime["action_history_length"] == 12


def test_cursor_navigation_preset_does_not_place_trail():
    action = ActionConfig(behavior="cursor_navigation")
    predicates, outcomes = action.resolved_pipeline(trail_mode=True)

    assert [item.name for item in predicates] == ["valid_action", "bounds", "unoccupied"]
    assert [item.name for item in outcomes] == ["cursor_movement"]

def test_duplicate_pipeline_names_are_rejected():
    import pytest

    with pytest.raises(ValueError, match="duplicate action predicate names"):
        ActionConfig(predicates=("bounds", "bounds"))
