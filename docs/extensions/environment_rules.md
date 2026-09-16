# Environment rule registry and preflight

AnySearch resolves named environment behavior before it creates a Ray trainer or
worker. This prevents a misspelled, incompatible, or incorrectly ordered rule
from becoming a delayed trial failure.

```yaml
env:
  action:
    predicates:
      - valid_action
      - bounds
      - unoccupied
    outcomes:
      - cursor_movement
      - trail_placement
```

The registry stores a typed metadata record for each selectable rule:

- selectable name and rule kind;
- contract version and implementation source;
- supported environment families;
- required rules;
- conflicting rules.

Dependencies must be selected and, within the same pipeline, must precede the
rule that requires them. Unknown names, duplicates, missing dependencies,
conflicts, unsupported environments, and stale native manifests fail with the
corresponding YAML path before Ray starts.

## Third-party registration

Compiled Rust predicates, outcomes, and rewards are registered from their
validated native manifest. Python packages can publish metadata without
changing trainer code:

```python
from theseo_anysearch.environment_rules import (
    EnvironmentRuleMetadata,
    register_environment_rule,
)

register_environment_rule(
    EnvironmentRuleMetadata(
        name="minimum_turn_radius",
        kind="predicate",
        source="python",
    )
)
```

Registration does not execute the rule. It makes its compatibility contract
discoverable and allows preflight resolution; the associated runtime extension
must still provide the selected implementation.
