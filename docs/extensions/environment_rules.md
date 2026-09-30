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

## Rust extension metadata

The Rust SDK attributes export a versioned JSON metadata symbol next to each
predicate, outcome, reward, and scenario symbol. The compiler probes those
symbols and records the results under `rule_metadata` in `extension.json`.
Training and evaluation metric capabilities receive corresponding metadata
records as well. This makes native compatibility inspectable before Ray or an
environment is constructed.

The default contract is version `1` and supports the `voxel` environment
family. Attribute arguments declare a more specific contract:

```rust
#[anysearch_predicate(
    version = 2,
    environment_families = "voxel,surface",
    dependencies = "predicate:bounds",
    conflicts = "predicate:legacy_bounds"
)]
fn bounded_turn(context: &PredicateContext) -> PredicateResult {
    // ...
}
```

References use `kind:name`; comma-separate multiple values. Valid kinds are
`predicate`, `outcome`, `reward`, `training_metrics`, `evaluation_metrics`, and
`scenario`. The manifest metadata schema is independently versioned as version
`1`. A host that cannot read that schema rejects the manifest during preflight
with a compatibility error.

Name-only manifests produced before metadata schema version 1 remain readable,
including archived run manifests. Their rule records are synthesized with rule
version `1` and the `voxel` environment family. A source extension may still
require recompilation when the bundled SDK contract changes; that produces the
existing explicit `run 'anysearch compile ...'` migration diagnostic and then
publishes structured metadata.
