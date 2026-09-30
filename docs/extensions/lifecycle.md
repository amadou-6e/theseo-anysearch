# Environment lifecycle rules

Lifecycle rules interpret a Rust transition as a Gymnasium episode outcome. They
do not mutate simulation state: movement, voxel placement, collisions, and other
state changes remain owned by the Rust action pipeline.

The default configuration preserves native task behavior:

```yaml
env:
  lifecycle:
    rules:
      - native
```

Rules run in list order. Each rule may set `success`, `failure`, `terminated`,
`truncated`, `reason`, and uniquely named JSON diagnostics. Omitted fields retain
their previous value. Success or failure implies termination, and termination
takes precedence over truncation. Conflicting terminal states, duplicate diagnostic
keys, unknown rules, and invalid results fail immediately.

## Python extension contract

Register rule factories before constructing environments. A factory receives its
YAML parameters and returns a callable receiving an immutable `LifecycleContext`:

```python
from theseo_anysearch.environments.lifecycle import LifecycleDecision, register_lifecycle_rule

def collision_budget(parameters):
    limit = int(parameters["limit"])
    def evaluate(context):
        reached = int(context.diagnostics["consecutive_collisions"]) >= limit
        return LifecycleDecision(
            failure=reached,
            reason="collision_budget" if reached else None,
            diagnostics={"collision_budget_reached": reached},
        )
    return evaluate

register_lifecycle_rule("collision_budget", collision_budget)
```

```yaml
env:
  lifecycle:
    rules:
      - native
      - name: collision_budget
        parameters:
          limit: 5
```

Rules must derive decisions solely from their context and parameters to retain
seeded determinism. Diagnostics are JSON-typed for stable trajectory, evaluation,
and distributed-worker consumption.

```text
Rust predicates -> Rust state mutation -> lifecycle rules -> Gymnasium result
```
