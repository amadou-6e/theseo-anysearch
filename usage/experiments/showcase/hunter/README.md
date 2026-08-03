# Hunter movement extension

This standalone showcase defines only the hunter movement behavior. It does not
pretend that AnySearch already supports heterogeneous agent pipelines.

The hunter uses two experiment-local Rust extensions:

- `double_step_in_bounds` evaluates the final doubled destination and rejects an
  action if that destination would leave the grid;
- `double_step` replaces normal cursor movement with
  `cursor + 2 * (selected_direction)`.

There is no trail outcome. The example uses an empty grid because the current
predicate context reports occupancy for the normal one-voxel destination, not an
arbitrary doubled destination. Supporting agent-specific pipelines, peer state,
ordered turns, and heterogeneous observations/rewards is tracked separately.

Compile and run the one-agent hunter demonstration:

```powershell
anysearch compile usage/experiments/showcase/hunter
anysearch train usage/experiments/showcase/hunter/experiment.yaml

# Deterministic extension check
python usage/experiments/showcase/hunter/validate.py
```

The YAML passes `multiplier: 2` independently to the predicate and outcome. Both
default to two if the parameter is omitted.