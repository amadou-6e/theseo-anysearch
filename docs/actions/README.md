# Action configuration

The policy action representation belongs under `env.action`.

```yaml
env:
  action:
    mode: discrete_26
```

## Fields

| Field | Default | Purpose |
|---|---:|---|
| `mode` | `discrete_26` | Selects the action representation. Accepted values are `discrete_26` and `vector_3`. |

`discrete_26` uses one discrete action for every non-zero movement vector in `{-1, 0, 1}³`. The existing single-agent environment exposes these 26 neighboring voxel moves.

```text
(-1, -1, -1)
(-1, -1,  0)
...
(+1, +1, +1)
```

`vector_3` identifies the compact three-component action configuration used by the vector-action experiment files. The nested schema carries this selection into the runtime environment dictionary; action-space behavior remains owned by the environment implementation.

The legacy `env.action_mode` field remains loadable during migration. Do not combine it with `env.action` in the same configuration.