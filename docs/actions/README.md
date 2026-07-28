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
| `mode` | `discrete_26` | Selects the action representation. Accepted values are `discrete_6`, `discrete_18`, `discrete_26`, and `vector_3`. |

`discrete_26` uses one discrete action for every non-zero movement vector in `{-1, 0, 1}³`. The existing single-agent environment exposes these 26 neighboring voxel moves.

```text
(-1, -1, -1)
(-1, -1,  0)
...
(+1, +1, +1)
```

The legacy `env.action_mode` field remains loadable during migration. Do not combine it with `env.action` in the same configuration.

`discrete_6` permits the six axial moves with `||delta||^2 <= 1`. `discrete_18`
adds the twelve two-axis diagonals with `||delta||^2 <= 2`. Both exclude zero.
`vector_3` exposes `MultiDiscrete([3, 3, 3])`; components map from
`0, 1, 2` to `-1, 0, 1`, including the no-op `[1, 1, 1]`.
