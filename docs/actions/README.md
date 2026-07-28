# Action configuration

The `env.action.mode` setting controls how a policy represents one movement of the
agent. Every movement is ultimately converted to a voxel offset
`(dx, dy, dz)`, where each component is `-1`, `0`, or `1`.

```yaml
env:
  action:
    mode: discrete_26
```

## Available modes

| Mode | Gymnasium space | Policy choices | Allowed movement |
|---|---|---:|---|
| `discrete_6` | `Discrete(6)` | 6 | One-axis moves only |
| `discrete_18` | `Discrete(18)` | 18 | One-axis and two-axis diagonal moves |
| `discrete_26` | `Discrete(26)` | 26 | Every neighboring voxel, including three-axis diagonals |
| `vector_3` | `MultiDiscrete([3, 3, 3])` | 27 combinations | Every neighboring voxel plus a no-op |

The default is `discrete_26`.

## Reduced discrete spaces

The discrete modes assign one integer action to each permitted movement vector.
The zero vector is excluded, so every valid discrete action requests movement.

The modes are defined using the squared Euclidean length of the offset:

```text
squared length = dx^2 + dy^2 + dz^2
```

### `discrete_6`

`discrete_6` permits offsets with squared length 1. These are the six axial
neighbors:

```text
(-1, 0, 0)  (+1, 0, 0)
( 0,-1, 0)  ( 0,+1, 0)
( 0, 0,-1)  ( 0, 0,+1)
```

This is the smallest action space. It may make exploration and action selection
easier, but diagonal routes require more steps.

### `discrete_18`

`discrete_18` includes the six axial moves and the twelve offsets with squared
length 2, such as:

```text
(+1,+1, 0)
(+1, 0,-1)
( 0,-1,+1)
```

It excludes the eight three-axis corner diagonals such as `(+1,+1,+1)`.

### `discrete_26`

`discrete_26` includes every non-zero vector in `{-1, 0, 1}^3`:

- 6 axial moves with squared length 1;
- 12 two-axis diagonals with squared length 2;
- 8 three-axis diagonals with squared length 3.

This allows the shortest geometric routes through the full 26-neighbor voxel
graph, at the cost of giving the policy more discrete actions to distinguish.

## Three-component vector space

`vector_3` exposes `MultiDiscrete([3, 3, 3])`. The policy selects three category
indices, one for each axis. Each component is decoded as follows:

| Policy value | Movement component |
|---:|---:|
| `0` | `-1` |
| `1` | `0` |
| `2` | `+1` |

For example:

```text
policy action [2, 1, 0]
movement      [+1, 0,-1]
```

The center action `[1, 1, 1]` decodes to `(0, 0, 0)` and is a true no-op: it
consumes a step but does not move the agent and is not treated as a collision.
All other combinations correspond to the same 26 neighboring voxels available
in `discrete_26`.

`vector_3` lets the policy model the axes as three categorical decisions instead
of one 26-class decision. This representation can exploit axis structure, but it
does not guarantee faster learning because the three component distributions
must still combine into a useful movement.

## Collisions and blocked movement

An allowed action can still fail to move the agent when its destination is
outside the grid or blocked by geometry or another filled voxel. The environment
then keeps the cursor in place and applies the configured collision behavior.
This is different from the intentional `vector_3` no-op.

## Choosing a mode

Use `discrete_6` when only axis-aligned motion is valid or when minimizing the
number of policy choices is the priority. Use `discrete_18` when edge diagonals
are useful but corner-cutting should be disallowed. Use `discrete_26` for the
full neighboring-voxel graph. Use `vector_3` when you want the policy output to
preserve the three-axis structure explicitly.

The best representation is task-dependent. Compare modes with the same seeds,
training budget, observation, and reward settings rather than assuming that the
smallest action space will always learn fastest.

## Legacy configuration

The legacy flat `env.action_mode` field remains loadable during migration. Do
not combine it with the nested `env.action` block in the same configuration.