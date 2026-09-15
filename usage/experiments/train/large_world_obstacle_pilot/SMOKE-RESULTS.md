# Six-gate imitation-to-PPO smoke result

This is feasibility evidence for issue #409, not a preregistered
perception-encoder result or evidence of 4096-action policy success. Governing
[pilot specification](https://github.com/amadou-6e/specs/blob/a94227bc4ee484287a026f89ec6cd47d5ca16d26/projects/theseo-anysearch/python/perception-encoder-pilots.md)
is pinned at `a94227bc4ee484287a026f89ec6cd47d5ca16d26`.

| Field | Value |
| --- | --- |
| Accepted run | `937a8975`, `COMPLETED` |
| Source revision | `88d7518` on `exp/409` |
| Config SHA-256 | `4a82de3319e8f42b18ddb34f36c538fd5cf0ee1ddfabe89bd84e4b8984650faa` |
| Compiled world identity | `ed3f7cf6a2ab67d5d9bb82537bfa8fa50638a599cabc7ded2213fd4c3cc20c05` |
| Authorized bounded run | 12 accepted demonstrations, at most 12 attempts, 2 behavior-cloning epochs, 2 PPO iterations, 1024 transitions per PPO batch |
| Local output | `runtime/gs/gate-smoke/937a8975/` |
| Local dataset | `runtime/gs/dataset-identity-pinned/` |

The collector accepted one successful demonstration for each of the 12 fixed
routes (2 through 4096 actions), totaling 8190 labeled actions. Its
`continue_route` branch used direct `shortest_actions`, not the configured
`replanning_astar` provider; preflight had established that these gate-axis
actions are collision-free. Eleven episodes / 8158 transitions went to
training and one 32-action episode to validation. The dataset manifest now
records the exact compiled-world identity. Behavior cloning completed two
epochs (best validation loss `0.00662`, validation action accuracy `1.0`) and
handed encoder and policy weights to PPO. PPO completed two iterations and
wrote both checkpoints. The one-episode pre-RL evaluation and iteration-2
evaluation each succeeded on the **initial 2-action stage only**.

These figures establish execution and artifact provenance, not learnability of
the later stages. The held-out episode is just one short, same-axis route, so
its accuracy is not an independent navigation generalization measure. No
all-stage retention evaluation or scratch control was run under this cap.

The earlier run `eed0206d` completed, but its demonstration manifest omitted
the compiled-world identity. It is excluded from the accepted evidence and
its local artifacts remain separate; the fresh run above used a new dataset
path and a source revision that pins the identity.

## Artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| `runtime/gs/dataset-identity-pinned/manifest.json` | `1a41708d893bec1ad2bc667547767098b08b635bed8ddd71cf5ee4144e941f69` |
| `runtime/gs/dataset-identity-pinned/demonstrations.npz` | `7c1134cd0411eae4ee3060c8fe985d82973e7a7388c77f0917337d71f7cd62ab` |
| `runtime/gs/gate-smoke/937a8975/imitation/policy_state.pt` | `c1a6179acb43eb91c3e9d06de359c88dd1f47b3958aae66620a98bac967f5347` |
| `runtime/gs/gate-smoke/937a8975/checkpoints/iter_000002/learner_group/learner/rl_module/default_policy/module_state.pkl` | `f4f2bd9754e2cf52be21cd97b0282646cf6129ce2db863db31f99a49699e7479` |

Next evidence-directed action: freeze a larger, separately budgeted run with
all-stage held-out evaluation and a matched from-scratch control. This bounded
smoke does not authorize or estimate the cost of that comparison.
