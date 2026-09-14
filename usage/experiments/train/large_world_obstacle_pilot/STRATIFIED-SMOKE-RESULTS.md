# Stratified gate-world smoke result

Issue [#409](https://github.com/amadou-6e/theseo-anysearch/issues/409),
PR [#410](https://github.com/amadou-6e/theseo-anysearch/pull/410), and the
[pilot spec](https://github.com/amadou-6e/specs/blob/a94227bc4ee484287a026f89ec6cd47d5ca16d26/projects/theseo-anysearch/python/perception-encoder-pilots.md)
at `a94227bc4ee484287a026f89ec6cd47d5ca16d26` govern this feasibility
check. It is not a preregistered model comparison. The run used source commit
`ce006c4` on `exp/409`, configuration SHA-256
`8fb0be6542b2ed423864a211e3f86efb93b4fddc119363a3b49f4558f71c32a1`,
and compiled-world identity
`ed3f7cf6a2ab67d5d9bb82537bfa8fa50638a599cabc7ded2213fd4c3cc20c05`
(extent 4096 x 2048 x 512).

The first attempt, `fdbfce3b` at source `ab94fc2`, completed 128-demo
collection and two behavior-cloning epochs, then failed after PPO iteration 1.
RLlib had sampled a valid batch without finished-episode reward/length means,
but the result parser required those means. It was marked `FAILED`; no PPO
result from that attempt is counted. Commit `ce006c4` makes those means
nullable and omits absent values from reporters, while still requiring episode
and environment-step counts. The retry is a new run, not a continuation.

Run `fb696335` completed on 2026-09-14. Collection accepted 128/128 unique,
successful A*-teacher demonstrations in 128 attempts, allocated
`[11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10]` across the 12 exact-length
stages. The dataset contains 66,998 training and 15,412 validation action
samples. Two behavior-cloning epochs completed; best validation loss was
0.0076844 and validation action accuracy was 0.998508. Both configured PPO
iterations completed and wrote checkpoints at iterations 1 and 2. RLlib's
lifetime completed-episode count at iteration 2 was 1; iteration 1 lacked
completed-episode averages, which are intentionally absent from its metrics.

**This does not demonstrate learned navigation.** The one-episode pre-RL
evaluation was 0/1, and the one-episode evaluations after each PPO iteration
were 0/1. The curriculum remained at stage 0 with no successful advancement.
The configured all-stage retention evaluation has frequency 10 and therefore
did not run in this two-iteration smoke. The 99.85% held-out action accuracy
is a supervised metric, not a gate-crossing success rate.

The ignored runtime artifacts are under
`runtime/gv/gate-stratified-smoke/fb696335/` in the `exp409` worktree. SHA-256:

| Artifact | SHA-256 |
| --- | --- |
| `runtime/gv/dataset/manifest.json` | `9e7043e9248ee598bd32bb3a70a3574e59d9fcce8721b0bc6d71c6fc32e499da` |
| `runtime/gv/dataset/demonstrations.npz` | `09453e0b73e1e84f905b7ff9a8c15a0a1878eb773c6844efe410b65c70499a5e` |
| `imitation/result.json` | `ea6d9e3c2addce31e914f7262d19f6be8f5ea8939321cff88dd9d4d339c541b2` |
| `imitation/policy_state.pt` | `bced0f3e225c231aae896ccd09f683a7e158ea69b7f6d8281769b2228fb2b2a5` |
| `evaluation/iter_000002.json` | `b4e3d1ef1f609702c96d1be5a0ee0cae2625b4161b68a02c06dde1bbec885176` |
| `checkpoints/iter_000002/learner_group/learner/rl_module/default_policy/module_state.pkl` | `b79d92829e6b4ddd4c5c57046a00596884bbfa720679159d326a492773241ca1` |
| `run.json` | `a0a1f997735eb3dfb837dfbec6090abb9a05f1a4b48cb68d13cf0d111c9a267f` |

Validation: 67 focused reporting tests passed. The affected unit suites had
720 passes and one unrelated worktree-layout failure:
`test_usage_experiments_do_not_use_flattened_environment_fields` filters all
experiment YAML paths containing `runtime`, including this entire worktree.
No longer training campaign was authorized; the bounded smoke budget was
128 demonstrations, two BC epochs, and two PPO iterations, completed on the
retry. The next experiment needs a separately frozen budget and actual
all-stage policy evaluation before any promotion decision.
