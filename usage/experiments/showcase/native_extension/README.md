# Native Rust extension example

This folder demonstrates an experiment-local compiled reward and metric extension.
The YAML selects `provider: native_collision`; `rewards.rs` defines that reward with
`#[anysearch_reward]`. The generated ABI wrapper adds its collision penalty, while
the library publishes one training and evaluation metric proving the native hooks ran.

The extension declares `anysearch-extension = "0.1.0"`; `anysearch compile`
redirects that dependency to the SDK bundled with the installed Python package.
No repository-relative dependency path is required.

Compile it before launching the experiment:

```powershell
anysearch compile usage/experiments/showcase/native_extension
```

The command writes a source-hashed build and manifest beneath `.anysearch/`.
AnySearch validates the ABI and source hash at startup, archives the exact binary
into each run or Tune trial, and then loads that archived copy in workers.

The extension ABI is versioned. Reward calls use fixed-layout C structs and avoid
JSON on the per-step path. Training and evaluation metrics use JSON because they
run infrequently and need flexible user-defined fields. Native code is trusted
code and executes inside the training process.

Reward authors write only the normal Rust function:

```rust
use anysearch_extension::{anysearch_reward, RewardContext, RewardResult};

#[anysearch_reward]
pub fn native_collision(context: &RewardContext) -> RewardResult {
    let penalty = if context.collision { -0.02 } else { 0.0 };
    RewardResult::add(penalty).with_component("native_collision", penalty)
}
```

The macro generates the `anysearch_reward_native_collision_v2` dynamic-library
symbol. `lib.rs` does not contain a reward wrapper.
