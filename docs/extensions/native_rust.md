# Native Rust extensions

AnySearch experiments can define reward and metric hooks in Rust. Put a Cargo
`cdylib` crate in the experiment's `extension/` directory and compile it with:

```powershell
anysearch compile path/to/experiment
```

The command generates `Cargo.lock`, builds a release library, validates ABI v1,
and stores a source-hashed artifact beneath the ignored `.anysearch/` directory.
A subsequent command reuses the artifact when its sources and binary hash still
match. Use `--force` to rebuild it.

At run startup, AnySearch rejects stale sources or an incompatible library. It
copies the exact manifest and platform library into the run directory; Tune does
the same independently for every trial. Worker processes load this archived copy.
Artifacts are platform-specific and should be recompiled on the target machine.

## ABI and precedence

Every library exports `anysearch_extension_abi_version` and
`anysearch_extension_capabilities`. It may then export any combination of:

- `anysearch_compute_reward_v1`
- `anysearch_compute_training_metrics_v1`
- `anysearch_compute_evaluation_metrics_v1`

Rewards use fixed-layout C-compatible context/result structs because this call is
made on every environment step. A result contains add/replace mode, a finite
reward, and up to eight named components. Metrics receive UTF-8 JSON and return a
JSON object because metric calls are infrequent and benefit from a flexible
context. Returned names get the normal `training_` or `evaluation_` prefix.

For each capability, compiled Rust takes precedence over the matching Python
hook. Capabilities not supplied by Rust continue to use the Python hook and then
the built-in behavior. This makes partial migrations possible.

Native extensions are trusted code loaded into the training process. Only compile
and run extension sources you trust. See
`usage/experiments/showcase/native_extension` for a dependency-free runnable
example and exact Rust structure definitions.
