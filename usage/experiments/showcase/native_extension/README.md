# Native Rust extension example

This folder demonstrates an experiment-local compiled reward and metric extension.
The Rust library adds a collision penalty and publishes one training and evaluation
metric proving that the native hook ran.

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
