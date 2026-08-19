# Native workspace and run UI

Launch the integrated interface for one workspace:

```powershell
anysearch ui .
```

The application has three primary tabs:

- **Runs** is workspace-level and remains available without a selected run.
- **Replay** becomes available after selecting a run containing saved trajectories.
- **Explain** becomes available when that selected replay has a compatible checkpoint-backed explanation service.

## Workspace discovery

The right pane displays files from exactly one active workspace. `Change workspace`
uses the native directory picker; `Rescan` rebuilds the disposable index. YAML
files are classified through the Python backend:

- `◆` is a valid AnySearch configuration.
- `!` resembles an AnySearch configuration but fails authoritative loading or preflight.
- `◇` is ordinary YAML and receives no run controls.

Invalid configurations remain visible. Their original field paths and validation
messages appear above the editor.

Generated dependency/cache directories (`.git`, `.venv`, auxiliary worktrees,
build outputs, Cargo `target`, Python package metadata, MLflow storage, and
test/lint caches) are not descended into. They cannot contain workspace-owned
configurations or AnySearch run manifests and would otherwise make every rescan
traverse dependency internals. AnySearch `runtime` and experiment directories
remain indexed.

Run artifact directories are represented in the workspace index by their
manifest and small metadata files. Checkpoints, trajectories, renders, and copied
extension binaries are loaded on demand from the selected run rather than held
in the file-tree index. This keeps long-running workspaces bounded in memory.

## Editing and launching

Select a recognized configuration, edit its YAML, and use `Validate` before
`Save` or `Start run`. Validation calls the same experiment loader and environment
rule preflight used by CLI training. The UI does not maintain a second schema.

`Start run` launches `anysearch run` outside the rendering event loop. Standard
output and standard error stream into the terminal pane without being interpreted
or translated. `Stop` terminates the process selected by the UI.

The YAML pane can be collapsed so the terminal uses the remaining page height.
The run history is the left pane and remains present on Runs, Replay, and Explain
so run context does not move when tabs change. The workspace tree is the right
pane on Runs. Both side panes are resizable.

## Run interoperability

Run history is reconstructed from `run.json` manifests inside the workspace.
Consequently, CLI-created runs appear after `Rescan`; the UI does not require a
private database. Selecting a run establishes the navigation context and loads
its `trajectories/` artifacts for Replay when present.

The environment variable `ANYSEARCH_PYTHON` identifies the Python interpreter
used by the native backend. `anysearch ui` sets it automatically to the active
interpreter.
