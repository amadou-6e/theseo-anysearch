# AnySearch UI shell (feat/200)

Tauri + React/TypeScript replacement for the `egui` desktop UI, tracked by
[#200](https://github.com/amadou-6e/theseo-anysearch/issues/200). Design
reference: `spec/ui-design/replayer-current.drawio` (tabs "Replay
(Optimized)" and "Sidebar Mock (HTML)").

## Architecture

- `src-tauri/` — Tauri backend, split into `workspace.rs` (scan/validate/
  edit/run lifecycle), `trajectory.rs` (trajectory + Tune-trial discovery),
  `explain_bridge.rs` (checkpoint-backed policy explanation process).
- `web/` — React/TypeScript/Vite frontend: Runs / Replay / Explain panels.
- `viewer-web/` — the simulation/voxel viewport, staying in `egui` per the
  architecture decision, compiled to WebAssembly (`eframe`'s web target) and
  mounted as a `<canvas>` inside the React `ReplayPanel`. Faithful port of
  the camera/projection/painter code in `voxel_replay.rs` — same math, same
  draw order, no file I/O (data is pushed in from JS).
- `theseo_anysearch/ui/{service,workspace}.py` — the authoritative workspace
  scanner, ported (not reimplemented) from `feat/197`'s `2bd4f62`. `Runs`
  shells out to `python -m theseo_anysearch.ui.service {scan,validate}`
  exactly like that branch's native `WorkspaceUi`, so both UIs discover the
  same `run.json` manifests and classify YAML the same way. See
  `docs/ui/workspace.md` (also ported from `feat/197`) for the full contract.
- `theseo_anysearch.cli.main run` / `theseo_anysearch.rllib.explain.native_bridge`
  — existing Python entry points the backend shells out to for Start-run and
  Explain, unchanged from what the native shell already used.

## Setup

```bash
# 1. one-time: install the wasm target and wasm-pack
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# 2. build the viewer-web wasm module (re-run after editing viewer-web/)
cd web && npm install && npm run build:viewer

# 3. run the app (needs Python + this package importable for the Runs tab's
#    workspace scan, Start-run, and Explain — set ANYSEARCH_PYTHON if
#    `python` on PATH isn't the right interpreter, same convention as
#    feat/197's native shell)
npx tauri dev   # from src-tauri/, or `cargo tauri dev` if the Tauri CLI is installed
```

## Status — what's real vs. scaffolded

Everything below was verified by actually launching the app (WebView2
remote debugging + CDP) and driving it against real workspace/trajectory
data under `usage/experiments/`, not just compiling it.

**Shared shell** — a persistent left "Run history" sidebar (`RunHistorySidebar.tsx`) is shown on Runs/Replay/Explain alike, per `docs/ui/workspace.md`'s "the run history is the left pane and remains present ... so run context does not move when tabs change" and `spec/ui-design/replayer-current.drawio`'s "All Windows" tab (which shows the same run-history column across all three window mockups). The header shows the workspace name and, once a run is selected, "SELECTED RUN — …" next to the tab bar. Workspace scan state (`root`/`index`/selected run) lives in `App.tsx`, not inside any one tab panel. Verified live: selecting a run in Runs, then switching to Replay/Explain, kept the same sidebar with the run still highlighted.

Clicking a run card does **not** show a list of its trajectory files to
choose from (an earlier pass here added exactly that, which wasn't in the
spec and was redundant besides: `ReplayPanel` loads the *entire* iteration
history for a trajectory's directory regardless of which file inside it
it's handed). One click resolves a representative file (preferring
`best.json`) purely to get a valid directory, and Replay/Explain become
available immediately — matching `docs/ui/workspace.md`'s "Replay becomes
available after selecting a run containing saved trajectories." Iteration/
step navigation happens with the Iterations/Steps sliders inside Replay.
Verified live: one click on a run card, then Replay tab, showed the full
5-iteration history with no intermediate file-picking step.

**Runs tab**
- `scan_workspace`/`validate_configuration` shell out to the real
  `theseo_anysearch.ui.service` backend — verified against a live workspace:
  correct `run.json`-derived run history (including real `RUNNING`/
  `COMPLETED` statuses), correct ◆/`!`/◇ classification, and a genuine
  Pydantic validation error surfaced for a real misconfigured YAML
  (`training.algorithm='ppo'` vs. `env.agent_count=4`).
- Collapsible directory tree (not a flat list), with search.
- YAML editor with Validate / Save / Start run / Stop, diagnostics shown
  inline, and live terminal output streamed via `run-output`/`run-exited`
  Tauri events. (Save/Start run were verified by code path and the
  Validate/diagnostics path was verified live; Save/Start run were not
  fired against real files during verification to avoid mutating the
  user's workspace or kicking off a real training job.)
- "Open trajectories folder" — a direct path-based entry point into Replay
  for runs/Tune trials with no `run.json` (legacy `ray_runtime.json`-keyed
  trials never get a Run History row, and `trajectories/` is deliberately
  excluded from the scanned file tree by `workspace.py`'s
  `_workspace_files` — this mirrors the native CLI's `--tune-dir`/file-mode
  entry points instead of trying to route everything through run.json).

**Replay tab**
- Iteration history: loads every `iter_*.json` (or `best.json`) in a run's
  `trajectories/` dir, with an Iterations scrubber — not just one file.
  Verified live (5 iterations loaded, scrubber navigated).
- Tune-trial navigation: auto-detects a sibling Tune sweep (trial dirs with
  `ray_runtime.json`) and shows a Trial scrubber (T/Y keys) when found.
  Verified live: switching trials reloaded a different trial's data.
- Playback (space to play/pause, auto-advances step then iteration,
  stopping at the end) — verified live (step counter advanced ~1 step per
  120ms while playing).
- Reward curve chart for the current iteration, with a marker at the
  selected step — verified live.
- Occlusion toggle actually changes rendering (voxels depth-sorted together
  with the cursor marker so geometry can hide it, vs. cursor always drawn
  on top) — not just UI chrome.
- `episode.init_filled_file` (`.npy` geometry sidecar) is now resolved —
  verified live against a real trajectory (4,601 real geometry voxels
  rendered, where it showed 0 before this pass).
- "Explain current step" hands the trajectory's `source_path` + selected
  step to the Explain tab and switches to it.

**Explain tab**
- Full port of `NativeExplainUi`'s `ExplanationBridge` protocol (same
  `theseo_anysearch.rllib.explain.native_bridge` subprocess, same
  line-delimited JSON-RPC) plus a React reimplementation of the UI: a large
  main area (geometry preview stacked above the policy-explanation result)
  with a narrower right sidebar for editing the observation (scalar fields,
  `local_grid` voxel-kind grid with slice/axis controls, "Explain policy
  decision") — matches the "All Windows" spec's Explain window layout, not
  three equal columns.
- Verified live for the connect → error path: attempting to connect against
  a real run surfaced a genuine backend error (a Pydantic schema mismatch
  in that run's own config) and the UI recovered cleanly (no stuck spinner,
  Connect re-enabled). **Not verified live for the success path** — no
  local run in this repo has both a saved checkpoint and a schema-compatible
  config, so the observation editor / geometry preview / result panel were
  not exercised against a real "ready" payload. The protocol and UI code
  are believed correct (direct port of the request/response shapes
  `explain.rs` already used, and TS ports of
  `encode_scaled_integer`/`decode_scaled_integer` are simple round-trippable
  math) but this is the one area to double-check against real data first.

## Known deviations from the draw.io "All Windows" spec

A pixel/element-level pass against `spec/ui-design/replayer-current.drawio`'s
"All Windows" tab (fixed): the first tab is labeled **"Overview"** there
(this app used "Runs" — note `docs/ui/workspace.md`, also from feat/197,
calls it "Runs" in prose, so the two spec sources disagree with each
other); the sidebar's "select a run" hint was worded/placed to only appear
*after* a run was already selected (backwards) and dropped "Overview" from
its own text; the run-history pane was missing its "RUN / STATE / PROGRESS"
column header; and workspace controls (root path, Rescan, Change workspace)
lived in a full-width top bar this app invented, instead of inside the
file-tree pane header where the spec puts them.

Not fixed — genuine missing features, not alignment nits:
- "Start a new run [expand]" quick-create panel and the live
  `[running] ... 37%` progress banner.
- "All states ▾" run-status filter dropdown.
- Per-run-card **Stop / Resume / Run again / Details** actions. Checked
  whether these could be wired honestly: `stop_run` kills a `Child` handle
  *this process* spawned via Start run — it has no way to stop an arbitrary
  already-running training process discovered via `run.json` (no PID is
  recorded in the manifest). Left undone rather than shipping buttons that
  don't work.
- Drag-and-drop a folder onto the window to open it as the workspace.

Added but not in the spec at all (kept, flagged rather than hidden):
- The raw "Workspace root" text-path input — the spec only shows the native
  folder picker + drag-and-drop. Kept as a secondary "or paste a path"
  affordance (useful for scripting/testing) rather than the primary action.
- The entire "Open trajectories folder" control in the sidebar — added to
  work around `trajectories/` being deliberately excluded from the scanned
  file tree (see Runs tab notes above); without it, legacy Tune trials with
  no `run.json` have no way into Replay at all.

## Known tech debt

- `StepData`/`EpisodeData`/`TrajectoryData` are duplicated across
  `voxel_replay.rs`, `viewer-web/src/lib.rs`, and `src-tauri/src/trajectory.rs`
  (as a passthrough `serde_json::Value`). `theseo-core` can't be depended on
  directly from `src-tauri` yet because it always builds as a PyO3
  `cdylib` (`extension-module` feature isn't optional) — factor a shared,
  non-PyO3 `theseo-core-data` crate before this grows further.
- Camera drag/zoom sensitivity constants in `viewer-web`, and the
  Explain-tab geometry preview's drag sensitivity, were reconstructed to
  match the native app's *behavior*, not copied from exact input-handling
  constants — worth a side-by-side comparison pass.
- The Runs tab shells out to Python per scan/validate call, and Explain
  keeps one long-lived Python subprocess per connect — same process model
  `workspace.rs`/`explain.rs` already used, fine at the workspace sizes
  exercised so far.
- `start_run`'s process-exit watcher polls every 400ms on a background
  thread rather than using a blocking wait + Tauri's async command support
  — simpler to reason about alongside the `Mutex<Option<Child>>`, but a
  true async rewrite would be cleaner.
