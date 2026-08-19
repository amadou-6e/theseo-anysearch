# AnySearch UI shell (feat/200)

Tauri + React/TypeScript replacement for the `egui` desktop UI, tracked by
[#200](https://github.com/amadou-6e/theseo-anysearch/issues/200). Design
reference: `spec/ui-design/replayer-current.drawio` (tabs "Replay
(Optimized)" and "Sidebar Mock (HTML)").

## Architecture

- `src-tauri/` — Tauri backend. Exposes trajectory data to the frontend via
  `invoke()` commands.
- `web/` — React/TypeScript/Vite frontend. All panels except the voxel
  viewport (Runs, Explain, config editor, terminal — per the draw.io design).
- `viewer-web/` — the simulation/voxel viewport, staying in `egui` per the
  architecture decision, compiled to WebAssembly (`eframe`'s web target) and
  mounted as a `<canvas>` inside the React `ReplayPanel`. This is a trimmed,
  faithful port of the camera/projection/painter code in
  `theseo_anysearch/core/src/bin/voxel_replay.rs` — same math, same draw
  order, no file I/O (data is pushed in from JS).

## Setup

```bash
# 1. one-time: install the wasm target and wasm-pack
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# 2. build the viewer-web wasm module (re-run after editing viewer-web/)
cd web && npm install && npm run build:viewer

# 3. run the app
npx tauri dev   # from src-tauri/, or `cargo tauri dev` if the Tauri CLI is installed
```

## Status — what's real vs. scaffolded

Working end-to-end (Tauri command → React state → wasm viewer):

- `list_trajectory_files` / `load_trajectory` Tauri commands (shallow scan +
  parse of trajectory JSON files).
- Runs panel: scan a workspace root, list trajectory files, open one.
- Replay panel: loads a trajectory, drives a step scrubber, shows episode/
  step stats with the color/hierarchy fixes from the draw.io review (red for
  negative reward and failure, "Step reward" vs. "Episode reward", grouped
  primary-control styling).
- Voxel viewport: wasm build of the camera/painter code, renders geometry
  voxels + step cursor, orbit-drag + scroll-zoom.

Not ported yet (tracked as follow-up work on this branch, not hidden):

- The full Runs/workspace screen — run states (running/stopped/completed),
  resume/stop actions, MLflow linkage, the `experiment.yaml` file tree +
  Monaco editor, live terminal output streaming. Currently just a flat file
  scan.
- Explain tab (`replay/explain.rs`'s `NativeExplainUi`) — still native only,
  `ExplainPanel.tsx` is a placeholder.
- Iteration/trial navigation (Tune-mode multi-trial browsing) — the viewer
  currently loads one trajectory file at a time, not a trial's full
  iteration history.
- Playback (play/pause autoplay through steps) and the reward-curve chart —
  present in the draw.io mock and the sidebar HTML mock, not wired into
  `ReplayPanel.tsx` yet.

## Known tech debt

- `StepData`/`EpisodeData`/`TrajectoryData` are duplicated across
  `voxel_replay.rs`, `viewer-web/src/lib.rs`, and `src-tauri/src/main.rs`
  (as a passthrough `serde_json::Value`). `theseo-core` can't be depended on
  directly from `src-tauri` yet because it always builds as a PyO3
  `cdylib` (`extension-module` feature isn't optional) — factor a shared,
  non-PyO3 `theseo-core-data` crate before this grows further.
- Camera drag/zoom sensitivity constants in `viewer-web` were reconstructed
  to match the native app's *behavior*, not copied from its exact input-
  handling code — worth a side-by-side comparison pass.
