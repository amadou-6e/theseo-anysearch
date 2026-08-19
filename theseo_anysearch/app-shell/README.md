# AnySearch UI shell (feat/200)

Tauri + React/TypeScript replacement for the `egui` desktop UI, tracked by
[#200](https://github.com/amadou-6e/theseo-anysearch/issues/200). Design
reference: `spec/ui-design/replayer-current.drawio` (tabs "Replay
(Optimized)" and "Sidebar Mock (HTML)").

## Architecture

- `src-tauri/` — Tauri backend. Exposes workspace and trajectory data to the
  frontend via `invoke()` commands.
- `web/` — React/TypeScript/Vite frontend. All panels except the voxel
  viewport (Runs, Explain, config editor, terminal — per the draw.io design).
- `viewer-web/` — the simulation/voxel viewport, staying in `egui` per the
  architecture decision, compiled to WebAssembly (`eframe`'s web target) and
  mounted as a `<canvas>` inside the React `ReplayPanel`. This is a trimmed,
  faithful port of the camera/projection/painter code in
  `theseo_anysearch/core/src/bin/voxel_replay.rs` — same math, same draw
  order, no file I/O (data is pushed in from JS).
- `theseo_anysearch/ui/{service,workspace}.py` — the authoritative workspace
  scanner (ported from `feat/197`'s `2bd4f62`, not reimplemented). `Runs`
  shells out to `python -m theseo_anysearch.ui.service scan <root>` exactly
  like `feat/197`'s native `WorkspaceUi` does, so both UIs discover the same
  `run.json` manifests and classify the same YAML files the same way. See
  `docs/ui/workspace.md` (also from `feat/197`) for the full contract.

## Setup

```bash
# 1. one-time: install the wasm target and wasm-pack
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# 2. build the viewer-web wasm module (re-run after editing viewer-web/)
cd web && npm install && npm run build:viewer

# 3. run the app (needs Python + this package importable for the Runs tab's
#    workspace scan — set ANYSEARCH_PYTHON if `python` on PATH isn't the
#    right interpreter, same convention as feat/197's native shell)
npx tauri dev   # from src-tauri/, or `cargo tauri dev` if the Tauri CLI is installed
```

## Status — what's real vs. scaffolded

Working end-to-end, verified by actually launching the app (WebView2 remote
debugging + CDP) and driving it, not just compiling it:

- **Runs tab**: `scan_workspace` shells out to the real
  `theseo_anysearch.ui.service` Python backend (ported from `feat/197`'s
  `2bd4f62`, not reimplemented) — same `run.json`-derived run history, same
  YAML classification (◆ valid config / `!` invalid / ◇ ordinary yaml) as
  the native egui shell. "Change workspace" uses the native folder picker
  (`@tauri-apps/plugin-dialog`). Selecting a run lists its trajectory files
  for opening in Replay. Clicking a file previews its raw text.
- **Replay tab**: loads a trajectory, drives a step scrubber, shows episode/
  step stats with the color/hierarchy fixes from the draw.io review (red for
  negative reward and failure, "Step reward" vs. "Episode reward", grouped
  primary-control styling).
- **Voxel viewport**: wasm build of the camera/painter code, renders the
  step cursor and geometry voxels (when present), orbit-drag + scroll-zoom.

Not ported yet (tracked as follow-up work on this branch, not hidden):

- The workspace file tree is a flat, filtered list (grouped visually by
  path prefix through sort order), not the native shell's collapsible
  directory tree.
- No YAML code editor / Validate / Save / Start run / Stop / live terminal
  streaming — `workspace.rs`'s `WorkspaceUi` has all of this (spawns
  `python -m theseo_anysearch.cli.main run`, streams stdout/stderr); this
  shell only previews files read-only so far.
- Explain tab (`replay/explain.rs`'s `NativeExplainUi`) — still native only,
  `ExplainPanel.tsx` is a placeholder.
- Iteration/trial navigation (Tune-mode multi-trial browsing) — the viewer
  currently loads one trajectory file at a time, not a trial's full
  iteration history.
- Playback (play/pause autoplay through steps) and the reward-curve chart —
  present in the draw.io mock and the sidebar HTML mock, not wired into
  `ReplayPanel.tsx` yet.
- `load_trajectory` doesn't resolve `episode.init_filled_file` (a separate
  `.npy` geometry reference some trajectories use instead of inline
  coordinates) — geometry voxels won't render for those until this is added.

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
- The Runs tab shells out to Python per scan/validate call, same as
  `workspace.rs` does — fine for the workspace sizes exercised so far, but
  worth watching if scans get slow on very large workspaces.
