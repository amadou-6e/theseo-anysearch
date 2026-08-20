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
#
#    There is no in-app "type a path" entry point — the window opens on a
#    workspace via `--workspace <path>`, falling back to the process's cwd.
#    `tauri dev`'s dev-server orchestration doesn't forward extra argv, so
#    for a workspace to auto-load in dev mode either run the built binary
#    directly with the flag, or `cd` into the workspace first:
./src-tauri/target/debug/anysearch-shell --workspace /path/to/workspace
# or: (cd /path/to/workspace && npx tauri dev)   # cwd fallback
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

First pass here compared against a text/coordinate dump extracted from the
`.drawio` XML, not an actual rendered image — that caught tab labels and
gross layout placement but missed everything visual. Second pass exported
the real page to PNG (`drawio --export --format png --page-index 4`) and
compared side by side; that surfaced a lot more, but *still* missed two
things a careful look at the same image should have caught the first
time: the Overview window's editor and file-tree panes were in the wrong
left-right order (file tree before the editor; the spec has the editor
immediately next to Run History and the file tree on the far right), and
the run-card column header text below was transcribed from a different,
earlier diagram page in the same file rather than read off "All Windows"
itself. A third pass, done twice over as a deliberate check rather than
once quickly, caught and fixed both — see the correction below the list.

Fixed (second pass):

- Tab labeled "Runs" → **"Overview"** (note `docs/ui/workspace.md`, also
  from feat/197, calls it "Runs" in prose — the two spec sources disagree
  with each other).
- The sidebar's "select a run" hint was worded/placed to only appear
  *after* a run was already selected (backwards) and dropped "Overview"
  from its own text.
- Missing "RUN / STATE / PROGRESS" column header.
- Workspace controls (root path, Rescan, Change workspace) lived in a
  full-width top bar this app invented, instead of inside the file-tree
  pane header where the spec puts them.
- Missing "All states ▾" run-status filter — now a real, working dropdown.
- Run cards had no color-coded state indicator — added a colored dot
  (green running / blue completed / faint otherwise), matching the spec's
  use of color to distinguish state without fabricating the per-card
  Stop/Resume/Details action buttons (see below).
- Missing "WORKSPACE INDEX" summary box at the bottom of the file-tree
  pane — added, wired to real `WorkspaceIndex` counts.
- Missing per-config run-count annotations (spec: `experiment.yaml DQN ·
  3 runs`) — added, computed from `WorkspaceRun.source_yaml`; only
  renders when a scanned workspace's run manifests actually reference a
  visible config file (didn't fire against the `usage/experiments/train`
  test data used here, since those runs' `source_yaml` don't resolve to
  the top-level template files shown — that's the real data, not a bug).
- Sidebar was wider than the spec's proportions (280px → 220px).

Fixed (third pass — the correction referenced above):
- **Overview pane order.** Built as Run history → file tree → editor;
  the spec is Run history → editor+terminal → file tree (the editor sits
  immediately next to Run history; the file tree is the narrower
  right-most column). Swapped in `RunsPanel.tsx`.
- **Run-card column header** read "RUN / STATE / PROGRESS" (copied from
  a different page's XML text, not "All Windows"). The actual "All
  Windows" text is **"EXPERIMENT / RUN ID"** — fixed in
  `RunHistorySidebar.tsx`.

Fixed (fourth pass): a designed "no workspace open" empty-state screen in
`RunsPanel.tsx` — including a dashed "Drop a folder to open it as the
workspace" panel — was removed. This app always opens on a workspace
supplied by the CLI (`initialWorkspace()`); per that architecture there is
no real "empty state" to design a flow around, only an edge case (the
supplied path doesn't resolve) to handle minimally. The prior round had
already been told once to stop building manual-workspace-entry UI (the
raw path input) and, instead of reconsidering the surrounding screen,
patched the one element that was named and left the promotional drop-zone
panel standing — a repeat of the same mistake pattern as the pane-order
miss above, this time on instruction-following rather than visual
inspection. Replaced with a single text line + "Change workspace" button,
nothing more. Verified live by launching with an invalid `--workspace`
path to force the fallback and confirming it renders minimally.

Fixed since (this round):
- **Drag-and-drop a folder onto the window to open it as the workspace** —
  wired via Tauri's `getCurrentWebview().onDragDropEvent`, listened at the
  window level in `App.tsx` so it works from any tab, not just Overview.
  **Not live-verified** — this is a native OS-level drop event, not a
  browser DOM event, and there's no way to simulate an actual OS file drop
  through CDP from this environment. Compiles and follows the documented
  API; someone should drag a real folder onto the window once to confirm.
  (A dashed drop-zone *hint panel* was added alongside this and then
  removed again a round later — see "Fixed (fourth pass)" below. The
  drag-and-drop event handling itself was never the problem.)
- **"Observation source: ● Current replay step / ○ Fictional observation"**
  radio toggle on the Explain tab — built. Picking "Current replay step"
  re-runs the seeded trajectory-step explanation (and is disabled when
  there's no seed); "Fictional observation" reveals the import/manual-edit
  controls. Matches `explain.rs`'s actual behavior: explaining a trajectory
  step computes a report without overwriting the editable observation
  fields, so the toggle governs *which backend call runs*, not what the
  editor displays.
- **The Explain tab's numeric grid editor is now overlaid directly on the
  geometry canvas**, not a separate sidebar-only control — small clickable
  chips are projected onto the active slice's cells using the same
  isometric math the canvas draws with, and click-to-cycle their voxel
  kind. The sidebar's grid-of-dropdowns editor (`LocalGridEditor`) is kept
  alongside it, matching the spec (which shows both surfaces editing the
  same "Active slice values").

Also fixed (this round): **"Start a new run [expand]"** quick-create panel
in the sidebar — expands to a real dropdown of every `anysearch`-classified
config in the workspace and a "Start run" button wired to the same
`startRun` command the Overview tab's editor uses, plus a green
`[running] <config-name>` banner while a run is active (state lifted from
`RunsPanel` to `App.tsx` so the banner can show on every tab, matching the
spec). **Not live-verified past expanding the panel** — clicking "Start
run" for real would launch an actual `anysearch run` training process
against the user's workspace, which this session deliberately avoided
doing the whole way through (same reason Start run/Save were never fired
live in earlier passes).

Still not fixed — genuine missing features/engineering, not alignment nits:
- YAML syntax highlighting and the hover autocomplete tooltip (spec shows
  a real code editor experience; this is a plain `<textarea>`). The
  original `CLAUDE.md` doc names Monaco for this — not attempted here.
- The progress banner shows `[running] <config-name>` only, not the
  spec's `iteration 37 / 100 - 37%` — that needs progress data (current/
  total iteration) this app doesn't have a source for yet: `run.json`
  manifests don't record it in a standard field, and the only live number
  available is parsed terminal output for the run *this app itself*
  started, not arbitrary discovered runs.
- Per-run-card **Stop / Resume / Run again / Details** actions. Checked
  whether these could be wired honestly: `stop_run` kills a `Child` handle
  *this process* spawned via Start run — it has no way to stop an arbitrary
  already-running training process discovered via `run.json` (no PID is
  recorded in the manifest). Left undone rather than shipping buttons that
  don't work.

Also unverified for the same reason as before: the Explain tab's editing
surfaces (scalar fields, the new overlay, `LocalGridEditor`) still haven't
been exercised against a real "ready" payload — no run in this repo has
both a saved checkpoint and a schema-compatible config.

Fixed since (this round): the raw "Workspace root" text-path input --
removed entirely. It was flagged in an earlier pass as "not in the spec,
kept as a secondary affordance," but the actual spec assumption is that the
window already opens inside a workspace, launched the same way the native
shell is (`anysearch ui <path>`, i.e. `cli/main.py`'s `native_ui` command
on feat/197 -- not yet ported to this branch's CLI -- invoking the binary
as `<binary> --workspace <path>` with `cwd=workspace`). Added a real
`initial_workspace` Tauri command that reads `--workspace <path>` from
argv, falling back to the process's current directory (`anysearch ui .`
semantics), and `App.tsx` calls it once on mount and auto-scans. "Change
workspace" (native picker) and drag-and-drop are the only ways to switch
workspaces now, matching the spec. Verified live: launching
`anysearch-shell.exe --workspace <path>` opens straight into that
workspace with no manual entry anywhere. **Not yet wired**: the Python
`anysearch ui` command itself still launches the old egui binary, not this
one -- that CLI change is out of this pass's scope.

Fixed (fifth pass): the "Open trajectories folder" control (manual
path input + Open button + "Trajectories found"/"No trajectories found"
status line) has been removed from `RunHistorySidebar.tsx` entirely. It
was flagged as an unrequested addition — "not in the spec at all" — in
the fourth-pass entry above, and again multiple times afterward in
direct feedback ("I also said a few times that the trajectories stuff
under the runs is not part of the drawio yet its still there") before
actually being removed this pass; documenting a known deviation is not
the same as fixing it, and it should not have taken repeated asks.
Verified live: `document.body.innerText` on the running app contains no
occurrence of "trajector" anywhere, and the sidebar screenshot shows the
run-card list running directly into the "Select a run to open Overview,
Replay, or Explain." footer with nothing in between.

Real, honest consequence of this removal (not silently worked around):
legacy Tune trials/runs with no `run.json` manifest (`trajectories/` is
deliberately excluded from the scanned file tree by `workspace.py`'s
`_workspace_files`) once again have no path into Replay or Explain
through this UI — that gap was the original reason the control was
added. No replacement was built; if a discovery path for those runs is
wanted, it needs its own explicit design (most likely something that
belongs in the actual drawio spec, not an ad hoc sidebar control).

## Scrollbar theming and positioning

Default WebView2/Chromium scrollbars didn't match the dark theme, and
vertical scrollbars for the leftmost (`RunHistorySidebar`) and Explain's
main content pane sat at the boundary shared with the next pane instead
of the window's outer edge. Fixed:

- `tokens.css` styles `::-webkit-scrollbar` (thin, dark thumb using
  `--border`/`--text-faint` on hover, transparent track, no arrow
  buttons). **Deliberately does not also set the standard
  `scrollbar-width`/`scrollbar-color` properties** — on this dev machine
  (WebView2 with an OS "always show scrollbars"-style setting), setting
  both together made Chromium fall back to the classic native scrollbar
  (arrow buttons included) and ignore the `::-webkit-scrollbar-button`
  override; the `-webkit-` pseudo-elements alone render correctly.
  Horizontal scrollbars needed no separate handling — they already
  render along a container's bottom edge by default; only the theming
  applies to them too.
- `RunHistorySidebar.tsx` and `ExplainPanel.tsx`'s main pane wrap their
  content in `direction: rtl` (outer scroll container) /
  `direction: ltr` (inner content wrapper) so their vertical scrollbar
  renders on the pane's outer-left edge instead of its inner edge
  against the next pane, without reversing any internal flex-row
  layouts. The rightmost pane on each tab (file tree / Replay's and
  Explain's right sidebars) already had its scrollbar on the window's
  true right edge by default and needed no change.

Verified live: forced DOM overflow in the sidebar (cloned run cards) and
screenshotted the actual window edges at real size — confirms a thin,
dark, button-free thumb sitting flush against the window's left edge
(previously it sat at the sidebar/main-content boundary, using the
native light-gray style). **Not live-verified**: `ExplainPanel`'s flip
uses the identical proven pattern but couldn't be exercised against its
`available` (checkpoint-connected) branch — no run in this repo has both
a saved checkpoint and a schema-compatible config, the same pre-existing
gap noted elsewhere in this file.

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
