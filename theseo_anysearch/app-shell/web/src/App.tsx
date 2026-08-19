import { useEffect, useState } from "react";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import RunHistorySidebar from "./panels/RunHistorySidebar";
import RunsPanel from "./panels/RunsPanel";
import ReplayPanel from "./panels/ReplayPanel";
import ExplainPanel from "./panels/ExplainPanel";
import {
  initialWorkspace,
  scanWorkspace,
  pickWorkspaceFolder,
  listTrajectoryFiles,
  startRun,
  stopRun,
  runIsActive,
  onRunExited,
  type WorkspaceIndex,
  type WorkspaceRun,
  type TrajectoryFile,
} from "./lib/tauri";

type Tab = "runs" | "replay" | "explain";

// Labeled "Overview" per spec/ui-design/replayer-current.drawio's "All
// Windows" tab (the tab bar reads "Overview | Replay | Explain" there).
// Note: docs/ui/workspace.md, also ported from feat/197, calls this same
// tab "Runs" in prose -- the two spec sources disagree with each other,
// not just with this app. Internal id kept as "runs" (matches the Rust
// commands / this codebase's own vocabulary); only the display label
// changed to follow the visual spec.
const TABS: { id: Tab; label: string }[] = [
  { id: "runs", label: "Overview" },
  { id: "replay", label: "Replay" },
  { id: "explain", label: "Explain" },
];

interface ExplainSeed {
  trajectorySourcePath: string;
  step: number;
}

export default function App() {
  const [tab, setTab] = useState<Tab>("runs");
  const [selected, setSelected] = useState<TrajectoryFile | null>(null);
  const [explainSeed, setExplainSeed] = useState<ExplainSeed | null>(null);

  // Workspace state lives here, not inside RunsPanel, because the run
  // history it drives is a *persistent* sidebar shown on every tab (see
  // RunHistorySidebar / docs/ui/workspace.md), not scoped to one tab.
  const [root, setRoot] = useState("");
  const [index, setIndex] = useState<WorkspaceIndex | null>(null);
  const [scanning, setScanning] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<WorkspaceRun | null>(null);
  const [manualTrajDir, setManualTrajDir] = useState("");
  const [manualTrajError, setManualTrajError] = useState<string | null>(null);

  // Run-lifecycle state lives here too (not just inside RunsPanel) so the
  // sidebar's "[running] <name>" progress banner -- shown per the spec on
  // every tab -- can reflect it. RunsPanel keeps its own terminal-output
  // buffer locally; this is only the "is something running, and what" bit.
  const [runActive, setRunActive] = useState(false);
  const [activeRunLabel, setActiveRunLabel] = useState<string | null>(null);

  useEffect(() => {
    runIsActive().then(setRunActive);
    const unlisten = onRunExited(() => {
      setRunActive(false);
      setActiveRunLabel(null);
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  async function startRunFor(configPath: string) {
    await startRun(root, configPath);
    setRunActive(true);
    setActiveRunLabel(configPath.split(/[/\\]/).pop() ?? configPath);
  }

  async function stopActiveRun() {
    await stopRun();
    setRunActive(false);
    setActiveRunLabel(null);
  }

  // ReplayPanel loads the *entire* iteration history for a trajectory's
  // directory regardless of which file within it you hand it (see
  // ReplayPanel.tsx's loadIterationHistory) -- so there is no real "pick
  // which trajectory file to open" step. Selecting a run should just make
  // Replay/Explain available, per docs/ui/workspace.md: "Replay becomes
  // available after selecting a run containing saved trajectories." This
  // resolves one representative file (preferring best.json, the run's
  // summary) purely to get a valid directory + display name.
  function pickRepresentative(files: TrajectoryFile[]): TrajectoryFile | null {
    return files.find((f) => f.name === "best.json") ?? files[0] ?? null;
  }

  async function rescan(nextRoot: string) {
    if (!nextRoot) return;
    setScanning(true);
    setScanError(null);
    try {
      setRoot(nextRoot);
      setIndex(await scanWorkspace(nextRoot));
    } catch (e) {
      setScanError(String(e));
      setIndex(null);
    } finally {
      setScanning(false);
    }
  }

  // Open on a workspace the same way the native shell does -- via
  // `anysearch ui <path>` (--workspace arg, or the process cwd) -- not an
  // in-app "type a path" entry point. See src-tauri/src/workspace.rs.
  useEffect(() => {
    initialWorkspace().then((path) => {
      if (path) rescan(path);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function changeWorkspace() {
    const folder = await pickWorkspaceFolder();
    if (folder) {
      setSelectedRun(null);
      setSelected(null);
      await rescan(folder);
    }
  }

  // "Drop a folder to open it as the workspace" -- the spec's Overview
  // window shows this as a drop zone; wired here at the window level so it
  // works from any tab, not just Overview. `scanWorkspace` fails (and
  // surfaces the error) harmlessly if the dropped path isn't a directory.
  useEffect(() => {
    const unlisten = getCurrentWebview().onDragDropEvent((event) => {
      if (event.payload.type === "drop") {
        const folder = event.payload.paths[0];
        if (folder) {
          setSelectedRun(null);
          setSelected(null);
          rescan(folder);
        }
      }
    });
    return () => {
      unlisten.then((fn) => fn());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function selectRun(run: WorkspaceRun) {
    setSelectedRun(run);
    setManualTrajError(null);
    try {
      const files = await listTrajectoryFiles(`${root}/${run.path}`);
      setSelected(pickRepresentative(files));
    } catch {
      setSelected(null);
    }
  }

  // Run artifacts (trajectories/) are deliberately excluded from the
  // scanned workspace tree, and Tune-trial sweep dirs (ray_runtime.json
  // keyed, no run.json) never get a WorkspaceRun row -- mirrors the native
  // CLI's --tune-dir/file-mode entry points. See RunHistorySidebar.
  async function openTrajectoriesDir() {
    if (!manualTrajDir) return;
    setManualTrajError(null);
    setSelectedRun(null);
    try {
      const files = await listTrajectoryFiles(manualTrajDir);
      const entry = pickRepresentative(files);
      if (!entry) setManualTrajError("No trajectory files found under that path.");
      setSelected(entry);
    } catch (e) {
      setManualTrajError(String(e));
      setSelected(null);
    }
  }

  function openTrajectory(file: TrajectoryFile) {
    setSelected(file);
    setExplainSeed(null);
    setTab("replay");
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <header
        style={{
          height: 64,
          borderBottom: "1px solid var(--border-soft)",
          display: "flex",
          alignItems: "center",
          gap: 16,
          padding: "0 22px",
          flexShrink: 0,
        }}
      >
        <div style={{ fontSize: 20, fontWeight: 600 }}>AnySearch</div>
        <div
          style={{
            background: "var(--panel)",
            border: "1px solid var(--border)",
            borderRadius: 5,
            padding: "8px 14px",
            fontSize: 12,
            color: "var(--text-dim)",
          }}
        >
          Workspace: {index ? (root.split(/[/\\]/).pop() ?? root) : "none"}
        </div>

        {selectedRun && (
          <div style={{ fontSize: 11, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Selected run - {selectedRun.run_id} - {selectedRun.status}
          </div>
        )}

        <nav style={{ display: "flex", gap: 4, marginLeft: "auto" }}>
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              disabled={t.id !== "runs" && !selected}
              style={{
                background: tab === t.id ? "var(--panel-raised)" : "transparent",
                border: tab === t.id ? "1px solid var(--blue)" : "1px solid transparent",
                color: tab === t.id ? "#fff" : "var(--text-dim)",
                borderRadius: 5,
                padding: "8px 16px",
                fontSize: 12.5,
                fontWeight: 600,
                cursor: t.id !== "runs" && !selected ? "not-allowed" : "pointer",
                opacity: t.id !== "runs" && !selected ? 0.4 : 1,
              }}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <div style={{ flex: 1, minHeight: 0, display: "flex" }}>
        <RunHistorySidebar
          root={root}
          index={index}
          selectedRun={selectedRun}
          hasReplayData={!!selected}
          manualTrajDir={manualTrajDir}
          onManualTrajDirChange={setManualTrajDir}
          manualTrajError={manualTrajError}
          onSelectRun={selectRun}
          onOpenTrajectoriesDir={openTrajectoriesDir}
          runActive={runActive}
          activeRunLabel={activeRunLabel}
          onStartRun={startRunFor}
        />

        <main style={{ flex: 1, minWidth: 0 }}>
          {tab === "runs" && (
            <RunsPanel
              root={root}
              index={index}
              scanning={scanning}
              error={scanError}
              onRescan={rescan}
              onChangeWorkspace={changeWorkspace}
              onOpenTrajectory={openTrajectory}
              runActive={runActive}
              onStartRun={startRunFor}
              onStopRun={stopActiveRun}
            />
          )}
          {tab === "replay" && selected && (
            <ReplayPanel
              file={selected}
              onExplainStep={(trajectorySourcePath, step) => {
                setExplainSeed({ trajectorySourcePath, step });
                setTab("explain");
              }}
            />
          )}
          {tab === "explain" && selected && <ExplainPanel file={selected} seed={explainSeed} />}
        </main>
      </div>
    </div>
  );
}
