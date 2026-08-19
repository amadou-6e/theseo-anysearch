import type { WorkspaceIndex, WorkspaceRun, TrajectoryFile } from "../lib/tauri";

// Persistent left pane, present on Runs/Replay/Explain alike (per
// docs/ui/workspace.md, ported from feat/197: "The run history is the left
// pane and remains present on Runs, Replay, and Explain so run context does
// not move when tabs change") -- matches spec/ui-design/replayer-current.drawio's
// "All Windows" tab, which shows the same run-history column at x=40 across
// all three window mockups.
export default function RunHistorySidebar({
  index,
  selectedRun,
  runTrajectories,
  manualTrajDir,
  onManualTrajDirChange,
  manualTrajError,
  onSelectRun,
  onOpenTrajectoriesDir,
  onOpenTrajectory,
}: {
  index: WorkspaceIndex | null;
  selectedRun: WorkspaceRun | null;
  runTrajectories: TrajectoryFile[];
  manualTrajDir: string;
  onManualTrajDirChange: (v: string) => void;
  manualTrajError: string | null;
  onSelectRun: (run: WorkspaceRun) => void;
  onOpenTrajectoriesDir: () => void;
  onOpenTrajectory: (f: TrajectoryFile) => void;
}) {
  return (
    <div style={{ width: 280, borderRight: "1px solid var(--border-soft)", overflowY: "auto", padding: 14, flexShrink: 0 }}>
      <div style={groupLabel()}>Run history</div>
      {!index ? (
        <div style={{ fontSize: 12, color: "var(--text-faint)" }}>No workspace open yet — open one from the Overview tab.</div>
      ) : (
        <>
          <div style={{ fontSize: 11, color: "var(--text-faint)", marginBottom: 10 }}>
            {index.configuration_count} configurations · {index.runs.length} runs
          </div>
          {index.runs.length === 0 && (
            <div style={{ fontSize: 12, color: "var(--text-faint)" }}>
              No run.json manifests found yet — runs appear here after `anysearch run`.
            </div>
          )}
          {index.runs.length > 0 && (
            <div style={{ fontSize: 9.5, fontWeight: 700, letterSpacing: "0.06em", color: "var(--text-faint)", textTransform: "uppercase", marginBottom: 6 }}>
              Run / state / progress
            </div>
          )}
          {index.runs.map((run) => (
            <div
              key={run.path}
              onClick={() => onSelectRun(run)}
              style={{
                padding: "10px 12px",
                marginBottom: 6,
                borderRadius: 5,
                cursor: "pointer",
                background: selectedRun?.path === run.path ? "var(--blue-soft, #1d3150)" : "var(--panel)",
                border: `1px solid ${selectedRun?.path === run.path ? "var(--blue)" : "var(--border-soft)"}`,
                fontSize: 11.5,
              }}
            >
              <div style={{ fontWeight: 600, color: "var(--text)" }}>{run.run_id}</div>
              <div style={{ color: "var(--text-dim)", marginTop: 2 }}>
                {run.status} · {run.algorithm ?? "unknown algorithm"}
              </div>
              <div style={{ color: "var(--text-faint)", fontFamily: "var(--mono)", marginTop: 2 }}>{run.path}</div>
            </div>
          ))}

          <div style={{ ...groupLabel(), marginTop: 18 }}>Open trajectories folder</div>
          <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginBottom: 6 }}>
            For runs/Tune trials with no run.json manifest (not shown above) — paste a run or trial directory.
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <input
              value={manualTrajDir}
              onChange={(e) => onManualTrajDirChange(e.target.value)}
              placeholder="Run or trial directory"
              style={{ ...inputStyle(), fontSize: 11 }}
            />
            <button onClick={onOpenTrajectoriesDir} disabled={!manualTrajDir} style={btnStyle("#232323", true)}>
              Open
            </button>
          </div>
          {manualTrajError && <div style={{ color: "var(--red)", fontSize: 11, marginTop: 6 }}>{manualTrajError}</div>}

          {(selectedRun || runTrajectories.length > 0) && (
            <>
              <div style={{ ...groupLabel(), marginTop: 18 }}>Trajectories</div>
              {runTrajectories.length === 0 && <div style={{ fontSize: 12, color: "var(--text-faint)" }}>None found.</div>}
              {runTrajectories.map((f) => (
                <div
                  key={f.path}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    padding: "6px 10px",
                    marginBottom: 4,
                    borderRadius: 4,
                    background: "var(--panel)",
                    border: "1px solid var(--border-soft)",
                    fontSize: 11.5,
                  }}
                >
                  <span style={{ fontFamily: "var(--mono)" }}>{f.name}</span>
                  <button onClick={() => onOpenTrajectory(f)} style={btnStyle("#232323", true)}>
                    Open
                  </button>
                </div>
              ))}
            </>
          )}

          {/* Static footer hint, always shown once a workspace is open --
              per the spec, not conditioned on a run already being selected
              (that reads backwards: "select a run" after one is selected). */}
          <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 18, paddingTop: 14, borderTop: "1px solid var(--border-soft)" }}>
            Select a run to open Overview, Replay, or Explain.
          </div>
        </>
      )}
    </div>
  );
}

function groupLabel(): React.CSSProperties {
  return { fontSize: 10, fontWeight: 700, letterSpacing: "0.07em", color: "var(--text-faint)", textTransform: "uppercase", marginBottom: 10 };
}

function inputStyle(): React.CSSProperties {
  return {
    flex: 1,
    background: "var(--panel)",
    border: "1px solid var(--border)",
    borderRadius: 5,
    color: "var(--text)",
    padding: "8px 10px",
    fontSize: 12.5,
  };
}

function btnStyle(bg: string, small = false): React.CSSProperties {
  return {
    background: bg,
    border: bg === "#232323" ? "1px solid var(--border)" : "none",
    borderRadius: 5,
    color: bg === "var(--blue)" ? "#fff" : "var(--text)",
    padding: small ? "4px 10px" : "8px 16px",
    fontWeight: 600,
    fontSize: small ? 11 : 12.5,
    whiteSpace: "nowrap",
  };
}
