import { useState } from "react";
import type { WorkspaceIndex, WorkspaceRun } from "../lib/tauri";

const STATUS_COLOR: Record<string, string> = {
  running: "var(--green)",
  completed: "var(--blue)",
  stopped: "var(--text-faint)",
  failed: "var(--red)",
};

function statusColor(status: string): string {
  return STATUS_COLOR[status.toLowerCase()] ?? "var(--text-faint)";
}

// Persistent left pane, present on Runs/Replay/Explain alike (per
// docs/ui/workspace.md, ported from feat/197: "The run history is the left
// pane and remains present on Runs, Replay, and Explain so run context does
// not move when tabs change") -- matches spec/ui-design/replayer-current.drawio's
// "All Windows" tab, which shows the same run-history column at x=40 across
// all three window mockups.
//
// Selecting a run does NOT show a list of its trajectory files to pick from
// -- ReplayPanel loads the *entire* iteration history for a trajectory
// directory regardless of which file within it it's handed, so there is no
// real "pick a file" step. Per docs/ui/workspace.md: "Replay becomes
// available after selecting a run containing saved trajectories" -- that's
// the whole interaction. Iteration/step navigation happens with the
// Iterations/Steps sliders inside Replay itself.
export default function RunHistorySidebar({
  root,
  index,
  selectedRun,
  onSelectRun,
  runActive,
  activeRunLabel,
  onStartRun,
}: {
  root: string;
  index: WorkspaceIndex | null;
  selectedRun: WorkspaceRun | null;
  onSelectRun: (run: WorkspaceRun) => void;
  runActive: boolean;
  activeRunLabel: string | null;
  onStartRun: (configPath: string) => Promise<void>;
}) {
  const [statusFilter, setStatusFilter] = useState("all");
  const [newRunExpanded, setNewRunExpanded] = useState(false);
  const [newRunConfig, setNewRunConfig] = useState("");
  const [newRunError, setNewRunError] = useState<string | null>(null);
  const statuses = index ? [...new Set(index.runs.map((r) => r.status))].sort() : [];
  const visibleRuns = index ? index.runs.filter((r) => statusFilter === "all" || r.status === statusFilter) : [];
  const startableConfigs = index ? index.files.filter((f) => f.kind === "anysearch") : [];

  async function handleStartNewRun() {
    if (!newRunConfig) return;
    setNewRunError(null);
    try {
      await onStartRun(`${root}/${newRunConfig}`);
      setNewRunExpanded(false);
    } catch (e) {
      setNewRunError(String(e));
    }
  }

  return (
    // direction: rtl on the scroll container + direction: ltr on the inner
    // wrapper flips the scrollbar to this pane's outer (left) edge instead
    // of its inner edge against the main content, without reversing any of
    // the flex-row layouts inside.
    <div style={{ width: 220, borderRight: "1px solid var(--border-soft)", overflowY: "auto", flexShrink: 0, direction: "rtl" }}>
    <div style={{ direction: "ltr", padding: 14 }}>
      {index && index.runs.length > 0 && !!startableConfigs.length && (
        <div style={{ marginBottom: 14 }}>
          <button onClick={() => setNewRunExpanded((v) => !v)} style={{ ...btnStyle("#232323", true), width: "100%" }}>
            Start a new run {newRunExpanded ? "▴" : "[expand]"}
          </button>
          {newRunExpanded && (
            <div style={{ marginTop: 8 }}>
              <select
                value={newRunConfig}
                onChange={(e) => setNewRunConfig(e.target.value)}
                style={{ width: "100%", background: "#1f1f1f", border: "1px solid var(--border)", borderRadius: 4, color: "var(--text-dim)", fontSize: 11, padding: "5px 4px", marginBottom: 6 }}
              >
                <option value="">Pick a config…</option>
                {startableConfigs.map((f) => (
                  <option key={f.path} value={f.path}>
                    {f.path}
                  </option>
                ))}
              </select>
              <button onClick={handleStartNewRun} disabled={!newRunConfig || runActive} style={{ ...btnStyle("var(--blue)", true), width: "100%" }}>
                Start run
              </button>
              {newRunError && <div style={{ color: "var(--red)", fontSize: 10.5, marginTop: 4 }}>{newRunError}</div>}
            </div>
          )}
        </div>
      )}

      {runActive && (
        <div
          style={{
            marginBottom: 14,
            padding: "8px 10px",
            borderRadius: 5,
            background: "#173e2b",
            border: "1px solid #2d8b5b",
            color: "#b9e7cd",
            fontSize: 11,
          }}
        >
          [running] {activeRunLabel ?? "…"}
        </div>
      )}

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 10 }}>
        <div style={{ ...groupLabel(), marginBottom: 0 }}>Run history</div>
        {index && index.runs.length > 0 && (
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            style={{ background: "#1f1f1f", border: "1px solid var(--border)", borderRadius: 4, color: "var(--text-dim)", fontSize: 10, padding: "2px 3px" }}
          >
            <option value="all">All states</option>
            {statuses.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        )}
      </div>
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
              Experiment / run ID
            </div>
          )}
          {visibleRuns.map((run) => (
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
              <div style={{ color: "var(--text-dim)", marginTop: 2, display: "flex", alignItems: "center", gap: 5 }}>
                <span style={{ color: statusColor(run.status), fontSize: 8 }}>●</span>
                {run.status} · {run.algorithm ?? "unknown algorithm"}
              </div>
              <div style={{ color: "var(--text-faint)", fontFamily: "var(--mono)", marginTop: 2, fontSize: 10, wordBreak: "break-all" }}>{run.path}</div>
            </div>
          ))}

          {/* Static footer hint, always shown once a workspace is open --
              per the spec, not conditioned on a run already being selected
              (that reads backwards: "select a run" after one is selected). */}
          <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 18, paddingTop: 14, borderTop: "1px solid var(--border-soft)" }}>
            Select a run to open Overview, Replay, or Explain.
          </div>
        </>
      )}
    </div>
    </div>
  );
}

function groupLabel(): React.CSSProperties {
  return { fontSize: 10, fontWeight: 700, letterSpacing: "0.07em", color: "var(--text-faint)", textTransform: "uppercase", marginBottom: 10 };
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
