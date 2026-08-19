import { useState } from "react";
import {
  scanWorkspace,
  pickWorkspaceFolder,
  listTrajectoryFiles,
  readTextFile,
  type WorkspaceIndex,
  type WorkspaceRun,
  type FileKind,
  type TrajectoryFile,
} from "../lib/tauri";

// Runs tab: workspace-level, matches theseo_anysearch/ui/workspace.py's
// WorkspaceIndex contract (see docs/ui/workspace.md on feat/197) — run
// history reconstructed from run.json manifests (left pane) plus the
// workspace file tree with config classification (right pane), same as the
// native egui shell's WorkspaceUi. Not ported yet: the YAML code editor with
// Validate/Save/Start run/Stop and live terminal streaming — this pass reads
// files read-only. See app-shell/README.md.

const KIND_MARKER: Record<FileKind, string> = {
  anysearch: "◆",
  invalid_anysearch: "!",
  yaml: "◇",
  file: "",
};

const KIND_COLOR: Record<FileKind, string> = {
  anysearch: "var(--blue)",
  invalid_anysearch: "var(--red)",
  yaml: "var(--text-dim)",
  file: "var(--text-faint)",
};

export default function RunsPanel({ onOpenTrajectory }: { onOpenTrajectory: (file: TrajectoryFile) => void }) {
  const [root, setRoot] = useState("");
  const [index, setIndex] = useState<WorkspaceIndex | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const [search, setSearch] = useState("");
  const [selectedRun, setSelectedRun] = useState<WorkspaceRun | null>(null);
  const [runTrajectories, setRunTrajectories] = useState<TrajectoryFile[]>([]);
  const [previewPath, setPreviewPath] = useState<string | null>(null);
  const [previewText, setPreviewText] = useState<string>("");

  async function rescan(nextRoot: string) {
    if (!nextRoot) return;
    setScanning(true);
    setError(null);
    try {
      setIndex(await scanWorkspace(nextRoot));
    } catch (e) {
      setError(String(e));
      setIndex(null);
    } finally {
      setScanning(false);
    }
  }

  async function changeWorkspace() {
    const folder = await pickWorkspaceFolder();
    if (folder) {
      setRoot(folder);
      setSelectedRun(null);
      setRunTrajectories([]);
      await rescan(folder);
    }
  }

  async function selectRun(run: WorkspaceRun) {
    setSelectedRun(run);
    try {
      setRunTrajectories(await listTrajectoryFiles(`${root}/${run.path}`));
    } catch {
      setRunTrajectories([]);
    }
  }

  async function previewFile(path: string) {
    setPreviewPath(path);
    try {
      setPreviewText(await readTextFile(`${root}/${path}`));
    } catch (e) {
      setPreviewText(String(e));
    }
  }

  const filteredFiles = (index?.files ?? []).filter(
    (f) => !search || f.path.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ display: "flex", gap: 8, padding: 14, borderBottom: "1px solid var(--border-soft)", flexShrink: 0 }}>
        <input
          value={root}
          onChange={(e) => setRoot(e.target.value)}
          placeholder="Workspace root"
          style={inputStyle()}
        />
        <button onClick={() => rescan(root)} disabled={!root || scanning} style={btnStyle("var(--blue)")}>
          {scanning ? "Scanning…" : "Rescan"}
        </button>
        <button onClick={changeWorkspace} style={btnStyle("#232323")}>
          Change workspace
        </button>
      </div>

      {error && <div style={{ color: "var(--red)", fontSize: 12, padding: "8px 14px", fontFamily: "var(--mono)" }}>{error}</div>}

      {index && (
        <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
          {/* Left: run history, reconstructed from run.json manifests. */}
          <div style={{ width: 320, borderRight: "1px solid var(--border-soft)", overflowY: "auto", padding: 14, flexShrink: 0 }}>
            <div style={groupLabel()}>Run history</div>
            <div style={{ fontSize: 11, color: "var(--text-faint)", marginBottom: 10 }}>
              {index.configuration_count} configurations · {index.runs.length} runs
            </div>
            {index.runs.length === 0 && (
              <div style={{ fontSize: 12, color: "var(--text-faint)" }}>
                No run.json manifests found yet — runs appear here after `anysearch run`.
              </div>
            )}
            {index.runs.map((run) => (
              <div
                key={run.path}
                onClick={() => selectRun(run)}
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

            {selectedRun && (
              <>
                <div style={{ ...groupLabel(), marginTop: 18 }}>Trajectories</div>
                {runTrajectories.length === 0 && (
                  <div style={{ fontSize: 12, color: "var(--text-faint)" }}>None found in this run.</div>
                )}
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
          </div>

          {/* Right: workspace file tree, with config classification markers. */}
          <div style={{ width: 320, borderRight: "1px solid var(--border-soft)", overflowY: "auto", padding: 14, flexShrink: 0 }}>
            <div style={groupLabel()}>Workspace files</div>
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search files…"
              style={{ ...inputStyle(), width: "100%", marginBottom: 10 }}
            />
            <div style={{ fontSize: 11, color: "var(--text-faint)", marginBottom: 10 }}>
              {index.file_count} files · {index.yaml_count} yaml · {index.invalid_configuration_count} invalid
            </div>
            {filteredFiles.map((f) => (
              <div
                key={f.path}
                onClick={() => previewFile(f.path)}
                style={{
                  display: "flex",
                  gap: 6,
                  padding: "4px 6px",
                  borderRadius: 4,
                  cursor: "pointer",
                  background: previewPath === f.path ? "var(--panel-raised)" : "transparent",
                  fontSize: 11.5,
                  fontFamily: "var(--mono)",
                }}
              >
                <span style={{ color: KIND_COLOR[f.kind], width: 12 }}>{KIND_MARKER[f.kind]}</span>
                <span style={{ color: "var(--text-dim)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {f.path}
                </span>
              </div>
            ))}
          </div>

          {/* Preview pane. */}
          <div style={{ flex: 1, overflowY: "auto", padding: 14 }}>
            {previewPath ? (
              <>
                <div style={{ ...groupLabel(), marginBottom: 8 }}>{previewPath}</div>
                <pre
                  style={{
                    fontFamily: "var(--mono)",
                    fontSize: 12,
                    color: "var(--text)",
                    background: "#0c0e12",
                    border: "1px solid var(--border-soft)",
                    borderRadius: 5,
                    padding: 14,
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                  }}
                >
                  {previewText}
                </pre>
              </>
            ) : (
              <div style={{ color: "var(--text-faint)", fontSize: 12 }}>Select a file to preview it.</div>
            )}
          </div>
        </div>
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
