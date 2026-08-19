import { useEffect, useState } from "react";
import {
  scanWorkspace,
  pickWorkspaceFolder,
  listTrajectoryFiles,
  readTextFile,
  writeTextFile,
  validateConfiguration,
  startRun,
  stopRun,
  runIsActive,
  onRunOutput,
  onRunExited,
  type WorkspaceIndex,
  type WorkspaceRun,
  type WorkspaceFile,
  type FileKind,
  type Diagnostic,
  type TrajectoryFile,
} from "../lib/tauri";

// Runs tab: workspace-level, matches theseo_anysearch/ui/workspace.py's
// WorkspaceIndex contract and workspace.rs's WorkspaceUi (both feat/197) --
// run history reconstructed from run.json manifests (left pane), a
// collapsible workspace file tree with config classification (middle pane),
// and a YAML editor with Validate/Save/Start run/Stop plus streamed terminal
// output (right pane), same operations as the native egui shell.

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

const EDITABLE_KINDS: FileKind[] = ["anysearch", "invalid_anysearch"];

interface TreeDir {
  dirs: Map<string, TreeDir>;
  files: WorkspaceFile[];
}

function buildTree(files: WorkspaceFile[]): TreeDir {
  const root: TreeDir = { dirs: new Map(), files: [] };
  for (const file of files) {
    const parts = file.path.split("/");
    let node = root;
    for (let i = 0; i < parts.length - 1; i++) {
      const part = parts[i];
      if (!node.dirs.has(part)) node.dirs.set(part, { dirs: new Map(), files: [] });
      node = node.dirs.get(part)!;
    }
    node.files.push(file);
  }
  return root;
}

export default function RunsPanel({ onOpenTrajectory }: { onOpenTrajectory: (file: TrajectoryFile) => void }) {
  const [root, setRoot] = useState("");
  const [index, setIndex] = useState<WorkspaceIndex | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const [search, setSearch] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const [selectedRun, setSelectedRun] = useState<WorkspaceRun | null>(null);
  const [runTrajectories, setRunTrajectories] = useState<TrajectoryFile[]>([]);
  const [manualTrajDir, setManualTrajDir] = useState("");
  const [manualTrajError, setManualTrajError] = useState<string | null>(null);

  const [selectedFile, setSelectedFile] = useState<WorkspaceFile | null>(null);
  const [editorText, setEditorText] = useState("");
  const [savedText, setSavedText] = useState("");
  const [openableTrajectory, setOpenableTrajectory] = useState<TrajectoryFile | null>(null);
  const [diagnostics, setDiagnostics] = useState<Diagnostic[]>([]);
  const [validating, setValidating] = useState(false);
  const [saving, setSaving] = useState(false);

  const [runActive, setRunActive] = useState(false);
  const [terminal, setTerminal] = useState<string[]>([]);

  useEffect(() => {
    let unlistenOutput: (() => void) | undefined;
    let unlistenExited: (() => void) | undefined;
    onRunOutput((line) => setTerminal((t) => [...t.slice(-1999), line])).then((fn) => (unlistenOutput = fn));
    onRunExited((status) => {
      setRunActive(false);
      setTerminal((t) => [...t.slice(-1999), `[process exited: ${status}]`]);
      if (root) rescan(root);
    }).then((fn) => (unlistenExited = fn));
    runIsActive().then(setRunActive);
    return () => {
      unlistenOutput?.();
      unlistenExited?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
      setSelectedFile(null);
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

  // Run artifacts (trajectories/) are deliberately excluded from the scanned
  // workspace tree (theseo_anysearch/ui/workspace.py's `_workspace_files`:
  // "loaded on demand from the selected run rather than held in the index"),
  // and Tune-trial sweep dirs (ray_runtime.json-keyed, no run.json) never
  // get a WorkspaceRun entry at all -- so they have no Run history row to
  // click. This mirrors the native CLI's `--tune-dir`/file-mode entry point:
  // open any directory's trajectories directly by path.
  async function openTrajectoriesDir() {
    if (!manualTrajDir) return;
    setManualTrajError(null);
    setSelectedRun(null);
    try {
      const files = await listTrajectoryFiles(manualTrajDir);
      if (files.length === 0) setManualTrajError("No trajectory files found under that path.");
      setRunTrajectories(files);
    } catch (e) {
      setManualTrajError(String(e));
      setRunTrajectories([]);
    }
  }

  async function selectFile(f: WorkspaceFile) {
    setSelectedFile(f);
    setDiagnostics(f.diagnostics);
    setOpenableTrajectory(null);
    try {
      const text = await readTextFile(`${root}/${f.path}`);
      setEditorText(text);
      setSavedText(text);
      // Trajectory JSON files (best.json / iter_*.json) aren't classified as
      // "anysearch" configs, but should still be openable in Replay -- this
      // is the only path into Replay for runs with no run.json manifest
      // (e.g. legacy Tune trials keyed by ray_runtime.json instead).
      if (f.path.endsWith(".json")) {
        try {
          const parsed = JSON.parse(text);
          if (parsed && typeof parsed === "object" && "episode" in parsed) {
            setOpenableTrajectory({ name: f.path.split("/").pop() ?? f.path, path: `${root}/${f.path}` });
          }
        } catch {
          // not JSON / not a trajectory -- ignore
        }
      }
    } catch (e) {
      setEditorText(String(e));
      setSavedText(String(e));
    }
  }

  async function validate() {
    if (!selectedFile) return;
    setValidating(true);
    try {
      const result = await validateConfiguration(root, `${root}/${selectedFile.path}`);
      setDiagnostics(result.diagnostics);
    } catch (e) {
      setError(String(e));
    } finally {
      setValidating(false);
    }
  }

  async function save() {
    if (!selectedFile) return;
    setSaving(true);
    try {
      await writeTextFile(`${root}/${selectedFile.path}`, editorText);
      setSavedText(editorText);
      await rescan(root);
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  }

  async function start() {
    if (!selectedFile) return;
    setTerminal([]);
    try {
      await startRun(root, `${root}/${selectedFile.path}`);
      setRunActive(true);
    } catch (e) {
      setError(String(e));
    }
  }

  async function stop() {
    await stopRun();
    setRunActive(false);
  }

  function toggleDir(path: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }

  const files = index?.files ?? [];
  const filteredFiles = search ? files.filter((f) => f.path.toLowerCase().includes(search.toLowerCase())) : files;
  const tree = buildTree(filteredFiles);
  const isEditable = selectedFile && EDITABLE_KINDS.includes(selectedFile.kind);
  const dirty = editorText !== savedText;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ display: "flex", gap: 8, padding: 14, borderBottom: "1px solid var(--border-soft)", flexShrink: 0 }}>
        <input value={root} onChange={(e) => setRoot(e.target.value)} placeholder="Workspace root" style={inputStyle()} />
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
          <div style={{ width: 300, borderRight: "1px solid var(--border-soft)", overflowY: "auto", padding: 14, flexShrink: 0 }}>
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

            <div style={{ ...groupLabel(), marginTop: 18 }}>Open trajectories folder</div>
            <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginBottom: 6 }}>
              For runs/Tune trials with no run.json manifest (not shown above) -- paste a run or trial directory.
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              <input
                value={manualTrajDir}
                onChange={(e) => setManualTrajDir(e.target.value)}
                placeholder="Run or trial directory"
                style={{ ...inputStyle(), fontSize: 11 }}
              />
              <button onClick={openTrajectoriesDir} disabled={!manualTrajDir} style={btnStyle("#232323", true)}>
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
          </div>

          {/* Middle: collapsible workspace file tree, with config classification markers. */}
          <div style={{ width: 300, borderRight: "1px solid var(--border-soft)", overflowY: "auto", padding: 14, flexShrink: 0 }}>
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
            <DirTree
              node={tree}
              prefix=""
              expanded={expanded}
              onToggle={toggleDir}
              selectedPath={selectedFile?.path ?? null}
              onSelectFile={selectFile}
              forceOpen={!!search}
            />
          </div>

          {/* Right: editor (for recognized/invalid configs) or read-only preview, plus terminal. */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
            <div style={{ flex: "1 1 55%", minHeight: 0, display: "flex", flexDirection: "column", padding: 14, borderBottom: "1px solid var(--border-soft)" }}>
              {selectedFile ? (
                <>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                    <div style={{ ...groupLabel(), marginBottom: 0 }}>
                      {selectedFile.path}
                      {dirty && <span style={{ color: "var(--blue)" }}> ●</span>}
                    </div>
                    <div style={{ display: "flex", gap: 6 }}>
                      {openableTrajectory && (
                        <button onClick={() => onOpenTrajectory(openableTrajectory)} style={btnStyle("var(--blue)", true)}>
                          Open in Replay
                        </button>
                      )}
                      {isEditable && (
                        <>
                          <button onClick={validate} disabled={validating} style={btnStyle("#232323", true)}>
                            {validating ? "Validating…" : "Validate"}
                          </button>
                          <button onClick={save} disabled={saving || !dirty} style={btnStyle("#232323", true)}>
                            {saving ? "Saving…" : "Save"}
                          </button>
                          <button onClick={start} disabled={runActive} style={btnStyle("var(--blue)", true)}>
                            Start run
                          </button>
                          <button onClick={stop} disabled={!runActive} style={btnStyle("#3b272b", true)}>
                            Stop
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                  {diagnostics.length > 0 && (
                    <div style={{ marginBottom: 8 }}>
                      {diagnostics.map((d, i) => (
                        <div key={i} style={{ color: "var(--red)", fontSize: 11.5, fontFamily: "var(--mono)" }}>
                          {d.path}: {d.message}
                        </div>
                      ))}
                    </div>
                  )}
                  {isEditable ? (
                    <textarea
                      value={editorText}
                      onChange={(e) => setEditorText(e.target.value)}
                      spellCheck={false}
                      style={{
                        flex: 1,
                        fontFamily: "var(--mono)",
                        fontSize: 12,
                        color: "var(--text)",
                        background: "#0c0e12",
                        border: "1px solid var(--border-soft)",
                        borderRadius: 5,
                        padding: 14,
                        resize: "none",
                      }}
                    />
                  ) : (
                    <pre
                      style={{
                        flex: 1,
                        margin: 0,
                        overflow: "auto",
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
                      {editorText}
                    </pre>
                  )}
                </>
              ) : (
                <div style={{ color: "var(--text-faint)", fontSize: 12 }}>Select a file to preview or edit it.</div>
              )}
            </div>

            {/* Terminal output — streamed from `start_run` via run-output/run-exited events. */}
            <div style={{ flex: "1 1 45%", minHeight: 0, display: "flex", flexDirection: "column", padding: 14 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <div style={{ ...groupLabel(), marginBottom: 0 }}>
                  Terminal output {runActive && <span style={{ color: "var(--green)" }}>· running</span>}
                </div>
                <button onClick={() => setTerminal([])} style={btnStyle("#232323", true)}>
                  Clear
                </button>
              </div>
              <div
                style={{
                  flex: 1,
                  overflow: "auto",
                  fontFamily: "var(--mono)",
                  fontSize: 11.5,
                  color: "var(--text-dim)",
                  background: "#090c11",
                  border: "1px solid var(--border-soft)",
                  borderRadius: 5,
                  padding: 10,
                  whiteSpace: "pre-wrap",
                }}
              >
                {terminal.length === 0 ? <span style={{ color: "var(--text-faint)" }}>No output yet.</span> : terminal.join("\n")}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function DirTree({
  node,
  prefix,
  expanded,
  onToggle,
  selectedPath,
  onSelectFile,
  forceOpen,
}: {
  node: TreeDir;
  prefix: string;
  expanded: Set<string>;
  onToggle: (path: string) => void;
  selectedPath: string | null;
  onSelectFile: (f: WorkspaceFile) => void;
  forceOpen: boolean;
}) {
  const dirNames = [...node.dirs.keys()].sort();
  return (
    <div>
      {dirNames.map((name) => {
        const dirPath = prefix ? `${prefix}/${name}` : name;
        const open = forceOpen || expanded.has(dirPath);
        return (
          <div key={dirPath}>
            <div
              onClick={() => onToggle(dirPath)}
              style={{ display: "flex", alignItems: "center", gap: 4, padding: "3px 4px", cursor: "pointer", fontSize: 11.5, color: "var(--text-dim)" }}
            >
              <span style={{ width: 10, display: "inline-block", color: "var(--text-faint)" }}>{open ? "▾" : "▸"}</span>
              <span>{name}</span>
            </div>
            {open && (
              <div style={{ marginLeft: 14 }}>
                <DirTree
                  node={node.dirs.get(name)!}
                  prefix={dirPath}
                  expanded={expanded}
                  onToggle={onToggle}
                  selectedPath={selectedPath}
                  onSelectFile={onSelectFile}
                  forceOpen={forceOpen}
                />
              </div>
            )}
          </div>
        );
      })}
      {node.files
        .slice()
        .sort((a, b) => a.path.localeCompare(b.path))
        .map((f) => (
          <div
            key={f.path}
            onClick={() => onSelectFile(f)}
            style={{
              display: "flex",
              gap: 6,
              padding: "3px 4px",
              borderRadius: 4,
              cursor: "pointer",
              background: selectedPath === f.path ? "var(--panel-raised)" : "transparent",
              fontSize: 11.5,
              fontFamily: "var(--mono)",
            }}
          >
            <span style={{ color: KIND_COLOR[f.kind], width: 12 }}>{KIND_MARKER[f.kind]}</span>
            <span style={{ color: "var(--text-dim)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {f.path.split("/").pop()}
            </span>
          </div>
        ))}
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
    border: bg === "#232323" || bg === "#3b272b" ? "1px solid var(--border)" : "none",
    borderRadius: 5,
    color: bg === "var(--blue)" ? "#fff" : bg === "#3b272b" ? "#f0d7d9" : "var(--text)",
    padding: small ? "4px 10px" : "8px 16px",
    fontWeight: 600,
    fontSize: small ? 11 : 12.5,
    whiteSpace: "nowrap",
  };
}
