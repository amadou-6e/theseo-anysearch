import { useEffect, useState } from "react";
import {
  readTextFile,
  writeTextFile,
  validateConfiguration,
  onRunOutput,
  onRunExited,
  type WorkspaceIndex,
  type WorkspaceFile,
  type FileKind,
  type Diagnostic,
  type TrajectoryFile,
} from "../lib/tauri";

// Runs tab: the workspace file tree (with config classification) and the
// YAML editor with Validate/Save/Start run/Stop plus streamed terminal
// output -- matches workspace.rs's WorkspaceUi (feat/197) right pane. Run
// history is a separate, persistent sidebar (see App.tsx/RunHistorySidebar)
// shared across Runs/Replay/Explain, per docs/ui/workspace.md.

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

/** Config filename -> number of runs whose manifest names it as source_yaml. */
function countRunsByConfigBasename(runs: { source_yaml: string | null }[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const run of runs) {
    if (!run.source_yaml) continue;
    const base = run.source_yaml.split(/[/\\]/).pop();
    if (!base) continue;
    counts.set(base, (counts.get(base) ?? 0) + 1);
  }
  return counts;
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

export default function RunsPanel({
  root,
  index,
  scanning,
  error,
  onRescan,
  onChangeWorkspace,
  onOpenTrajectory,
  runActive,
  onStartRun,
  onStopRun,
}: {
  root: string;
  index: WorkspaceIndex | null;
  scanning: boolean;
  error: string | null;
  onRescan: (root: string) => void;
  onChangeWorkspace: () => void;
  onOpenTrajectory: (file: TrajectoryFile) => void;
  runActive: boolean;
  onStartRun: (configPath: string) => Promise<void>;
  onStopRun: () => Promise<void>;
}) {
  const [search, setSearch] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const [selectedFile, setSelectedFile] = useState<WorkspaceFile | null>(null);
  const [editorText, setEditorText] = useState("");
  const [savedText, setSavedText] = useState("");
  const [openableTrajectory, setOpenableTrajectory] = useState<TrajectoryFile | null>(null);
  const [diagnostics, setDiagnostics] = useState<Diagnostic[]>([]);
  const [validating, setValidating] = useState(false);
  const [saving, setSaving] = useState(false);

  const [terminal, setTerminal] = useState<string[]>([]);

  useEffect(() => {
    let unlistenOutput: (() => void) | undefined;
    let unlistenExited: (() => void) | undefined;
    onRunOutput((line) => setTerminal((t) => [...t.slice(-1999), line])).then((fn) => (unlistenOutput = fn));
    onRunExited((status) => {
      setTerminal((t) => [...t.slice(-1999), `[process exited: ${status}]`]);
      if (root) onRescan(root);
    }).then((fn) => (unlistenExited = fn));
    return () => {
      unlistenOutput?.();
      unlistenExited?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
      onRescan(root);
    } finally {
      setSaving(false);
    }
  }

  async function start() {
    if (!selectedFile) return;
    setTerminal([]);
    await onStartRun(`${root}/${selectedFile.path}`);
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

  const workspaceName = root.split(/[/\\]/).pop() || root;
  const runCounts = index ? countRunsByConfigBasename(index.runs) : new Map<string, number>();

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {error && <div style={{ color: "var(--red)", fontSize: 12, padding: "8px 14px", fontFamily: "var(--mono)" }}>{error}</div>}
      {!index && !error && (
        // Not a designed flow -- this window always opens on a workspace
        // supplied by the CLI (App.tsx's initialWorkspace() call), so this
        // only shows for the edge case where that resolves to nothing.
        // No promotional drop-zone panel, no "type a path" field: just the
        // minimum needed to recover.
        <div style={{ padding: 14, flexShrink: 0, display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 12.5, color: "var(--text-faint)" }}>{scanning ? "Opening workspace…" : "No workspace loaded."}</span>
          {!scanning && (
            <button onClick={onChangeWorkspace} style={btnStyle("var(--blue)", true)}>
              Change workspace
            </button>
          )}
        </div>
      )}

      {index ? (
        <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
          {/* Editor (for recognized/invalid configs) or read-only preview, plus
              terminal -- this pane sits immediately next to Run History, with
              the file tree to its right. Confirmed by inspecting the actual
              rendered "All Windows" spec page twice (not the earlier
              text/coordinate dump, which mispositioned this): the editor is
              the wide middle column, the file tree is the narrower rightmost
              one -- this file previously had them swapped. */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, borderRight: "1px solid var(--border-soft)" }}>
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
                          <button onClick={onStopRun} disabled={!runActive} style={btnStyle("#3b272b", true)}>
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
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <div style={{ ...groupLabel(), marginBottom: 0 }}>Terminal output</div>
                  {runActive && selectedFile && (
                    <span
                      style={{
                        fontSize: 9.5,
                        fontWeight: 700,
                        letterSpacing: "0.05em",
                        color: "var(--green)",
                        background: "#173e2b",
                        border: "1px solid #2d8b5b",
                        borderRadius: 4,
                        padding: "2px 8px",
                      }}
                    >
                      {selectedFile.path.split("/").pop()} · LIVE
                    </span>
                  )}
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

          {/* Workspace file tree, collapsible, with config classification markers. --
              header (workspace name + Change workspace/Rescan) lives in this pane,
              matching the spec's placement rather than a separate top bar. */}
          <div style={{ width: 300, overflowY: "auto", padding: 14, flexShrink: 0 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 10 }}>
              <div style={{ fontSize: 15, fontWeight: 700, color: "var(--text)" }}>{workspaceName}</div>
              <span style={{ fontSize: 10, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: "0.05em" }}>workspace</span>
            </div>
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search files…"
              style={{ ...inputStyle(), width: "100%", marginBottom: 10 }}
            />
            <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
              <button onClick={onChangeWorkspace} style={{ ...btnStyle("#232323", true), flex: 1 }}>
                Change workspace
              </button>
              <button onClick={() => onRescan(root)} disabled={scanning} style={{ ...btnStyle("#232323", true), flex: 1 }}>
                {scanning ? "Scanning…" : "↻ Rescan"}
              </button>
            </div>
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
              runCounts={runCounts}
            />

            {/* Workspace-level summary, matching the spec's "WORKSPACE INDEX" box. */}
            <div style={{ marginTop: 18, paddingTop: 14, borderTop: "1px solid var(--border-soft)" }}>
              <div style={groupLabel()}>Workspace index</div>
              <div style={{ fontSize: 11, color: "var(--text-dim)", lineHeight: 1.7 }}>
                {index.configuration_count} configurations · {index.runs.length} runs indexed
                <br />
                {index.file_count} files · {index.invalid_configuration_count} invalid configs
              </div>
            </div>
          </div>
        </div>
      ) : null}
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
  runCounts,
}: {
  node: TreeDir;
  prefix: string;
  expanded: Set<string>;
  onToggle: (path: string) => void;
  selectedPath: string | null;
  onSelectFile: (f: WorkspaceFile) => void;
  forceOpen: boolean;
  runCounts: Map<string, number>;
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
                  runCounts={runCounts}
                />
              </div>
            )}
          </div>
        );
      })}
      {node.files
        .slice()
        .sort((a, b) => a.path.localeCompare(b.path))
        .map((f) => {
          const name = f.path.split("/").pop() ?? f.path;
          const count = runCounts.get(name);
          return (
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
              <span style={{ color: "var(--text-dim)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{name}</span>
              {!!count && (
                <span style={{ color: "var(--text-faint)", marginLeft: "auto", flexShrink: 0 }}>
                  · {count} run{count === 1 ? "" : "s"}
                </span>
              )}
            </div>
          );
        })}
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
