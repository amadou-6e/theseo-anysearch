import { useState } from "react";
import { listTrajectoryFiles, type TrajectoryFile } from "../lib/tauri";

// Minimal first slice of the "Runs" screen from the draw.io design: it scans
// a workspace root for trajectory JSON files and lets you open one in Replay.
// The full run-history feature (run states, resume/stop, MLflow linkage,
// config editor, terminal streaming) is tracked as follow-up work on
// feat/200 — this panel exists to prove the Tauri <-> React data path works.
export default function RunsPanel({ onOpenTrajectory }: { onOpenTrajectory: (file: TrajectoryFile) => void }) {
  const [root, setRoot] = useState("");
  const [files, setFiles] = useState<TrajectoryFile[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function scan() {
    setLoading(true);
    setError(null);
    try {
      setFiles(await listTrajectoryFiles(root));
    } catch (e) {
      setError(String(e));
      setFiles([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: 24, maxWidth: 640 }}>
      <h2 style={{ fontSize: 15, fontWeight: 700, margin: "0 0 12px" }}>Run history</h2>
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input
          value={root}
          onChange={(e) => setRoot(e.target.value)}
          placeholder="Workspace root (e.g. usage/runs)"
          style={{
            flex: 1,
            background: "var(--panel)",
            border: "1px solid var(--border)",
            borderRadius: 5,
            color: "var(--text)",
            padding: "8px 10px",
            fontSize: 12.5,
          }}
        />
        <button
          onClick={scan}
          disabled={!root || loading}
          style={{
            background: "var(--blue)",
            border: "none",
            borderRadius: 5,
            color: "#fff",
            padding: "8px 16px",
            fontWeight: 600,
            fontSize: 12.5,
          }}
        >
          {loading ? "Scanning…" : "Scan"}
        </button>
      </div>

      {error && (
        <div style={{ color: "var(--red)", fontSize: 12, marginBottom: 12, fontFamily: "var(--mono)" }}>{error}</div>
      )}

      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {files.map((f) => (
          <div
            key={f.path}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              background: "var(--panel)",
              border: "1px solid var(--border-soft)",
              borderRadius: 5,
              padding: "8px 12px",
            }}
          >
            <span style={{ fontFamily: "var(--mono)", fontSize: 12 }}>{f.name}</span>
            <button
              onClick={() => onOpenTrajectory(f)}
              style={{
                background: "#232323",
                border: "1px solid var(--border)",
                borderRadius: 4,
                color: "var(--text)",
                padding: "4px 10px",
                fontSize: 11.5,
              }}
            >
              Open
            </button>
          </div>
        ))}
        {!loading && files.length === 0 && !error && (
          <div style={{ color: "var(--text-faint)", fontSize: 12 }}>No trajectories scanned yet.</div>
        )}
      </div>
    </div>
  );
}
