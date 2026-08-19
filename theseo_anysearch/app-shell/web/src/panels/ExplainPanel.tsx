import type { TrajectoryFile } from "../lib/tauri";

// Placeholder — the Explain tab (checkpoint-backed observation inspection,
// currently native in NativeExplainUi / replay/explain.rs) is not ported yet.
// Tracked as follow-up work on feat/200.
export default function ExplainPanel({ file }: { file: TrajectoryFile }) {
  return (
    <div style={{ padding: 24, color: "var(--text-faint)", fontSize: 13 }}>
      Explain tab for <span style={{ fontFamily: "var(--mono)", color: "var(--text-dim)" }}>{file.name}</span> is not
      ported yet — still native in <span style={{ fontFamily: "var(--mono)" }}>replay/explain.rs</span>.
    </div>
  );
}
