import { useState } from "react";
import RunsPanel from "./panels/RunsPanel";
import ReplayPanel from "./panels/ReplayPanel";
import ExplainPanel from "./panels/ExplainPanel";
import type { TrajectoryFile } from "./lib/tauri";

type Tab = "runs" | "replay" | "explain";

const TABS: { id: Tab; label: string }[] = [
  { id: "runs", label: "Runs" },
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
        <nav style={{ display: "flex", gap: 4, marginLeft: 24 }}>
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

      <main style={{ flex: 1, minHeight: 0 }}>
        {tab === "runs" && (
          <RunsPanel
            onOpenTrajectory={(file) => {
              setSelected(file);
              setExplainSeed(null);
              setTab("replay");
            }}
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
  );
}
