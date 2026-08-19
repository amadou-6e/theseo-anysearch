import { useEffect, useMemo, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import {
  explainStart,
  explainAvailable,
  explainTrajectoryStep,
  explainObservation,
  explainImportObservation,
  type TrajectoryFile,
  type FieldSchema,
} from "../lib/tauri";

// Port of theseo_anysearch/core/src/replay/explain.rs's `NativeExplainUi`
// (feat/197) to React: same ExplanationBridge protocol (see
// src-tauri/src/explain_bridge.rs), same 3-pane layout (observation editor /
// geometry preview / policy explanation), same encode/decode_scaled_integer
// math for the local_grid voxel-kind editor.

function encodeScaledInteger(value: number, scale: number, validValues: number[]): number {
  if (!(scale > 0) || !Number.isFinite(scale)) throw new Error(`invalid scale ${scale}`);
  if (!validValues.includes(value)) throw new Error(`invalid categorical value ${value}`);
  return value / scale;
}

function decodeScaledInteger(value: number, scale: number, validValues: number[]): number {
  if (!Number.isFinite(value)) throw new Error("network value must be finite");
  const raw = value * scale;
  const rounded = Math.round(raw);
  if (Math.abs(raw - rounded) > 1e-5 || rounded < 0 || rounded > 255) {
    throw new Error(`network value ${value} is not an integer scaled by ${scale}`);
  }
  if (!validValues.includes(rounded)) throw new Error(`decoded value ${rounded} is not valid`);
  return rounded;
}

interface ExplainSeed {
  trajectorySourcePath: string;
  step: number;
}

export default function ExplainPanel({ file, seed }: { file: TrajectoryFile; seed: ExplainSeed | null }) {
  const [runDir, setRunDir] = useState(() => dirname(dirname(file.path)));
  const [checkpoint, setCheckpoint] = useState("latest");
  const [connecting, setConnecting] = useState(false);
  const [available, setAvailable] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [observation, setObservation] = useState<Record<string, unknown>>({});
  const [fields, setFields] = useState<Record<string, FieldSchema>>({});
  const [importedFrom, setImportedFrom] = useState<string | null>(null);
  const [axis, setAxis] = useState(0);
  const [sliceIndex, setSliceIndex] = useState(0);
  const [cameraYaw, setCameraYaw] = useState(45 * (Math.PI / 180));
  const [cameraPitch, setCameraPitch] = useState(30 * (Math.PI / 180));

  const [result, setResult] = useState<any>(null);
  // "Observation source" toggle, matching the spec's "● Current replay step
  // / ○ Fictional observation" radio choice -- picking "current" reloads
  // the seeded trajectory step (discarding any fictional edits); "fictional"
  // reveals the import/manual-edit controls.
  const [observationSource, setObservationSource] = useState<"current" | "fictional">(seed ? "current" : "fictional");

  useEffect(() => {
    explainAvailable().then(setAvailable);
  }, []);

  async function connect() {
    setConnecting(true);
    setError(null);
    try {
      const ready = await explainStart(runDir, checkpoint);
      setObservation(ready.observation);
      setFields(ready.fields);
      setAvailable(true);
      if (seed) await runExplainStep(seed.trajectorySourcePath, seed.step);
    } catch (e) {
      setError(String(e));
      setAvailable(false);
    } finally {
      setConnecting(false);
    }
  }

  async function runExplainStep(trajectoryPath: string, step: number) {
    try {
      const response = await explainTrajectoryStep(trajectoryPath, step);
      setResult((response as any).report ?? null);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }

  async function runExplainObservation(obs: Record<string, unknown>) {
    try {
      const response = await explainObservation(obs);
      setResult((response as any).report ?? null);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }

  async function loadFictionalObservation() {
    const path = await open({
      multiple: false,
      filters: [{ name: "Observation", extensions: ["json", "npy", "npz", "pb", "tensor"] }],
    });
    if (typeof path !== "string") return;
    try {
      const response = await explainImportObservation(path);
      setObservation(response.observation);
      setImportedFrom(`${path} (${response.format})`);
      setError(null);
      await runExplainObservation(response.observation);
    } catch (e) {
      setError(String(e));
    }
  }

  // Auto-run the seeded step once connected (e.g. from Replay's "Explain
  // current step" button); re-runs if the seed changes while connected.
  // Only while "Current replay step" is the selected observation source --
  // matches explain.rs's run_request, which computes the report for a
  // trajectory step without overwriting the editable observation fields
  // (those keep whatever was loaded at connect-time or last imported).
  useEffect(() => {
    if (available && seed && observationSource === "current") runExplainStep(seed.trajectorySourcePath, seed.step);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [available, seed?.trajectorySourcePath, seed?.step, observationSource]);

  function updateField(name: string, values: unknown[]) {
    const next = { ...observation, [name]: values };
    setObservation(next);
    setResult(null);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", padding: 14, borderBottom: "1px solid var(--border-soft)", flexShrink: 0 }}>
        <input value={runDir} onChange={(e) => setRunDir(e.target.value)} placeholder="Run directory" style={inputStyle()} />
        <input
          value={checkpoint}
          onChange={(e) => setCheckpoint(e.target.value)}
          placeholder="checkpoint (latest or iteration)"
          style={{ ...inputStyle(), flex: "0 0 220px" }}
        />
        <button onClick={connect} disabled={connecting} style={btnStyle("var(--blue)")}>
          {connecting ? "Restoring checkpoint…" : available ? "Reconnect" : "Connect"}
        </button>
        {available && <span style={{ color: "var(--green)", fontSize: 11.5 }}>● connected</span>}
      </div>

      {error && <div style={{ color: "var(--red)", fontSize: 12, padding: "8px 14px", fontFamily: "var(--mono)" }}>{error}</div>}

      {!available ? (
        <div style={{ padding: 24, color: "var(--text-faint)", fontSize: 12.5 }}>
          Connect to a checkpoint-backed run to explain policy decisions. Defaults to the run directory inferred from{" "}
          <span style={{ fontFamily: "var(--mono)" }}>{file.name}</span>.
        </div>
      ) : (
        // Layout matches spec/ui-design/replayer-current.drawio's "All Windows"
        // Explain window: a large left/main area with the geometry preview
        // stacked above the policy-explanation result, and a narrower right
        // sidebar for editing the observation (source, local_grid, actions) --
        // not three equal columns.
        <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
          <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", padding: 14, borderRight: "1px solid var(--border-soft)", overflowY: "auto" }}>
            <div style={groupLabel()}>Observation geometry</div>
            <GeometryPreview
              observation={observation}
              fields={fields}
              axis={axis}
              sliceIndex={sliceIndex}
              yaw={cameraYaw}
              pitch={cameraPitch}
              onDrag={(dx, dy) => {
                setCameraYaw((y) => y + dx * 0.008);
                setCameraPitch((p) => Math.max(-1.35, Math.min(1.35, p - dy * 0.008)));
              }}
              onGridChange={(vals) => updateField("local_grid", vals)}
            />

            <div style={{ ...groupLabel(), marginTop: 18, flexShrink: 0 }}>Policy explanation</div>
            <div style={{ flexShrink: 0 }}>
              <ResultPanel result={result} />
            </div>
          </div>

          <div style={{ width: 340, overflowY: "auto", padding: 14, flexShrink: 0 }}>
            <div style={groupLabel()}>Explain policy</div>
            <div style={{ fontSize: 11, color: "var(--text-faint)", marginBottom: 10 }}>
              Edit a recorded observation or construct a fictional one.
            </div>

            <div style={{ ...groupLabel(), marginBottom: 6, fontSize: 9.5 }}>Observation source</div>
            <RadioRow
              label="Current replay step"
              active={observationSource === "current"}
              disabled={!seed}
              onClick={() => setObservationSource("current")}
            />
            <RadioRow
              label="Fictional observation"
              active={observationSource === "fictional"}
              onClick={() => setObservationSource("fictional")}
            />

            {observationSource === "fictional" ? (
              <div style={{ marginTop: 10 }}>
                <button onClick={loadFictionalObservation} style={{ ...btnStyle("#232323", true), marginBottom: 4 }}>
                  Load fictional observation…
                </button>
                <div style={{ fontSize: 10, color: "var(--text-faint)" }}>Fictional input — format auto-detected.</div>
                {importedFrom && <div style={{ color: "var(--green)", fontSize: 10.5, marginTop: 4 }}>Loaded: {importedFrom}</div>}
              </div>
            ) : (
              <div style={{ marginTop: 10, fontSize: 10.5, color: "var(--text-faint)" }}>
                {seed ? `Step ${seed.step} of the selected trajectory.` : "Open Replay and pick a step to explain it here."}
              </div>
            )}

            <div style={{ marginTop: 14, fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", color: "var(--text-faint)", textTransform: "uppercase" }}>
              Edit geometry
            </div>
            <ScalarFields observation={observation} fields={fields} onChange={updateField} />

            <LocalGridEditor
              observation={observation}
              fields={fields}
              axis={axis}
              setAxis={setAxis}
              sliceIndex={sliceIndex}
              setSliceIndex={setSliceIndex}
              onChange={(values) => updateField("local_grid", values)}
            />

            {observationSource === "fictional" && (
              <button
                onClick={() => runExplainObservation(observation)}
                style={{ ...btnStyle("var(--blue)"), width: "100%", marginTop: 12 }}
              >
                Explain policy decision
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function dirname(path: string): string {
  const idx = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return idx >= 0 ? path.slice(0, idx) : path;
}

// ---------------------------------------------------------------------------
// Scalar fields (everything except local_grid)
// ---------------------------------------------------------------------------

function ScalarFields({
  observation,
  fields,
  onChange,
}: {
  observation: Record<string, unknown>;
  fields: Record<string, FieldSchema>;
  onChange: (name: string, values: unknown[]) => void;
}) {
  const names = Object.keys(observation).filter((n) => n !== "local_grid");
  if (names.length === 0) {
    return <div style={{ color: "#e0c341", fontSize: 11.5, marginTop: 8 }}>This policy exposes no scalar observation fields.</div>;
  }
  return (
    <div style={{ marginTop: 8 }}>
      {names.map((name) => {
        const raw = observation[name];
        if (!Array.isArray(raw)) return null;
        const bounds = fields[name];
        const scale = bounds?.input_encoding?.type === "integer_scaled" ? bounds.input_encoding.scale : undefined;
        const validValues = bounds?.input_encoding?.valid_values ?? [];
        return (
          <div key={name} style={{ marginBottom: 14 }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text)" }}>{name}</div>
            {scale ? (
              <>
                <div style={{ fontSize: 10.5, color: "var(--text-faint)" }}>Valid integers: [{validValues.join(", ")}]</div>
                {raw.map((v, i) => {
                  let decoded: number;
                  try {
                    decoded = decodeScaledInteger(Number(v), scale, validValues);
                  } catch {
                    return (
                      <div key={i} style={{ color: "var(--red)", fontSize: 11 }}>
                        [{i}] decode error
                      </div>
                    );
                  }
                  return (
                    <select
                      key={i}
                      value={decoded}
                      onChange={(e) => {
                        const next = [...raw];
                        next[i] = encodeScaledInteger(Number(e.target.value), scale, validValues);
                        onChange(name, next);
                      }}
                      style={{ ...selectStyle(), marginRight: 6, marginTop: 4 }}
                    >
                      {validValues.map((cand) => (
                        <option key={cand} value={cand}>
                          [{i}] {cand}
                        </option>
                      ))}
                    </select>
                  );
                })}
              </>
            ) : (
              raw.map((v, i) => {
                const low = bounds?.low?.[i] ?? bounds?.low?.[0] ?? -1;
                const high = bounds?.high?.[i] ?? bounds?.high?.[0] ?? 1;
                return (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 4 }}>
                    <span style={{ fontSize: 10.5, color: "var(--text-faint)", width: 24 }}>[{i}]</span>
                    <input
                      type="range"
                      min={low}
                      max={high}
                      step={(high - low) / 200 || 0.01}
                      value={Number(v)}
                      onChange={(e) => {
                        const next = [...raw];
                        next[i] = Number(e.target.value);
                        onChange(name, next);
                      }}
                      style={{ flex: 1 }}
                    />
                    <span style={{ fontSize: 10.5, fontFamily: "var(--mono)", color: "var(--text-dim)", width: 48, textAlign: "right" }}>
                      {Number(v).toFixed(3)}
                    </span>
                  </div>
                );
              })
            )}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// local_grid voxel-kind editor
// ---------------------------------------------------------------------------

function LocalGridEditor({
  observation,
  fields,
  axis,
  setAxis,
  sliceIndex,
  setSliceIndex,
  onChange,
}: {
  observation: Record<string, unknown>;
  fields: Record<string, FieldSchema>;
  axis: number;
  setAxis: (a: number) => void;
  sliceIndex: number;
  setSliceIndex: (i: number) => void;
  onChange: (values: unknown[]) => void;
}) {
  const values = observation.local_grid as unknown[] | undefined;
  const field = fields.local_grid;
  if (!values) return <div style={{ fontSize: 11.5, color: "var(--text-faint)", marginTop: 8 }}>This policy has no local_grid field.</div>;
  const scale = field?.input_encoding?.type === "integer_scaled" ? field.input_encoding.scale : undefined;
  const validValues = field?.input_encoding?.valid_values ?? [];
  const side = Math.round(Math.cbrt(values.length));
  if (!scale || side === 0 || side ** 3 !== values.length) {
    return <div style={{ color: "var(--red)", fontSize: 11.5, marginTop: 8 }}>local_grid has no usable categorical encoding.</div>;
  }
  const clampedSlice = Math.min(sliceIndex, side - 1);
  let decoded: number[];
  try {
    decoded = values.map((v) => decodeScaledInteger(Number(v), scale, validValues));
  } catch {
    return <div style={{ color: "var(--red)", fontSize: 11.5, marginTop: 8 }}>local_grid contains an invalid value.</div>;
  }

  function indexOf(row: number, col: number): number {
    if (axis === 0) return clampedSlice * side * side + row * side + col;
    if (axis === 1) return row * side * side + clampedSlice * side + col;
    return row * side * side + col * side + clampedSlice;
  }

  function setCell(row: number, col: number, kind: number) {
    const idx = indexOf(row, col);
    const next = decoded.slice();
    next[idx] = kind;
    onChange(next.map((v) => encodeScaledInteger(v, scale!, validValues)));
  }

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text)", marginBottom: 6 }}>local_grid</div>
      <div style={{ display: "flex", gap: 6, marginBottom: 6 }}>
        {["X", "Y", "Z"].map((label, i) => (
          <button
            key={label}
            onClick={() => setAxis(i)}
            style={{ ...btnStyle(axis === i ? "var(--blue)" : "#232323", true) }}
          >
            {label}
          </button>
        ))}
      </div>
      <input type="range" min={0} max={side - 1} value={clampedSlice} onChange={(e) => setSliceIndex(Number(e.target.value))} style={{ width: "100%" }} />
      <div style={{ fontSize: 10, color: "var(--text-faint)", marginBottom: 6 }}>
        slice {clampedSlice + 1} / {side} · kinds: 0 empty · 1 occupied/boundary · 2 start · 3 goal · 5 filled/trail
      </div>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${side}, minmax(0, 1fr))`, gap: 2, overflowX: "auto" }}>
        {Array.from({ length: side }).map((_, row) =>
          Array.from({ length: side }).map((_, col) => {
            const value = decoded[indexOf(row, col)];
            return (
              <select
                key={`${row}-${col}`}
                value={value}
                onChange={(e) => setCell(row, col, Number(e.target.value))}
                style={{ ...selectStyle(), fontSize: 9, padding: "2px 0", textAlign: "center" }}
              >
                {validValues.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            );
          }),
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Geometry preview — isometric slice-based voxel preview, drag to rotate.
// Same projection math as explain.rs's project_explanation_point.
// ---------------------------------------------------------------------------

/** Same isometric projection the canvas draw uses -- factored out so the
 * clickable cell overlay can compute identical screen positions. */
function projectPoint(x: number, y: number, z: number, yaw: number, pitch: number, center: { x: number; y: number }, cubeSize: number) {
  const xr = x * Math.cos(yaw) - z * Math.sin(yaw);
  const zr = x * Math.sin(yaw) + z * Math.cos(yaw);
  const yr = y * Math.cos(pitch) - zr * Math.sin(pitch);
  return { x: center.x + xr * cubeSize, y: center.y - yr * cubeSize };
}

function GeometryPreview({
  observation,
  fields,
  axis,
  sliceIndex,
  yaw,
  pitch,
  onDrag,
  onGridChange,
}: {
  observation: Record<string, unknown>;
  fields: Record<string, FieldSchema>;
  axis: number;
  sliceIndex: number;
  yaw: number;
  pitch: number;
  onDrag: (dx: number, dy: number) => void;
  onGridChange: (values: unknown[]) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragging = useRef(false);
  const dragged = useRef(false);
  const last = useRef({ x: 0, y: 0 });
  const [renderSize, setRenderSize] = useState(320);

  const values = observation.local_grid as unknown[] | undefined;
  const field = fields.local_grid;
  const scale = field?.input_encoding?.type === "integer_scaled" ? field.input_encoding.scale : undefined;
  const validValues = field?.input_encoding?.valid_values ?? [];
  const side = values ? Math.round(Math.cbrt(values.length)) : 0;

  const kinds = useMemo(() => {
    if (!values || !scale || side === 0 || side ** 3 !== values.length) return null;
    try {
      return values.map((v) => decodeScaledInteger(Number(v), scale, validValues));
    } catch {
      return null;
    }
  }, [values, scale, validValues, side]);

  // Editable cells for the *active slice only*, projected to the same
  // screen coordinates the canvas draws at -- this is what the spec shows
  // (the numeric grid overlaid directly on the geometry view), not a
  // separate disconnected control.
  const overlayCells = useMemo(() => {
    if (!kinds || side === 0) return [];
    const center = { x: renderSize / 2, y: renderSize / 2 };
    const cubeSize = Math.min(renderSize / (side * 1.9), 72);
    const centerIndex = Math.floor(side / 2);
    const cells: { index: number; value: number; screenX: number; screenY: number }[] = [];
    for (let row = 0; row < side; row++) {
      for (let col = 0; col < side; col++) {
        let x: number, y: number, z: number;
        if (axis === 0) {
          x = sliceIndex;
          y = row;
          z = col;
        } else if (axis === 1) {
          x = row;
          y = sliceIndex;
          z = col;
        } else {
          x = row;
          y = col;
          z = sliceIndex;
        }
        const index = x * side * side + y * side + z;
        const p = projectPoint(x - centerIndex, y - centerIndex, z - centerIndex, yaw, pitch, center, cubeSize);
        cells.push({ index, value: kinds[index], screenX: p.x, screenY: p.y });
      }
    }
    return cells;
  }, [kinds, side, axis, sliceIndex, yaw, pitch, renderSize]);

  function cycleCell(index: number) {
    if (!kinds || !scale || validValues.length === 0) return;
    const current = kinds[index];
    const at = validValues.indexOf(current);
    const next = validValues[(at + 1) % validValues.length];
    const nextKinds = kinds.slice();
    nextKinds[index] = next;
    onGridChange(nextKinds.map((k) => encodeScaledInteger(k, scale, validValues)));
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !kinds) return;
    const size = Math.min(canvas.clientWidth, canvas.clientHeight) || 320;
    setRenderSize(size);
    canvas.width = size * devicePixelRatio;
    canvas.height = size * devicePixelRatio;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    ctx.fillStyle = "#0a0d12";
    ctx.fillRect(0, 0, size, size);

    const center = { x: size / 2, y: size / 2 };
    const cubeSize = Math.min(size / (side * 1.9), 72);
    const centerIndex = Math.floor(side / 2);
    const project = (x: number, y: number, z: number) => projectPoint(x, y, z, yaw, pitch, center, cubeSize);

    type V = { x: number; y: number; z: number; kind: number };
    const voxels: V[] = [];
    for (let x = 0; x < side; x++)
      for (let y = 0; y < side; y++)
        for (let z = 0; z < side; z++) {
          const kind = kinds[x * side * side + y * side + z];
          if (kind > 0 || (x === centerIndex && y === centerIndex && z === centerIndex)) {
            voxels.push({ x, y, z, kind });
          }
        }

    const view = {
      x: Math.sin(yaw) * Math.cos(pitch),
      y: Math.sin(pitch),
      z: Math.cos(yaw) * Math.cos(pitch),
    };
    voxels.sort((a, b) => a.x * view.x + a.y * view.y + a.z * view.z - (b.x * view.x + b.y * view.y + b.z * view.z));

    const occupied = new Set(voxels.map((v) => `${v.x},${v.y},${v.z}`));
    const selectedAxisValue = (v: V) => (axis === 0 ? v.x : axis === 1 ? v.y : v.z);

    for (const v of voxels) {
      const isAgent = v.x === centerIndex && v.y === centerIndex && v.z === centerIndex;
      const base = isAgent || v.kind === 2 || v.kind === 5 ? [70, 140, 210] : v.kind === 3 ? [70, 190, 110] : [120, 120, 130];
      const selected = selectedAxisValue(v) === sliceIndex;
      const alpha = selected ? 1 : 0.22;
      const cx = v.x - centerIndex;
      const cy = v.y - centerIndex;
      const cz = v.z - centerIndex;
      const h = 0.5;

      const face = (faceAxis: number, sign: number, shade: number) => {
        const nx = faceAxis === 0 ? v.x + sign : v.x;
        const ny = faceAxis === 1 ? v.y + sign : v.y;
        const nz = faceAxis === 2 ? v.z + sign : v.z;
        if (occupied.has(`${nx},${ny},${nz}`)) {
          const neighborSelected = !selected || selectedAxisValue({ x: nx, y: ny, z: nz, kind: 0 }) === sliceIndex;
          if (neighborSelected) return;
        }
        let points: [number, number, number][];
        if (faceAxis === 0) {
          points = [
            [cx + sign * h, cy - h, cz - h],
            [cx + sign * h, cy - h, cz + h],
            [cx + sign * h, cy + h, cz + h],
            [cx + sign * h, cy + h, cz - h],
          ];
        } else if (faceAxis === 1) {
          points = [
            [cx - h, cy + sign * h, cz - h],
            [cx + h, cy + sign * h, cz - h],
            [cx + h, cy + sign * h, cz + h],
            [cx - h, cy + sign * h, cz + h],
          ];
        } else {
          points = [
            [cx - h, cy - h, cz + sign * h],
            [cx + h, cy - h, cz + sign * h],
            [cx + h, cy + h, cz + sign * h],
            [cx - h, cy + h, cz + sign * h],
          ];
        }
        const screen = points.map(([px, py, pz]) => project(px, py, pz));
        const [r, g, b] = base.map((c) => Math.max(0, Math.min(255, c + shade)));
        ctx.beginPath();
        ctx.moveTo(screen[0].x, screen[0].y);
        for (const p of screen.slice(1)) ctx.lineTo(p.x, p.y);
        ctx.closePath();
        ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
        ctx.fill();
        ctx.strokeStyle = `rgba(25,28,35,${alpha})`;
        ctx.lineWidth = 0.7;
        ctx.stroke();
      };
      face(0, view.x >= 0 ? 1 : -1, -10);
      face(1, view.y >= 0 ? 1 : -1, 35);
      face(2, view.z >= 0 ? 1 : -1, -40);
    }

    ctx.fillStyle = "#646464";
    ctx.font = "11px var(--mono)";
    ctx.textAlign = "right";
    ctx.fillText("Drag to rotate", size - 8, size - 8);
  }, [kinds, side, axis, sliceIndex, yaw, pitch]);

  if (!values) return <div style={{ fontSize: 11.5, color: "var(--text-faint)" }}>This policy has no local_grid field.</div>;
  if (!kinds) return <div style={{ fontSize: 11.5, color: "var(--red)" }}>local_grid is not cubic or has an invalid encoding.</div>;

  return (
    <div
      style={{ flex: 1, minHeight: 240, position: "relative" }}
      onPointerDown={(e) => {
        dragging.current = true;
        dragged.current = false;
        last.current = { x: e.clientX, y: e.clientY };
      }}
      onPointerMove={(e) => {
        if (!dragging.current) return;
        const dx = e.clientX - last.current.x;
        const dy = e.clientY - last.current.y;
        if (Math.abs(dx) + Math.abs(dy) > 2) dragged.current = true;
        onDrag(dx, dy);
        last.current = { x: e.clientX, y: e.clientY };
      }}
      onPointerUp={() => (dragging.current = false)}
      onPointerLeave={() => (dragging.current = false)}
    >
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%", cursor: "grab", borderRadius: 8, display: "block" }} />
      {/* Editable numeric grid, overlaid on the geometry view -- click a
          cell to cycle its voxel kind. Matches the spec's Explain window,
          where the active-slice values are edited directly on the 3D
          preview rather than only in a separate sidebar control. */}
      {overlayCells.map((cell) => (
        <div
          key={cell.index}
          onClick={(e) => {
            e.stopPropagation();
            if (!dragged.current) cycleCell(cell.index);
          }}
          title="Click to cycle this cell's voxel kind"
          style={{
            position: "absolute",
            left: cell.screenX,
            top: cell.screenY,
            transform: "translate(-50%, -50%)",
            width: 26,
            height: 26,
            borderRadius: 5,
            background: "rgba(15, 18, 24, 0.85)",
            border: "1px solid rgba(59, 147, 255, 0.55)",
            color: "#dfe6ee",
            fontSize: 11,
            fontFamily: "var(--mono)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            userSelect: "none",
            pointerEvents: "auto",
          }}
        >
          {cell.value}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Result panel
// ---------------------------------------------------------------------------

function ResultPanel({ result }: { result: any }) {
  const step = result?.steps?.[0];
  if (!step) {
    return <div style={{ fontSize: 12, color: "var(--text-faint)" }}>Explain a replay step or edited observation to see policy scores.</div>;
  }
  const scores: number[] = step.action_scores ?? [];
  const finite = scores.filter((s: number) => Number.isFinite(s));
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const span = max - min || 1;

  return (
    <div>
      <div style={{ fontSize: 16, fontWeight: 700, color: "var(--text)" }}>
        Action {step.chosen_action} {step.chosen_direction ? String(step.chosen_direction) : ""}
      </div>
      <div style={{ fontSize: 11.5, color: "var(--text-dim)", marginTop: 4 }}>
        Margin over best safe action: {Number(step.score_margin).toFixed(6)}
      </div>
      <div style={{ ...groupLabel(), marginTop: 14 }}>Action scores</div>
      {scores.map((s, i) => {
        const normalized = Math.max(0, Math.min(1, (s - min) / span));
        return (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <span style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--text-faint)", width: 20 }}>{i}</span>
            <div style={{ flex: 1, height: 12, background: "#1a1a1a", borderRadius: 3, overflow: "hidden" }}>
              <div style={{ width: `${normalized * 100}%`, height: "100%", background: "var(--blue)" }} />
            </div>
            <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", width: 60, textAlign: "right" }}>{s.toFixed(5)}</span>
          </div>
        );
      })}
      {step.group_attributions && (
        <>
          <div style={{ ...groupLabel(), marginTop: 14 }}>Grouped attribution</div>
          {Object.entries(step.group_attributions as Record<string, number>).map(([name, value]) => (
            <div key={name} style={{ fontSize: 11.5, color: "var(--text-dim)" }}>
              {name}: {value.toFixed(6)}
            </div>
          ))}
        </>
      )}
    </div>
  );
}

function RadioRow({ label, active, disabled, onClick }: { label: string; active: boolean; disabled?: boolean; onClick: () => void }) {
  return (
    <div
      onClick={disabled ? undefined : onClick}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "6px 8px",
        borderRadius: 5,
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.4 : 1,
        background: active ? "var(--panel-raised)" : "transparent",
        fontSize: 12,
        color: active ? "var(--text)" : "var(--text-dim)",
      }}
    >
      <span
        style={{
          width: 12,
          height: 12,
          borderRadius: "50%",
          border: `1.5px solid ${active ? "var(--blue)" : "var(--border)"}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}
      >
        {active && <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--blue)" }} />}
      </span>
      {label}
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

function selectStyle(): React.CSSProperties {
  return {
    background: "#1f1f1f",
    border: "1px solid var(--border)",
    borderRadius: 4,
    color: "var(--text)",
    fontSize: 11,
    padding: "3px 4px",
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
    cursor: "pointer",
  };
}
