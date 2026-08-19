import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";

// ---------------------------------------------------------------------------
// Workspace scan/validate — mirrors theseo_anysearch/ui/workspace.py's
// Pydantic models (WorkspaceIndex/WorkspaceFile/WorkspaceRun) exactly, since
// the Tauri `scan_workspace`/`validate_configuration` commands are thin
// passthroughs to that Python backend, not a reimplementation.
// ---------------------------------------------------------------------------

export interface Diagnostic {
  path: string;
  message: string;
}

export type FileKind = "file" | "yaml" | "anysearch" | "invalid_anysearch";

export interface WorkspaceFile {
  path: string;
  kind: FileKind;
  diagnostics: Diagnostic[];
}

export interface WorkspaceRun {
  run_id: string;
  path: string;
  status: string;
  source_yaml: string | null;
  algorithm: string | null;
  manifest: Record<string, unknown>;
}

export interface WorkspaceIndex {
  schema_version: number;
  workspace: string;
  files: WorkspaceFile[];
  runs: WorkspaceRun[];
  file_count: number;
  yaml_count: number;
  configuration_count: number;
  invalid_configuration_count: number;
}

/** The workspace this window should open on, resolved from `--workspace
 * <path>` or the process cwd (see src-tauri/src/workspace.rs) -- there is
 * no in-app "type a path" entry point; the app opens on a workspace the
 * same way the native shell does, via `anysearch ui <path>`. */
export function initialWorkspace(): Promise<string | null> {
  return invoke("initial_workspace");
}

export function scanWorkspace(root: string): Promise<WorkspaceIndex> {
  return invoke("scan_workspace", { root });
}

export function validateConfiguration(
  root: string,
  path: string,
): Promise<{ valid: boolean; diagnostics: Diagnostic[] }> {
  return invoke("validate_configuration", { root, path });
}

export function readTextFile(path: string): Promise<string> {
  return invoke("read_text_file", { path });
}

export function writeTextFile(path: string, contents: string): Promise<void> {
  return invoke("write_text_file", { path, contents });
}

/** Native directory picker (Tauri dialog plugin). Returns null if cancelled. */
export async function pickWorkspaceFolder(): Promise<string | null> {
  const result = await open({ directory: true, multiple: false });
  return typeof result === "string" ? result : null;
}

// ---------------------------------------------------------------------------
// Run lifecycle — `anysearch run <config>`, streamed to `run-output` /
// `run-exited` events (see src-tauri/src/workspace.rs).
// ---------------------------------------------------------------------------

export function startRun(root: string, configPath: string): Promise<void> {
  return invoke("start_run", { root, configPath });
}

export function stopRun(): Promise<void> {
  return invoke("stop_run");
}

export function runIsActive(): Promise<boolean> {
  return invoke("run_is_active");
}

export function onRunOutput(handler: (line: string) => void): Promise<UnlistenFn> {
  return listen<string>("run-output", (event) => handler(event.payload));
}

export function onRunExited(handler: (status: string) => void): Promise<UnlistenFn> {
  return listen<string>("run-exited", (event) => handler(event.payload));
}

// ---------------------------------------------------------------------------
// Trajectory loading (Replay tab)
// ---------------------------------------------------------------------------

export interface TrajectoryFile {
  name: string;
  path: string;
}

export interface StepData {
  cursor_x: number;
  cursor_y: number;
  cursor_z: number;
  reward: number;
  action: number;
  voxel_count: number;
  placed: boolean;
}

export interface EpisodeData {
  init_filled: [number, number, number][];
  steps: StepData[];
  total_reward: number;
  steps_taken: number;
  success: boolean;
}

export interface TrajectoryData {
  iteration: number;
  grid_size: number;
  episode: EpisodeData;
  source_path: string;
}

/** List `.json` trajectory files under a run/workspace root (shallow, 2 levels). */
export function listTrajectoryFiles(root: string): Promise<TrajectoryFile[]> {
  return invoke("list_trajectory_files", { root });
}

/** Load one trajectory file's full JSON contents (geometry sidecar resolved). */
export function loadTrajectory(path: string): Promise<TrajectoryData> {
  return invoke("load_trajectory", { path });
}

/** Load every iter_*.json (or best.json) in a run's trajectories/ dir, sorted by iteration. */
export function loadIterationHistory(trajectoriesDir: string): Promise<TrajectoryData[]> {
  return invoke("load_iteration_history", { trajectoriesDir });
}

// ---------------------------------------------------------------------------
// Tune-mode trial scanning (Tune trial navigation)
// ---------------------------------------------------------------------------

export interface TrialInfo {
  trial_id: string;
  trial_name: string;
  best_reward: number;
  params: Record<string, unknown>;
  trajectory_dir: string;
  sort_key: number;
}

export function scanTuneTrials(dir: string): Promise<TrialInfo[]> {
  return invoke("scan_tune_trials", { dir });
}

// ---------------------------------------------------------------------------
// Explain tab — checkpoint-backed policy explanation bridge.
// ---------------------------------------------------------------------------

export interface FieldSchema {
  low: number[];
  high: number[];
  input_encoding?: { type: string; scale?: number; valid_values?: number[] };
}

export interface ExplainReady {
  observation: Record<string, unknown>;
  fields: Record<string, FieldSchema>;
}

export function explainStart(run: string, checkpoint: string): Promise<ExplainReady> {
  return invoke("explain_start", { run, checkpoint });
}

export function explainAvailable(): Promise<boolean> {
  return invoke("explain_available");
}

export function explainTrajectoryStep(trajectory: string, step: number): Promise<{ report: unknown }> {
  return invoke("explain_trajectory_step", { trajectory, step });
}

export function explainObservation(observation: Record<string, unknown>): Promise<{ report: unknown }> {
  return invoke("explain_observation", { observation });
}

export function explainImportObservation(path: string): Promise<{ observation: Record<string, unknown>; format: string }> {
  return invoke("explain_import_observation", { path });
}
