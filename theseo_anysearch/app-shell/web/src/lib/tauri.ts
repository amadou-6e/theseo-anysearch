import { invoke } from "@tauri-apps/api/core";
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

/** Native directory picker (Tauri dialog plugin). Returns null if cancelled. */
export async function pickWorkspaceFolder(): Promise<string | null> {
  const result = await open({ directory: true, multiple: false });
  return typeof result === "string" ? result : null;
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
}

/** List `.json` trajectory files under a run/workspace root (shallow, 2 levels). */
export function listTrajectoryFiles(root: string): Promise<TrajectoryFile[]> {
  return invoke("list_trajectory_files", { root });
}

/** Load one trajectory file's full JSON contents. */
export function loadTrajectory(path: string): Promise<TrajectoryData> {
  return invoke("load_trajectory", { path });
}
