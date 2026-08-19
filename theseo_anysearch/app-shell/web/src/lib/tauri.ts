import { invoke } from "@tauri-apps/api/core";

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

/** List `.json` trajectory files under a workspace root (shallow, 2 levels). */
export function listTrajectoryFiles(root: string): Promise<TrajectoryFile[]> {
  return invoke("list_trajectory_files", { root });
}

/** Load one trajectory file's full JSON contents. */
export function loadTrajectory(path: string): Promise<TrajectoryData> {
  return invoke("load_trajectory", { path });
}
