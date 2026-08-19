//! Trajectory and Tune-trial discovery for the Replay tab.
//!
//! Ported (not reinvented) from `theseo_anysearch/core/src/bin/voxel_replay.rs`:
//! `parse_npy_uint16_2d_3cols`, `load_trajectory`'s `init_filled_file` sidecar
//! resolution, `load_trial_trajectories`, and `scan_tune_dir`/`TrialEntry` are
//! the same logic, same field names, same file-naming conventions.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

// ---------------------------------------------------------------------------
// .npy sidecar geometry (episode.init_filled_file)
// ---------------------------------------------------------------------------

/// Parse a numpy `.npy` file holding a `(N, 3)` array of `uint16` — the
/// format `init_filled_file` sidecars use for large geometry that doesn't
/// fit inline in the trajectory JSON. Verbatim port of
/// `voxel_replay.rs::parse_npy_uint16_2d_3cols`.
fn parse_npy_uint16_2d_3cols(bytes: &[u8]) -> Result<Vec<[u16; 3]>, String> {
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return Err("not a numpy file".into());
    }
    let major = bytes[6];
    let (header_len, hdr_offset) = if major == 1 {
        (u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize, 10)
    } else {
        if bytes.len() < 12 {
            return Err("truncated v2 header".into());
        }
        (u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, 12)
    };
    let hdr_end = hdr_offset + header_len;
    if bytes.len() < hdr_end {
        return Err("truncated header data".into());
    }
    let header = std::str::from_utf8(&bytes[hdr_offset..hdr_end])
        .map_err(|e| format!("header utf-8: {e}"))?;
    if !header.contains("<u2") {
        return Err("expected uint16 numpy array".into());
    }

    let shape_pos = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or("no 'shape' key in header")?;
    let after = &header[shape_pos..];
    let tuple_start = after.find('(').ok_or("no '(' in shape")? + shape_pos;
    let tuple_end = header[tuple_start..].find(')').ok_or("no ')' in shape")? + tuple_start;
    let tuple_str = &header[tuple_start + 1..tuple_end];
    let dims: Vec<usize> = tuple_str
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    if dims.len() != 2 || dims[1] != 3 {
        return Err(format!("expected shape (N, 3), got {:?}", dims));
    }

    let expected = dims[0] * dims[1] * std::mem::size_of::<u16>();
    let data_bytes = &bytes[hdr_end..];
    if data_bytes.len() < expected {
        return Err(format!("data too short: {} < {}", data_bytes.len(), expected));
    }

    let mut coords = Vec::with_capacity(dims[0]);
    for chunk in data_bytes[..expected].chunks_exact(6) {
        coords.push([
            u16::from_le_bytes([chunk[0], chunk[1]]),
            u16::from_le_bytes([chunk[2], chunk[3]]),
            u16::from_le_bytes([chunk[4], chunk[5]]),
        ]);
    }
    Ok(coords)
}

/// Read one trajectory JSON file, resolving `episode.init_filled_file` (a
/// `.npy` sidecar, relative to the trajectory file) into inline
/// `episode.init_filled` coordinates when present, matching
/// `voxel_replay.rs::load_trajectory`.
fn load_trajectory_value(path: &Path) -> Result<serde_json::Value, String> {
    let contents = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut value: serde_json::Value = serde_json::from_str(&contents).map_err(|e| e.to_string())?;

    let episode = value.get_mut("episode");
    if let Some(episode) = episode {
        let init_filled_empty = episode
            .get("init_filled")
            .and_then(|v| v.as_array())
            .map(|a| a.is_empty())
            .unwrap_or(true);
        let sidecar = episode
            .get("init_filled_file")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        if init_filled_empty {
            if let Some(sidecar) = sidecar {
                let npy_path = path.parent().unwrap_or(Path::new(".")).join(&sidecar);
                let bytes = fs::read(&npy_path)
                    .map_err(|e| format!("could not read geometry sidecar {}: {e}", npy_path.display()))?;
                let coords = parse_npy_uint16_2d_3cols(&bytes)?;
                episode["init_filled"] = serde_json::to_value(coords).map_err(|e| e.to_string())?;
            }
        }
    }
    // Not part of the on-disk schema (native app tracks it separately as
    // `TrajectoryData::source_path`) but the frontend/Explain tab need the
    // originating file path, so stamp it onto the returned JSON.
    value["source_path"] = serde_json::Value::String(path.to_string_lossy().to_string());
    Ok(value)
}

#[tauri::command]
pub fn load_trajectory(path: String) -> Result<serde_json::Value, String> {
    load_trajectory_value(Path::new(&path))
}

// ---------------------------------------------------------------------------
// Trajectory file listing (Runs tab -> Replay handoff)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct TrajectoryFile {
    name: String,
    path: String,
}

#[tauri::command]
pub fn list_trajectory_files(root: String) -> Result<Vec<TrajectoryFile>, String> {
    let root_path = Path::new(&root);
    if !root_path.is_dir() {
        return Err(format!("{root} is not a directory"));
    }
    let mut out = Vec::new();
    collect_json_files(root_path, 0, &mut out)?;
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

fn collect_json_files(dir: &Path, depth: u8, out: &mut Vec<TrajectoryFile>) -> Result<(), String> {
    if depth > 2 {
        return Ok(());
    }
    let entries = fs::read_dir(dir).map_err(|e| e.to_string())?;
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_dir() {
            collect_json_files(&path, depth + 1, out)?;
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Ok(contents) = fs::read_to_string(&path) else { continue };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&contents) else { continue };
        if value.get("episode").is_some() {
            out.push(TrajectoryFile {
                name: path.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string(),
                path: path.to_string_lossy().to_string(),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Iteration history: all iter_*.json (or best.json) for one run/trial's
// trajectories/ dir, sorted — powers the "Iterations" scrubber. Verbatim
// port of `voxel_replay.rs::load_trial_trajectories`, minus the in-process
// TrajectoryData parsing (we hand raw JSON to the frontend/wasm viewer).
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn load_iteration_history(trajectories_dir: String) -> Result<Vec<serde_json::Value>, String> {
    let dir = Path::new(&trajectories_dir);
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.starts_with("iter_") && name.ends_with(".json")
        })
        .collect();
    files.sort();

    if files.is_empty() {
        let best = dir.join("best.json");
        if best.exists() {
            files.push(best);
        }
    }

    let mut trajectories: Vec<(u64, serde_json::Value)> = Vec::new();
    for path in &files {
        if let Ok(value) = load_trajectory_value(path) {
            let iteration = value.get("iteration").and_then(|v| v.as_u64()).unwrap_or(0);
            trajectories.push((iteration, value));
        }
    }
    trajectories.sort_by_key(|(iteration, _)| *iteration);
    Ok(trajectories.into_iter().map(|(_, v)| v).collect())
}

// ---------------------------------------------------------------------------
// Tune-mode trial scanning — verbatim port of `voxel_replay.rs::scan_tune_dir`
// / `TrialEntry`.
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct TrialInfo {
    trial_id: String,
    trial_name: String,
    best_reward: f32,
    params: serde_json::Value,
    trajectory_dir: String,
    sort_key: u64,
}

#[tauri::command]
pub fn scan_tune_trials(dir: String) -> Result<Vec<TrialInfo>, String> {
    let dir = Path::new(&dir);
    let mut entries: Vec<TrialInfo> = Vec::new();

    let rd = fs::read_dir(dir).map_err(|e| e.to_string())?;
    for de in rd.filter_map(|e| e.ok()) {
        let path = de.path();
        if !path.is_dir() {
            continue;
        }
        let dir_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        let runtime_path = path.join("ray_runtime.json");
        if !runtime_path.exists() {
            continue;
        }
        let traj_dir = path.join("trajectories");
        if !traj_dir.exists() {
            continue;
        }

        let runtime_val: serde_json::Value = fs::read_to_string(&runtime_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        let trial_name = runtime_val["experiment_tag"]
            .as_str()
            .filter(|s| !s.is_empty())
            .unwrap_or(&dir_name)
            .to_string();

        let sort_key: u64 = dir_name.rsplit('_').next().and_then(|s| s.parse().ok()).unwrap_or(u64::MAX);

        let best_reward = {
            let meta_path = traj_dir.join("best_meta.json");
            let reward_from_meta = fs::read_to_string(&meta_path)
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                .and_then(|v| v["episode_reward_mean"].as_f64())
                .map(|f| f as f32);
            if let Some(r) = reward_from_meta {
                r
            } else {
                let best_path = traj_dir.join("best.json");
                fs::read_to_string(&best_path)
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                    .and_then(|v| v["episode_reward_mean"].as_f64())
                    .map(|f| f as f32)
                    .unwrap_or(f32::NEG_INFINITY)
            }
        };

        let params_path = path.join("params.json");
        let params = fs::read_to_string(&params_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(serde_json::Value::Object(Default::default()));

        entries.push(TrialInfo {
            trial_id: dir_name,
            trial_name,
            best_reward,
            params,
            trajectory_dir: traj_dir.to_string_lossy().to_string(),
            sort_key,
        });
    }

    entries.sort_by_key(|e| e.sort_key);
    Ok(entries)
}
