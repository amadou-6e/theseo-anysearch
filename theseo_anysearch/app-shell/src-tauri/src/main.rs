// AnySearch UI shell (feat/200). Tauri backend exposing trajectory data to the
// React frontend over `invoke()`. Scope note: this is the first real vertical
// slice of the shell — file listing + single-trajectory loading — proving the
// Tauri <-> React <-> viewer-web (wasm) data path end to end. It intentionally
// does NOT yet reimplement the full Runs-tab workspace index (run states,
// MLflow linkage, config editor, terminal streaming) from the draw.io design;
// those are tracked as follow-up work on feat/200.

use std::fs;
use std::path::Path;

use serde::Serialize;

#[derive(Serialize)]
struct TrajectoryFile {
    name: String,
    path: String,
}

/// List `.json` files under `root` that look like trajectory files (they
/// deserialize with an `episode` key). Shallow: one level of subdirectories.
#[tauri::command]
fn list_trajectory_files(root: String) -> Result<Vec<TrajectoryFile>, String> {
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

/// Read one trajectory JSON file and return it verbatim (already in the
/// shape both the React sidebar and the viewer-web wasm module expect).
#[tauri::command]
fn load_trajectory(path: String) -> Result<serde_json::Value, String> {
    let contents = fs::read_to_string(&path).map_err(|e| e.to_string())?;
    serde_json::from_str(&contents).map_err(|e| e.to_string())
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![list_trajectory_files, load_trajectory])
        .run(tauri::generate_context!())
        .expect("error while running AnySearch shell");
}
