// AnySearch UI shell (feat/200). Tauri backend exposing workspace and
// trajectory data to the React frontend over `invoke()`.
//
// `scan_workspace`/`validate_configuration` are thin wrappers around the same
// `theseo_anysearch.ui.service` Python backend that feat/197's native egui
// shell uses (see theseo_anysearch/core/src/replay/workspace.rs on that
// branch) — ported here rather than reimplemented, per that module's own
// stated principle: "The UI intentionally receives structured data from this
// module instead of reimplementing YAML or Pydantic validation in Rust."

use std::fs;
use std::path::Path;
use std::process::Command;

use serde::Serialize;

#[derive(Serialize)]
struct TrajectoryFile {
    name: String,
    path: String,
}

fn python() -> String {
    std::env::var("ANYSEARCH_PYTHON").unwrap_or_else(|_| "python".to_owned())
}

/// Run `python -m theseo_anysearch.ui.service <operation> <path>` with cwd
/// `root`, matching workspace.rs's `Self::backend`.
fn ui_backend(root: &str, operation: &str, path: &str) -> Result<serde_json::Value, String> {
    let output = Command::new(python())
        .args(["-m", "theseo_anysearch.ui.service", operation, path])
        .current_dir(root)
        .output()
        .map_err(|e| format!("UI backend could not start: {e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    serde_json::from_slice(&output.stdout).map_err(|e| e.to_string())
}

/// Rebuild the workspace index (files + run.json-backed runs) for `root`.
#[tauri::command]
fn scan_workspace(root: String) -> Result<serde_json::Value, String> {
    ui_backend(&root, "scan", &root)
}

/// Validate one AnySearch YAML configuration with the same loader used by
/// CLI training (`path` is absolute).
#[tauri::command]
fn validate_configuration(root: String, path: String) -> Result<serde_json::Value, String> {
    ui_backend(&root, "validate", &path)
}

/// Read one file's raw text contents (workspace file-tree preview / YAML editor).
#[tauri::command]
fn read_text_file(path: String) -> Result<String, String> {
    fs::read_to_string(&path).map_err(|e| e.to_string())
}

/// List `.json` files under `root` that look like trajectory files (they
/// deserialize with an `episode` key). Shallow: one level of subdirectories.
/// Used to populate Replay once a run is selected in the Runs tab.
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
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            scan_workspace,
            validate_configuration,
            read_text_file,
            list_trajectory_files,
            load_trajectory,
        ])
        .run(tauri::generate_context!())
        .expect("error while running AnySearch shell");
}
