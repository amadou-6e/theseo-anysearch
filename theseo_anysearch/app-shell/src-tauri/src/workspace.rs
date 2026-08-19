//! Workspace scan/validate + file editing + run lifecycle.
//!
//! `scan_workspace`/`validate_configuration` are thin passthroughs to the
//! `theseo_anysearch.ui.service` Python backend (ported from feat/197,
//! `theseo_anysearch/ui/{service,workspace}.py`) — same principle as that
//! module's own docstring: UI receives structured data instead of
//! reimplementing YAML/Pydantic validation. `start_run`/`stop_run` and
//! terminal streaming are a port of `workspace.rs`'s `WorkspaceUi` process
//! management (feat/197), using Tauri events instead of an egui poll loop.

use std::fs;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use tauri::{AppHandle, Emitter, Manager, State};

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

/// The workspace this window should open on, resolved once at startup --
/// there is no in-app "type a path" entry point (see App.tsx/RunsPanel.tsx):
/// per docs/ui/workspace.md and cli/main.py's `native_ui` command (feat/197,
/// not yet ported to this branch's CLI), the native shell is launched as
/// `<binary> --workspace <path>` with the process cwd already set to that
/// workspace. Accepts either convention: an explicit `--workspace <path>`
/// argument, or (absent that) the process's current directory, matching
/// `anysearch ui .` semantics. Returns `None` only if neither resolves to
/// an existing directory, so the "Change workspace"/drop-zone empty state
/// still applies when launched with no context (e.g. during development).
#[tauri::command]
pub fn initial_workspace() -> Option<String> {
    let mut args = std::env::args().skip(1);
    let mut from_flag: Option<String> = None;
    while let Some(arg) = args.next() {
        if arg == "--workspace" {
            from_flag = args.next();
            break;
        }
    }
    let candidate = from_flag.map(std::path::PathBuf::from).or_else(|| std::env::current_dir().ok());
    candidate.filter(|p| p.is_dir()).map(|p| p.to_string_lossy().to_string())
}

/// Rebuild the workspace index (files + run.json-backed runs) for `root`.
#[tauri::command]
pub fn scan_workspace(root: String) -> Result<serde_json::Value, String> {
    ui_backend(&root, "scan", &root)
}

/// Validate one AnySearch YAML configuration with the same loader used by
/// CLI training (`path` is absolute).
#[tauri::command]
pub fn validate_configuration(root: String, path: String) -> Result<serde_json::Value, String> {
    ui_backend(&root, "validate", &path)
}

/// Read one file's raw text contents (workspace file-tree preview / YAML editor).
#[tauri::command]
pub fn read_text_file(path: String) -> Result<String, String> {
    fs::read_to_string(&path).map_err(|e| e.to_string())
}

/// Write the YAML editor's contents back to disk (`workspace.rs::save`).
#[tauri::command]
pub fn write_text_file(path: String, contents: String) -> Result<(), String> {
    fs::write(&path, contents).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Run lifecycle: `anysearch run <config>` spawned as a child process, with
// stdout/stderr streamed to the frontend as `run-output` events and process
// exit as a `run-exited` event — the Tauri-event equivalent of workspace.rs's
// `poll_output`, which polled an mpsc channel inside the egui update loop.
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct RunProcessState(pub Mutex<Option<Child>>);

fn stream_lines(app: AppHandle, stream: impl std::io::Read + Send + 'static, stream_name: &'static str) {
    std::thread::spawn(move || {
        for line in BufReader::new(stream).lines().map_while(Result::ok) {
            let _ = app.emit("run-output", format!("[{stream_name}] {line}"));
        }
    });
}

#[tauri::command]
pub fn start_run(app: AppHandle, state: State<RunProcessState>, root: String, config_path: String) -> Result<(), String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    if guard.is_some() {
        return Err("a run is already active".into());
    }
    let mut command = Command::new(python());
    command
        .args(["-m", "theseo_anysearch.cli.main", "run"])
        .arg(&config_path)
        .current_dir(&root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().map_err(|e| format!("run process could not start: {e}"))?;
    if let Some(stdout) = child.stdout.take() {
        stream_lines(app.clone(), stdout, "stdout");
    }
    if let Some(stderr) = child.stderr.take() {
        stream_lines(app.clone(), stderr, "stderr");
    }

    let pid = child.id();
    *guard = Some(child);
    drop(guard);

    // Poll for process exit on a background thread and emit `run-exited`
    // once it finishes, since Tauri commands can't block the caller here.
    // `AppHandle` is 'static + Clone, and re-fetching managed state through
    // it (rather than capturing the `State` borrow) is the supported way to
    // reach it from a spawned thread.
    let app_for_thread = app.clone();
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_millis(400));
        let state = app_for_thread.state::<RunProcessState>();
        let mut guard = match state.0.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let Some(child) = guard.as_mut() else { return };
        if child.id() != pid {
            return;
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                let _ = app_for_thread.emit("run-exited", format!("{status}"));
                *guard = None;
                return;
            }
            Ok(None) => continue,
            Err(_) => return,
        }
    });

    Ok(())
}

#[tauri::command]
pub fn stop_run(state: State<RunProcessState>) -> Result<(), String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(mut child) = guard.take() {
        child.kill().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
pub fn run_is_active(state: State<RunProcessState>) -> Result<bool, String> {
    Ok(state.0.lock().map_err(|e| e.to_string())?.is_some())
}
