//! Checkpoint-backed policy explanation bridge.
//!
//! Port of `theseo_anysearch/core/src/replay/explain.rs`'s `ExplanationBridge`
//! (feat/197 additions) to a Tauri-managed long-lived child process: same
//! `theseo_anysearch.rllib.explain.native_bridge` subprocess, same
//! line-delimited JSON-RPC protocol (`{"command": ...}` in, `{"ok": true/false,
//! ...}` out). The React side owns the observation-editing/geometry-preview UI
//! that `NativeExplainUi::show_embedded` used to own in egui; this module only
//! owns the process and the request/response plumbing.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

use serde_json::{json, Value};
use tauri::State;

fn python() -> String {
    std::env::var("ANYSEARCH_PYTHON").unwrap_or_else(|_| "python".to_owned())
}

pub struct ExplanationBridge {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl ExplanationBridge {
    fn start(run: &str, checkpoint: &str) -> Result<(Self, Value), String> {
        let mut child = Command::new(python())
            .args(["-u", "-m", "theseo_anysearch.rllib.explain.native_bridge", "--run"])
            .arg(run)
            .args(["--checkpoint", checkpoint])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("failed to start explanation service: {e}"))?;
        let stdin = child.stdin.take().ok_or("explanation service has no stdin")?;
        let stdout = child.stdout.take().ok_or("explanation service has no stdout")?;
        let mut bridge = Self { _child: child, stdin, stdout: BufReader::new(stdout) };
        let ready = bridge.read_response()?;
        Ok((bridge, ready))
    }

    fn request(&mut self, request: Value) -> Result<Value, String> {
        serde_json::to_writer(&mut self.stdin, &request).map_err(|e| format!("could not encode explanation request: {e}"))?;
        self.stdin.write_all(b"\n").map_err(|e| format!("could not send explanation request: {e}"))?;
        self.stdin.flush().map_err(|e| format!("could not flush explanation request: {e}"))?;
        self.read_response()
    }

    fn read_response(&mut self) -> Result<Value, String> {
        let mut line = String::new();
        let bytes = self.stdout.read_line(&mut line).map_err(|e| format!("could not read explanation response: {e}"))?;
        if bytes == 0 {
            return Err("explanation service exited without a response".into());
        }
        let response: Value = serde_json::from_str(&line).map_err(|e| format!("explanation service returned invalid JSON ({e}): {line}"))?;
        if !response.get("ok").and_then(Value::as_bool).unwrap_or(false) {
            let kind = response.get("error_type").and_then(Value::as_str).unwrap_or("Error");
            let message = response.get("error").and_then(Value::as_str).unwrap_or("explanation request failed without a message");
            return Err(format!("{kind}: {message}"));
        }
        Ok(response)
    }
}

#[derive(Default)]
pub struct ExplainState(pub Mutex<Option<ExplanationBridge>>);

/// Start (or replace) the explanation bridge for `run`/`checkpoint`. Returns
/// the `ready` payload (`observation`/`fields`) the React editor needs to
/// build its scalar-field and local_grid controls.
#[tauri::command]
pub fn explain_start(state: State<ExplainState>, run: String, checkpoint: String) -> Result<Value, String> {
    let (bridge, ready) = ExplanationBridge::start(&run, &checkpoint)?;
    *state.0.lock().map_err(|e| e.to_string())? = Some(bridge);
    Ok(ready)
}

#[tauri::command]
pub fn explain_available(state: State<ExplainState>) -> Result<bool, String> {
    Ok(state.0.lock().map_err(|e| e.to_string())?.is_some())
}

fn with_bridge<T>(state: &State<ExplainState>, f: impl FnOnce(&mut ExplanationBridge) -> Result<T, String>) -> Result<T, String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    let bridge = guard.as_mut().ok_or("no checkpoint-backed explanation service is configured")?;
    f(bridge)
}

#[tauri::command]
pub fn explain_trajectory_step(state: State<ExplainState>, trajectory: String, step: usize) -> Result<Value, String> {
    with_bridge(&state, |bridge| {
        bridge.request(json!({"command": "explain_trajectory", "trajectory": trajectory, "step": step}))
    })
}

#[tauri::command]
pub fn explain_observation(state: State<ExplainState>, observation: Value) -> Result<Value, String> {
    with_bridge(&state, |bridge| bridge.request(json!({"command": "explain_observation", "observation": observation})))
}

#[tauri::command]
pub fn explain_import_observation(state: State<ExplainState>, path: String) -> Result<Value, String> {
    with_bridge(&state, |bridge| bridge.request(json!({"command": "load_observation_file", "path": path})))
}
