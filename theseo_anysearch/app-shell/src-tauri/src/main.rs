// AnySearch UI shell (feat/200). Tauri backend exposing workspace, run, and
// trajectory data to the React frontend over `invoke()`.
//
// This does NOT depend on `theseo-core` yet: that crate builds as a PyO3
// extension module (crate-type cdylib, `extension-module` feature always on),
// which is not linkable as a normal binary dependency. The data types/loading
// logic in `trajectory.rs` are a deliberate short-term duplication until a
// shared `theseo-core-data` library crate is factored out for both to depend
// on (see app-shell/README.md).

mod explain_bridge;
mod trajectory;
mod workspace;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(workspace::RunProcessState::default())
        .manage(explain_bridge::ExplainState::default())
        .invoke_handler(tauri::generate_handler![
            workspace::initial_workspace,
            workspace::scan_workspace,
            workspace::validate_configuration,
            workspace::read_text_file,
            workspace::write_text_file,
            workspace::start_run,
            workspace::stop_run,
            workspace::run_is_active,
            trajectory::list_trajectory_files,
            trajectory::load_trajectory,
            trajectory::load_iteration_history,
            trajectory::scan_tune_trials,
            explain_bridge::explain_start,
            explain_bridge::explain_available,
            explain_bridge::explain_trajectory_step,
            explain_bridge::explain_observation,
            explain_bridge::explain_import_observation,
        ])
        .run(tauri::generate_context!())
        .expect("error while running AnySearch shell");
}
