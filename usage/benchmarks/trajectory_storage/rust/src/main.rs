mod codecs;
mod viewer_probe;
use std::{env, fs, path::Path, time::Instant};
use serde_json::json;

#[cfg(windows)]
fn memory() -> serde_json::Value {
    #[repr(C)] struct Counters {
        cb: u32, faults: u32, peak: usize, rss: usize, rest: [usize; 6],
    }
    #[link(name="kernel32")] extern "system" { fn GetCurrentProcess() -> *mut std::ffi::c_void; }
    #[link(name="psapi")] extern "system" {
        fn GetProcessMemoryInfo(process: *mut std::ffi::c_void, counters: *mut Counters, size: u32) -> i32;
    }
    let mut value = Counters { cb: std::mem::size_of::<Counters>() as u32,
                              faults:0, peak:0, rss:0, rest:[0;6] };
    let success = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut value, std::mem::size_of::<Counters>() as u32) };
    if success == 0 { json!({"rss_bytes":null,"peak_rss_bytes":null}) }
    else { json!({"rss_bytes":value.rss,"peak_rss_bytes":value.peak}) }
}
#[cfg(not(windows))]
fn memory() -> serde_json::Value { json!({"rss_bytes":null,"peak_rss_bytes":null}) }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 3 {
        return Err("usage: probe <format-directory> <format> <episode-count> [original-json]".into());
    }
    let count: usize = args[2].parse()?;
    if count == 0 { return Err("episode count must be positive".into()); }
    let baseline = memory();
    let started = Instant::now();
    let value = codecs::read(Path::new(&args[0]), &args[1])?;
    let decode_ms = started.elapsed().as_secs_f64()*1000.0;
    let steps = value["episode"]["steps"].as_array().ok_or("missing steps")?.len();
    if steps == 0 { return Err("empty trace".into()); }
    if let Some(original) = args.get(3) {
        let expected: serde_json::Value = serde_json::from_slice(&fs::read(original)?)?;
        if value != expected { return Err("cross-language round-trip mismatch".into()); }
    }
    let started = Instant::now();
    let trace = viewer_probe::parse(value)?;
    let materialize_ms = started.elapsed().as_secs_f64()*1000.0;
    let mut retained = vec![trace];
    let started = Instant::now();
    for _ in 1..count { retained.push(viewer_probe::parse(codecs::read(Path::new(&args[0]), &args[1])?)?); }
    let subsequent_ms = started.elapsed().as_secs_f64()*1000.0;
    let after_load = memory();
    let mut access = Vec::new();
    for query in 0..128 {
        let index = (query * 7919 + 456) % steps;
        let started = Instant::now();
        viewer_probe::query(&retained[0], index);
        access.push(started.elapsed().as_secs_f64()*1000.0);
    }
    access.sort_by(f64::total_cmp);
    let overlay = [0, steps/4, steps/2, steps-1].into_iter().map(|index| {
        let started = Instant::now();
        let changed = viewer_probe::overlay(&retained[0], index);
        json!({"step":index,"ms":started.elapsed().as_secs_f64()*1000.0,"coordinates":changed})
    }).collect::<Vec<_>>();
    let scene = if let Some(original) = args.get(3) {
        // Synthetic fixtures have no actual world pack; measure records/overlay only.
        match viewer_probe::artifact_probe(Path::new(original)) {
            Ok(probe) => probe,
            Err(reason) => json!({"unavailable":reason}),
        }
    } else { json!(null) };
    println!("{}", json!({"format":args[1],"retained_episodes":retained.len(),
        "decode_ms":decode_ms,"viewer_materialize_ms":materialize_ms,
        "subsequent_total_ms":subsequent_ms,"baseline_memory":baseline,"after_load_memory":after_load,
        "step_access_median_ms":access[64],"step_access_p95_ms":access[121],
        "actual_viewer_overlay":overlay,"scene":scene,"cross_language_round_trip":args.len()>3,
        "build_profile":if cfg!(debug_assertions) {"debug"} else {"release"}}));
    Ok(())
}
