use std::{fs, mem::size_of, path::PathBuf, time::Instant};

use serde::Serialize;
use sha2::{Digest, Sha256};
use theseo_core::replay::{
    lod::{select_chunks, CameraChunkView, ChunkBudgets},
    regional::RegionalReplaySource,
};

const EXTENT: [u32; 3] = [60_000, 40_000, 20_000];
const CHUNK_EDGE: u32 = 32;
const CHUNK_COUNT: u32 = 256;
const SAMPLES: usize = 100;

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    extent: [u32; 3],
    indexed_chunks: usize,
    visible_budget: usize,
    detailed_budget: usize,
    time_to_first_overview_ms: f64,
    first_refinement_ms: f64,
    frame_preparation_p50_ms: f64,
    frame_preparation_p95_ms: f64,
    viewer_rss_bytes: Option<usize>,
    considered_chunks: usize,
    detailed_chunks: usize,
    coarse_chunks: usize,
    resident_chunks: usize,
    pack_reads: u64,
}

fn main() {
    let root = fixture();
    let source = RegionalReplaySource::open(&root, 64 * 1024 * 1024).unwrap();
    let indexed = source.indexed_chunks();
    let budgets = ChunkBudgets {
        visible: 64,
        detailed: 16,
    };
    let overview_started = Instant::now();
    let selection = select_chunks(
        indexed.iter().map(|(chunk, _)| *chunk),
        CameraChunkView {
            center: [8.0, 8.0, 0.5],
            half_extent: [10_000.0; 3],
            forward: [0.0, 0.0, 1.0],
            minimum_forward_dot: -1.0,
        },
        budgets,
    );
    let overview_ms = overview_started.elapsed().as_secs_f64() * 1_000.0;
    let visible = selection
        .detailed
        .iter()
        .chain(&selection.coarse)
        .copied()
        .collect::<Vec<_>>();
    let refinement_started = Instant::now();
    source
        .load_chunk_selection(&selection.detailed, &visible, &visible, &[])
        .unwrap();
    let first_refinement_ms = refinement_started.elapsed().as_secs_f64() * 1_000.0;
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let started = Instant::now();
        source
            .load_chunk_selection(&selection.detailed, &visible, &visible, &[])
            .unwrap();
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    samples.sort_by(f64::total_cmp);
    let metrics = source
        .load_chunk_selection(&selection.detailed, &visible, &visible, &[])
        .unwrap()
        .cache_metrics
        .unwrap();
    let report = Report {
        schema_version: 1,
        extent: EXTENT,
        indexed_chunks: indexed.len(),
        visible_budget: budgets.visible,
        detailed_budget: budgets.detailed,
        time_to_first_overview_ms: overview_ms,
        first_refinement_ms,
        frame_preparation_p50_ms: percentile(&samples, 0.50),
        frame_preparation_p95_ms: percentile(&samples, 0.95),
        viewer_rss_bytes: process_rss_bytes(),
        considered_chunks: selection.considered,
        detailed_chunks: selection.detailed.len(),
        coarse_chunks: selection.coarse.len(),
        resident_chunks: metrics.resident_chunks,
        pack_reads: metrics.pack_reads,
    };
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    fs::remove_dir_all(root).unwrap();
}

fn fixture() -> PathBuf {
    let root = std::env::temp_dir().join(format!("theseo-lod-replay-bench-{}", std::process::id()));
    if root.exists() {
        fs::remove_dir_all(&root).unwrap();
    }
    fs::create_dir_all(&root).unwrap();
    let mut pack = Vec::new();
    let mut chunks = Vec::new();
    for index in 0..CHUNK_COUNT {
        let coordinate = [index % 16, index / 16, 0];
        let payload = [b"AWC1".as_slice(), &[2], &0u32.to_le_bytes()].concat();
        let offset = pack.len();
        pack.extend_from_slice(&payload);
        chunks.push(serde_json::json!({
            "coordinate": {"x": coordinate[0], "y": coordinate[1], "z": coordinate[2]},
            "pack_offset": offset, "byte_length": payload.len(), "occupied_voxels": 1,
            "sha256": format!("{:x}", Sha256::digest(&payload))
        }));
    }
    fs::write(root.join("world.pack"), pack).unwrap();
    fs::write(
        root.join("manifest.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1, "coordinate_type": "u32",
            "extent": {"x": EXTENT[0], "y": EXTENT[1], "z": EXTENT[2]},
            "chunk_shape": {"x": CHUNK_EDGE, "y": CHUNK_EDGE, "z": CHUNK_EDGE},
            "chunks": chunks
        }))
        .unwrap(),
    )
    .unwrap();
    root
}

fn percentile(samples: &[f64], percentile: f64) -> f64 {
    samples[((samples.len() - 1) as f64 * percentile).round() as usize]
}

#[cfg(windows)]
fn process_rss_bytes() -> Option<usize> {
    #[repr(C)]
    struct Counters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
    }
    #[link(name = "psapi")]
    extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut std::ffi::c_void,
            counters: *mut Counters,
            size: u32,
        ) -> i32;
    }
    let mut counters = Counters {
        cb: size_of::<Counters>() as u32,
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
    };
    let success = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    (success != 0).then_some(counters.working_set_size)
}

#[cfg(not(windows))]
fn process_rss_bytes() -> Option<usize> {
    None
}
