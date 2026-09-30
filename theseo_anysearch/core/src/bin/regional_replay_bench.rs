use std::{fs, mem::size_of, path::PathBuf, time::Instant};

use serde::Serialize;
use sha2::{Digest, Sha256};
use theseo_core::{replay::regional::RegionalReplaySource, voxel::world::StorageCoord};

const EXTENT: [u32; 3] = [60_000, 40_000, 20_000];
const CHUNK_EDGE: u32 = 32;
const RADIUS: u32 = 16;
const SAMPLES: usize = 100;

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    extent: [u32; 3],
    radius: u32,
    indexed_chunks: usize,
    first_frame_ms: f64,
    frame_preparation_p50_ms: f64,
    frame_preparation_p95_ms: f64,
    viewer_rss_bytes: Option<usize>,
    visible_voxels: usize,
    resident_chunks: usize,
    pack_reads: u64,
    cache_hits: u64,
    cache_misses: u64,
}

fn main() {
    let root = fixture();
    let source = RegionalReplaySource::open(&root, 64 * 1024 * 1024).unwrap();
    let centers = fixture_centers();
    let started = Instant::now();
    let first = source.load_agent_region(centers[0], RADIUS, &[]).unwrap();
    let first_frame_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let mut samples = Vec::with_capacity(SAMPLES);
    let mut last = first;
    for index in 0..SAMPLES {
        let started = Instant::now();
        last = source
            .load_agent_region(centers[index % centers.len()], RADIUS, &[])
            .unwrap();
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    samples.sort_by(f64::total_cmp);
    let metrics = last.cache_metrics.unwrap();
    let report = Report {
        schema_version: 1,
        extent: EXTENT,
        radius: RADIUS,
        indexed_chunks: centers.len(),
        first_frame_ms,
        frame_preparation_p50_ms: percentile(&samples, 0.50),
        frame_preparation_p95_ms: percentile(&samples, 0.95),
        viewer_rss_bytes: process_rss_bytes(),
        visible_voxels: last.occupied.len(),
        resident_chunks: metrics.resident_chunks,
        pack_reads: metrics.pack_reads,
        cache_hits: metrics.cache_hits,
        cache_misses: metrics.cache_misses,
    };
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    fs::remove_dir_all(root).unwrap();
}

fn fixture_centers() -> Vec<StorageCoord> {
    (0..16)
        .map(|index| StorageCoord {
            x: 1_000 + index * 2_000,
            y: 2_000 + index * 1_000,
            z: 3_000 + index * 500,
        })
        .collect()
}

fn fixture() -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "theseo-regional-replay-bench-{}",
        std::process::id()
    ));
    if root.exists() {
        fs::remove_dir_all(&root).unwrap();
    }
    fs::create_dir_all(&root).unwrap();
    let mut pack = Vec::new();
    let mut chunks = Vec::new();
    for center in fixture_centers() {
        let local_index = (center.x % CHUNK_EDGE) * CHUNK_EDGE * CHUNK_EDGE
            + (center.y % CHUNK_EDGE) * CHUNK_EDGE
            + center.z % CHUNK_EDGE;
        let payload = [b"AWC1".as_slice(), &[2], &local_index.to_le_bytes()].concat();
        let offset = pack.len();
        pack.extend_from_slice(&payload);
        chunks.push(serde_json::json!({
            "coordinate": {
                "x": center.x / CHUNK_EDGE,
                "y": center.y / CHUNK_EDGE,
                "z": center.z / CHUNK_EDGE
            },
            "pack_offset": offset,
            "byte_length": payload.len(),
            "occupied_voxels": 1,
            "sha256": format!("{:x}", Sha256::digest(&payload))
        }));
    }
    fs::write(root.join("world.pack"), pack).unwrap();
    fs::write(
        root.join("manifest.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "coordinate_type": "u32",
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
