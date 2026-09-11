use std::{collections::HashSet, time::Instant};

use serde::Serialize;
use theseo_core::{
    replay::render_cache::{
        chunk_occupancy_revision, ChunkCoord, ChunkRenderCache, RenderCacheKey,
    },
    voxel::world::StorageCoord,
};

const CHUNK_EDGE: u32 = 32;
const SAMPLES: usize = 200;

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    chunk_edge: u32,
    occupied_voxels: usize,
    exposed_faces: usize,
    emitted_triangles: usize,
    naive_faces: usize,
    face_reduction_fraction: f64,
    cold_build_ms: f64,
    frame_geometry_prep_p50_ms: f64,
    frame_geometry_prep_p95_ms: f64,
    cache_builds: u64,
    cache_hits: u64,
    cache_hit_rate: f64,
    greedy_meshing_recommended: bool,
}

fn main() {
    let occupied = dense_chunk();
    let chunk = ChunkCoord { x: 0, y: 0, z: 0 };
    let key = RenderCacheKey {
        world_identity: "dense-benchmark".to_string(),
        chunk,
        overlay_revision: chunk_occupancy_revision(chunk, &occupied, CHUNK_EDGE),
        settings_revision: 0,
    };
    let mut cache = ChunkRenderCache::default();
    let started = Instant::now();
    let exposed_faces = cache
        .get_or_build(key.clone(), &occupied, CHUNK_EDGE)
        .faces
        .len();
    let cold_build_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let started = Instant::now();
        std::hint::black_box(cache.get_or_build(key.clone(), &occupied, CHUNK_EDGE));
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    samples.sort_by(f64::total_cmp);
    let naive_faces = occupied.len() * 6;
    let hit_rate = cache.hits() as f64 / (cache.hits() + cache.builds()) as f64;
    let report = Report {
        schema_version: 1,
        chunk_edge: CHUNK_EDGE,
        occupied_voxels: occupied.len(),
        exposed_faces,
        emitted_triangles: exposed_faces * 2,
        naive_faces,
        face_reduction_fraction: 1.0 - exposed_faces as f64 / naive_faces as f64,
        cold_build_ms,
        frame_geometry_prep_p50_ms: percentile(&samples, 0.50),
        frame_geometry_prep_p95_ms: percentile(&samples, 0.95),
        cache_builds: cache.builds(),
        cache_hits: cache.hits(),
        cache_hit_rate: hit_rate,
        // A solid chunk still emits 6 * edge^2 quads. This is small enough for
        // the current CPU painter; greedy merging can wait for camera-LOD data.
        greedy_meshing_recommended: exposed_faces > 10_000,
    };
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn dense_chunk() -> HashSet<StorageCoord> {
    let mut occupied = HashSet::with_capacity((CHUNK_EDGE.pow(3)) as usize);
    for z in 0..CHUNK_EDGE {
        for y in 0..CHUNK_EDGE {
            for x in 0..CHUNK_EDGE {
                occupied.insert(StorageCoord { x, y, z });
            }
        }
    }
    occupied
}

fn percentile(samples: &[f64], percentile: f64) -> f64 {
    let index = ((samples.len() - 1) as f64 * percentile).round() as usize;
    samples[index]
}
