use std::{hint::black_box, mem::size_of, time::Instant};

use serde::Serialize;

use crate::{
    environments::Environment,
    voxel::{
        world::{
            Block, BoundedRegion, ChunkedWorld, GridRay, StorageCoord, WorldExtent, WorldMutation,
            WorldRead, WorldState,
        },
        VoxelAction, VoxelEnv,
    },
};

pub const DEFAULT_SEED: u64 = 0x2262_1921_8219;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryDistribution {
    Empty,
    IsolatedSparse,
    ClusteredSparse,
    Shell,
    DenseBlock,
}

#[derive(Clone, Debug)]
pub struct BenchmarkConfig {
    pub seed: u64,
    pub warmup_samples: usize,
    pub measured_samples: usize,
    pub operations_per_sample: usize,
    pub distributions: Vec<GeometryDistribution>,
}

impl BenchmarkConfig {
    pub fn smoke() -> Self {
        Self {
            seed: DEFAULT_SEED,
            warmup_samples: 0,
            measured_samples: 1,
            operations_per_sample: 1,
            distributions: vec![GeometryDistribution::Empty],
        }
    }

    pub fn baseline() -> Self {
        Self {
            seed: DEFAULT_SEED,
            warmup_samples: 1,
            measured_samples: 5,
            operations_per_sample: 256,
            distributions: Self::full().distributions,
        }
    }

    pub fn full() -> Self {
        Self {
            seed: DEFAULT_SEED,
            warmup_samples: 3,
            measured_samples: 25,
            operations_per_sample: 2_048,
            distributions: vec![
                GeometryDistribution::Empty,
                GeometryDistribution::IsolatedSparse,
                GeometryDistribution::ClusteredSparse,
                GeometryDistribution::Shell,
                GeometryDistribution::DenseBlock,
            ],
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Percentiles {
    pub minimum_ns_per_operation: f64,
    pub p50_ns_per_operation: f64,
    pub p95_ns_per_operation: f64,
    pub p99_ns_per_operation: f64,
    pub maximum_ns_per_operation: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChunkBenchmark {
    pub chunk_edge: u32,
    pub distribution: GeometryDistribution,
    pub block_count: u64,
    pub resident_chunk_count: usize,
    pub point_queries: Percentiles,
    pub region_radius_2: Percentiles,
    pub region_radius_8: Percentiles,
    pub ray_length_8: Percentiles,
    pub ray_length_32: Percentiles,
    pub mutations: Percentiles,
    pub reset: Percentiles,
    pub environment_reset: Percentiles,
    pub environment_step: Percentiles,
    pub full_enumeration: Percentiles,
    pub storage_overhead_estimate_bytes: usize,
    pub decoded_chunk_memory_estimate_bytes: usize,
    pub encoded_bytes: Option<usize>,
    pub overlay_memory_bytes: Option<usize>,
    pub pinned_memory_bytes: Option<usize>,
    pub operating_system_file_cache_bytes: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct BenchmarkReport {
    pub schema_version: u32,
    pub seed: u64,
    pub warmup_samples: usize,
    pub measured_samples: usize,
    pub operations_per_sample: usize,
    pub target_os: String,
    pub target_arch: String,
    pub build_profile: String,
    pub processor: Option<String>,
    pub query_phase: String,
    pub cold_persistence_measurements_available: bool,
    pub measurements: Vec<ChunkBenchmark>,
}

pub fn run(config: &BenchmarkConfig) -> BenchmarkReport {
    let extent = WorldExtent {
        x: 130,
        y: 97,
        z: 65,
    };
    let mut measurements = Vec::new();
    for chunk_edge in [16, 32, 64] {
        for distribution in &config.distributions {
            measurements.push(measure_chunk(extent, chunk_edge, *distribution, config));
        }
    }
    BenchmarkReport {
        schema_version: 1,
        seed: config.seed,
        warmup_samples: config.warmup_samples,
        measured_samples: config.measured_samples,
        operations_per_sample: config.operations_per_sample,
        target_os: std::env::consts::OS.to_owned(),
        target_arch: std::env::consts::ARCH.to_owned(),
        build_profile: if cfg!(debug_assertions) {
            "debug".to_owned()
        } else {
            "release".to_owned()
        },
        processor: std::env::var("PROCESSOR_IDENTIFIER").ok(),
        query_phase: "hot_in_memory_after_separate_warmup".to_owned(),
        cold_persistence_measurements_available: false,
        measurements,
    }
}

fn measure_chunk(
    extent: WorldExtent,
    chunk_edge: u32,
    distribution: GeometryDistribution,
    config: &BenchmarkConfig,
) -> ChunkBenchmark {
    eprintln!("benchmarking chunk edge {chunk_edge}, distribution {distribution:?}");
    let coordinates = geometry(distribution, extent, config.seed);
    let mut world = build_world(extent, chunk_edge, &coordinates);
    let mut random = Lcg::new(config.seed ^ u64::from(chunk_edge));
    let points = (0..config.operations_per_sample)
        .map(|_| random.coordinate(extent))
        .collect::<Vec<_>>();
    let region_2 = centered_region(points[0], 2, extent);
    let region_8 = centered_region(points[0], 8, extent);
    let ray_8 = GridRay::new(points[0], [1, 0, 0], 8).expect("valid ray");
    let ray_32 = GridRay::new(points[0], [1, 1, 0], 32).expect("valid ray");
    let full = BoundedRegion::new(
        StorageCoord { x: 0, y: 0, z: 0 },
        StorageCoord {
            x: extent.x,
            y: extent.y,
            z: extent.z,
        },
        extent,
    )
    .expect("non-empty extent");

    let point_queries = sample(config, config.operations_per_sample, || {
        for coordinate in &points {
            black_box(
                world
                    .get_block_value(*coordinate)
                    .expect("valid coordinate"),
            );
        }
    });
    let region_radius_2 = sample(config, config.operations_per_sample, || {
        for _ in 0..config.operations_per_sample {
            black_box(world.blocks_in_region(region_2).expect("valid region"));
        }
    });
    let region_radius_8 = sample(config, config.operations_per_sample, || {
        for _ in 0..config.operations_per_sample {
            black_box(world.blocks_in_region(region_8).expect("valid region"));
        }
    });
    let ray_length_8 = sample(config, config.operations_per_sample, || {
        for _ in 0..config.operations_per_sample {
            black_box(world.raycast(ray_8).expect("valid ray"));
        }
    });
    let ray_length_32 = sample(config, config.operations_per_sample, || {
        for _ in 0..config.operations_per_sample {
            black_box(world.raycast(ray_32).expect("valid ray"));
        }
    });
    let mutations = sample(config, config.operations_per_sample * 2, || {
        for coordinate in &points {
            black_box(
                world
                    .set_block_value(*coordinate, Block::default())
                    .expect("valid coordinate"),
            );
            black_box(
                world
                    .remove_block_value(*coordinate)
                    .expect("valid coordinate"),
            );
        }
    });
    let reset = sample(config, 1, || {
        world.clear_blocks();
        for coordinate in &coordinates {
            world
                .set_block_value(*coordinate, Block::default())
                .expect("valid fixture coordinate");
        }
        black_box(world.block_count());
    });
    let mut environment = VoxelEnv::new(WorldState::new_chunked(chunk_edge), u32::MAX)
        .with_grid_size(extent.x as u16)
        .with_geometry(Vec::new());
    let environment_reset = sample(config, 1, || {
        black_box(environment.reset(config.seed));
    });
    environment.reset(config.seed);
    let environment_step = sample(config, config.operations_per_sample, || {
        for _ in 0..config.operations_per_sample {
            black_box(environment.step(VoxelAction::Noop));
        }
    });
    let full_enumeration = sample(config, 1, || {
        black_box(world.blocks_in_region(full).expect("valid full region"));
    });
    eprintln!("completed chunk edge {chunk_edge}, distribution {distribution:?}");

    let resident_chunk_count = world.resident_chunk_count();
    let block_count = world.block_count();
    let entry_estimate = size_of::<(StorageCoord, Block)>() * block_count as usize;
    let chunk_estimate = resident_chunk_count * size_of::<usize>() * 8;
    ChunkBenchmark {
        chunk_edge,
        distribution,
        block_count,
        resident_chunk_count,
        point_queries,
        region_radius_2,
        region_radius_8,
        ray_length_8,
        ray_length_32,
        mutations,
        reset,
        environment_reset,
        environment_step,
        full_enumeration,
        storage_overhead_estimate_bytes: entry_estimate + chunk_estimate,
        decoded_chunk_memory_estimate_bytes: entry_estimate + chunk_estimate,
        encoded_bytes: None,
        overlay_memory_bytes: None,
        pinned_memory_bytes: None,
        operating_system_file_cache_bytes: None,
    }
}

fn sample(
    config: &BenchmarkConfig,
    operation_count: usize,
    mut operation: impl FnMut(),
) -> Percentiles {
    for _ in 0..config.warmup_samples {
        operation();
    }
    let mut samples = Vec::with_capacity(config.measured_samples);
    for _ in 0..config.measured_samples {
        let started = Instant::now();
        operation();
        samples.push(started.elapsed().as_nanos() as f64 / operation_count.max(1) as f64);
    }
    percentiles(samples)
}

fn percentiles(mut samples: Vec<f64>) -> Percentiles {
    samples.sort_by(f64::total_cmp);
    let at = |quantile: f64| {
        let index = ((samples.len() - 1) as f64 * quantile).round() as usize;
        samples[index]
    };
    Percentiles {
        minimum_ns_per_operation: samples[0],
        p50_ns_per_operation: at(0.50),
        p95_ns_per_operation: at(0.95),
        p99_ns_per_operation: at(0.99),
        maximum_ns_per_operation: samples[samples.len() - 1],
    }
}

fn build_world(extent: WorldExtent, edge: u32, coordinates: &[StorageCoord]) -> ChunkedWorld {
    let mut world = ChunkedWorld::new(extent, WorldExtent::cubic(edge)).expect("valid chunk edge");
    for coordinate in coordinates {
        world
            .set_block_value(*coordinate, Block::default())
            .expect("fixture coordinates fit extent");
    }
    world
}

fn centered_region(center: StorageCoord, radius: u32, extent: WorldExtent) -> BoundedRegion {
    let minimum = StorageCoord {
        x: center.x.saturating_sub(radius),
        y: center.y.saturating_sub(radius),
        z: center.z.saturating_sub(radius),
    };
    let maximum_exclusive = StorageCoord {
        x: center.x.saturating_add(radius + 1).min(extent.x),
        y: center.y.saturating_add(radius + 1).min(extent.y),
        z: center.z.saturating_add(radius + 1).min(extent.z),
    };
    BoundedRegion::new(minimum, maximum_exclusive, extent).expect("clamped region is non-empty")
}

fn geometry(
    distribution: GeometryDistribution,
    extent: WorldExtent,
    seed: u64,
) -> Vec<StorageCoord> {
    match distribution {
        GeometryDistribution::Empty => Vec::new(),
        GeometryDistribution::IsolatedSparse => {
            let mut random = Lcg::new(seed);
            (0..512).map(|_| random.coordinate(extent)).collect()
        }
        GeometryDistribution::ClusteredSparse => {
            let centers = [(20, 20, 20), (64, 48, 32), (110, 75, 50)];
            centers
                .into_iter()
                .flat_map(|(x, y, z)| {
                    (0..8).flat_map(move |dz| {
                        (0..8).flat_map(move |dy| {
                            (0..8).filter_map(move |dx| {
                                ((dx + dy + dz) % 3 == 0).then_some(StorageCoord {
                                    x: x + dx,
                                    y: y + dy,
                                    z: z + dz,
                                })
                            })
                        })
                    })
                })
                .collect()
        }
        GeometryDistribution::Shell => (20..52)
            .flat_map(|z| {
                (24..56).flat_map(move |y| {
                    (40..72).filter_map(move |x| {
                        (x == 40 || x == 71 || y == 24 || y == 55 || z == 20 || z == 51)
                            .then_some(StorageCoord { x, y, z })
                    })
                })
            })
            .collect(),
        GeometryDistribution::DenseBlock => (16..40)
            .flat_map(|z| {
                (16..40).flat_map(move |y| (16..40).map(move |x| StorageCoord { x, y, z }))
            })
            .collect(),
    }
}

struct Lcg(u64);

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn coordinate(&mut self, extent: WorldExtent) -> StorageCoord {
        StorageCoord {
            x: (self.next() % u64::from(extent.x)) as u32,
            y: (self.next() % u64::from(extent.y)) as u32,
            z: (self.next() % u64::from(extent.z)) as u32,
        }
    }
}
