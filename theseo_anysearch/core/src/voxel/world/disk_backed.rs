use std::{
    collections::{HashMap, VecDeque},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{mpsc, Arc, Mutex, OnceLock, Weak},
};

use flate2::read::ZlibDecoder;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::{
    regional::validate_coordinate, Block, BoundedRegion, StorageCoord, WorldAccessError,
    WorldExtent, WorldRead, WorldResidency,
};

type ChunkKey = (u32, u32, u32);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DiskCacheMetrics {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub pack_reads: u64,
    pub evictions: u64,
    pub decoded_bytes: usize,
    pub pinned_bytes: usize,
    pub pinned_overcommit_bytes: usize,
    pub resident_chunks: usize,
    pub pinned_chunks: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct AxisExtent {
    x: u32,
    y: u32,
    z: u32,
}

impl AxisExtent {
    const fn world_extent(&self) -> WorldExtent {
        WorldExtent {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct ChunkCoordinate {
    x: u32,
    y: u32,
    z: u32,
}

#[derive(Clone, Debug, Deserialize)]
struct ChunkManifest {
    coordinate: ChunkCoordinate,
    pack_offset: u64,
    byte_length: usize,
    occupied_voxels: u64,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct PackManifest {
    schema_version: u32,
    coordinate_type: String,
    extent: AxisExtent,
    chunk_shape: AxisExtent,
    chunks: Vec<ChunkManifest>,
}

#[derive(Clone, Debug)]
struct ChunkLocation {
    offset: u64,
    byte_length: usize,
    shape: WorldExtent,
    sha256: [u8; 32],
}

#[derive(Debug)]
struct DecodedChunk {
    occupied: Vec<u8>,
    shape: WorldExtent,
}

impl DecodedChunk {
    fn occupied(&self, local: StorageCoord) -> bool {
        let index = local.x as usize * self.shape.y as usize * self.shape.z as usize
            + local.y as usize * self.shape.z as usize
            + local.z as usize;
        self.occupied[index / 8] & (1 << (index % 8)) != 0
    }

    fn decoded_bytes(&self) -> usize {
        self.occupied.len()
    }
}

#[derive(Debug)]
struct CacheState {
    values: HashMap<ChunkKey, Arc<DecodedChunk>>,
    lru: VecDeque<ChunkKey>,
    pins: HashMap<ChunkKey, usize>,
    metrics: DiskCacheMetrics,
}

impl CacheState {
    fn touch(&mut self, key: ChunkKey) {
        self.lru.retain(|candidate| *candidate != key);
        self.lru.push_back(key);
    }
}

#[derive(Debug)]
struct DiskInner {
    pack_path: PathBuf,
    extent: WorldExtent,
    chunk_shape: WorldExtent,
    locations: HashMap<ChunkKey, ChunkLocation>,
    block_count: u64,
    maximum_decoded_bytes: usize,
    cache: Mutex<CacheState>,
    prefetch_sender: OnceLock<PrefetchSender>,
}

type PrefetchResultSender = mpsc::SyncSender<Result<(), WorldAccessError>>;
type PrefetchSender = mpsc::Sender<(BoundedRegion, PrefetchResultSender)>;

type RegistryKey = (PathBuf, usize);

fn process_registry() -> &'static Mutex<HashMap<RegistryKey, Weak<DiskInner>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<RegistryKey, Weak<DiskInner>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Clone, Debug)]
pub struct DiskBackedWorld {
    inner: Arc<DiskInner>,
}

#[derive(Debug)]
pub struct DiskResidentGuard {
    inner: Arc<DiskInner>,
    keys: Vec<ChunkKey>,
    region: BoundedRegion,
}

#[derive(Debug)]
pub struct PrefetchRequest {
    result: Mutex<mpsc::Receiver<Result<(), WorldAccessError>>>,
}

impl PrefetchRequest {
    pub fn wait(self) -> Result<(), WorldAccessError> {
        self.result
            .into_inner()
            .map_err(|_| WorldAccessError::BackendFailure)?
            .recv()
            .map_err(|_| WorldAccessError::BackendFailure)?
    }
}

impl DiskResidentGuard {
    pub const fn region(&self) -> BoundedRegion {
        self.region
    }
}

impl Drop for DiskResidentGuard {
    fn drop(&mut self) {
        let mut cache = self.inner.cache.lock().expect("disk cache mutex poisoned");
        for key in &self.keys {
            if let Some(count) = cache.pins.get_mut(key) {
                *count -= 1;
                if *count == 0 {
                    cache.pins.remove(key);
                }
            }
        }
        self.inner.refresh_metrics_and_evict(&mut cache);
    }
}

impl DiskInner {
    fn chunk_key(&self, coord: StorageCoord) -> ChunkKey {
        (
            coord.x / self.chunk_shape.x,
            coord.y / self.chunk_shape.y,
            coord.z / self.chunk_shape.z,
        )
    }

    fn local_coord(&self, coord: StorageCoord) -> StorageCoord {
        StorageCoord {
            x: coord.x % self.chunk_shape.x,
            y: coord.y % self.chunk_shape.y,
            z: coord.z % self.chunk_shape.z,
        }
    }

    fn keys_for_region(&self, region: BoundedRegion) -> Vec<ChunkKey> {
        let minimum = self.chunk_key(region.minimum);
        let maximum = self.chunk_key(StorageCoord {
            x: region.maximum_exclusive.x - 1,
            y: region.maximum_exclusive.y - 1,
            z: region.maximum_exclusive.z - 1,
        });
        let mut keys = Vec::new();
        for z in minimum.2..=maximum.2 {
            for y in minimum.1..=maximum.1 {
                for x in minimum.0..=maximum.0 {
                    let key = (x, y, z);
                    if self.locations.contains_key(&key) {
                        keys.push(key);
                    }
                }
            }
        }
        keys
    }

    fn load(&self, key: ChunkKey) -> Result<Arc<DecodedChunk>, WorldAccessError> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| WorldAccessError::BackendFailure)?;
        if let Some(chunk) = cache.values.get(&key).cloned() {
            cache.metrics.cache_hits += 1;
            cache.touch(key);
            return Ok(chunk);
        }
        let location = self
            .locations
            .get(&key)
            .ok_or(WorldAccessError::BackendFailure)?;
        cache.metrics.cache_misses += 1;
        let mut pack = File::open(&self.pack_path).map_err(|_| WorldAccessError::BackendFailure)?;
        pack.seek(SeekFrom::Start(location.offset))
            .map_err(|_| WorldAccessError::BackendFailure)?;
        let mut payload = vec![0; location.byte_length];
        pack.read_exact(&mut payload)
            .map_err(|_| WorldAccessError::BackendFailure)?;
        cache.metrics.pack_reads += 1;
        if Sha256::digest(&payload).as_slice() != location.sha256 {
            return Err(WorldAccessError::BackendFailure);
        }
        let chunk = Arc::new(decode_chunk(&payload, location.shape)?);
        cache.values.insert(key, chunk.clone());
        cache.touch(key);
        self.refresh_metrics_and_evict(&mut cache);
        Ok(chunk)
    }

    fn refresh_metrics_and_evict(&self, cache: &mut CacheState) {
        let mut decoded = cache
            .values
            .values()
            .map(|chunk| chunk.decoded_bytes())
            .sum();
        while decoded > self.maximum_decoded_bytes {
            let Some(key) = cache.lru.pop_front() else {
                break;
            };
            if cache.pins.contains_key(&key) {
                cache.lru.push_back(key);
                if cache
                    .lru
                    .iter()
                    .all(|candidate| cache.pins.contains_key(candidate))
                {
                    break;
                }
                continue;
            }
            if let Some(chunk) = cache.values.remove(&key) {
                decoded -= chunk.decoded_bytes();
                cache.metrics.evictions += 1;
            }
        }
        let pinned_bytes = cache
            .pins
            .keys()
            .filter_map(|key| cache.values.get(key))
            .map(|chunk| chunk.decoded_bytes())
            .sum();
        cache.metrics.decoded_bytes = decoded;
        cache.metrics.pinned_bytes = pinned_bytes;
        cache.metrics.pinned_overcommit_bytes = decoded.saturating_sub(self.maximum_decoded_bytes);
        cache.metrics.resident_chunks = cache.values.len();
        cache.metrics.pinned_chunks = cache.pins.len();
    }
}

impl DiskBackedWorld {
    pub fn open(root: &Path, maximum_decoded_bytes: usize) -> Result<Self, WorldAccessError> {
        if maximum_decoded_bytes == 0 {
            return Err(WorldAccessError::BackendFailure);
        }
        let root = root
            .canonicalize()
            .map_err(|_| WorldAccessError::BackendFailure)?;
        let registry_key = (root.clone(), maximum_decoded_bytes);
        if let Some(inner) = process_registry()
            .lock()
            .map_err(|_| WorldAccessError::BackendFailure)?
            .get(&registry_key)
            .and_then(Weak::upgrade)
        {
            return Ok(Self { inner });
        }
        let manifest: PackManifest = serde_json::from_slice(
            &std::fs::read(root.join("manifest.json"))
                .map_err(|_| WorldAccessError::BackendFailure)?,
        )
        .map_err(|_| WorldAccessError::BackendFailure)?;
        if manifest.schema_version != 1 || manifest.coordinate_type != "u32" {
            return Err(WorldAccessError::Unsupported);
        }
        let extent = manifest.extent.world_extent();
        let chunk_shape = manifest.chunk_shape.world_extent();
        if chunk_shape.voxel_count().is_none()
            || chunk_shape.x == 0
            || chunk_shape.y == 0
            || chunk_shape.z == 0
        {
            return Err(WorldAccessError::BackendFailure);
        }
        let mut locations = HashMap::new();
        let mut block_count = 0u64;
        for chunk in manifest.chunks {
            let key = (chunk.coordinate.x, chunk.coordinate.y, chunk.coordinate.z);
            let origin = (
                key.0.checked_mul(chunk_shape.x),
                key.1.checked_mul(chunk_shape.y),
                key.2.checked_mul(chunk_shape.z),
            );
            let (Some(origin_x), Some(origin_y), Some(origin_z)) = origin else {
                return Err(WorldAccessError::BackendFailure);
            };
            if origin_x >= extent.x || origin_y >= extent.y || origin_z >= extent.z {
                return Err(WorldAccessError::BackendFailure);
            }
            let shape = WorldExtent {
                x: chunk_shape.x.min(extent.x - origin_x),
                y: chunk_shape.y.min(extent.y - origin_y),
                z: chunk_shape.z.min(extent.z - origin_z),
            };
            locations.insert(
                key,
                ChunkLocation {
                    offset: chunk.pack_offset,
                    byte_length: chunk.byte_length,
                    shape,
                    sha256: parse_sha256(&chunk.sha256)?,
                },
            );
            block_count = block_count
                .checked_add(chunk.occupied_voxels)
                .ok_or(WorldAccessError::BackendFailure)?;
        }
        let inner = Arc::new(DiskInner {
            pack_path: root.join("world.pack"),
            extent,
            chunk_shape,
            locations,
            block_count,
            maximum_decoded_bytes,
            cache: Mutex::new(CacheState {
                values: HashMap::new(),
                lru: VecDeque::new(),
                pins: HashMap::new(),
                metrics: DiskCacheMetrics::default(),
            }),
            prefetch_sender: OnceLock::new(),
        });
        start_prefetch_worker(&inner)?;
        let mut registry = process_registry()
            .lock()
            .map_err(|_| WorldAccessError::BackendFailure)?;
        if let Some(existing) = registry.get(&registry_key).and_then(Weak::upgrade) {
            return Ok(Self { inner: existing });
        }
        registry.retain(|_, candidate| candidate.strong_count() > 0);
        registry.insert(registry_key, Arc::downgrade(&inner));
        Ok(Self { inner })
    }

    pub fn prefetch_region(&self, region: BoundedRegion) -> Result<(), WorldAccessError> {
        let region =
            BoundedRegion::new(region.minimum, region.maximum_exclusive, self.inner.extent)?;
        for key in self.inner.keys_for_region(region) {
            self.inner.load(key)?;
        }
        Ok(())
    }

    pub fn request_prefetch(&self, region: BoundedRegion) -> PrefetchRequest {
        let (result_sender, result) = mpsc::sync_channel(1);
        let request = PrefetchRequest {
            result: Mutex::new(result),
        };
        let sent = self
            .inner
            .prefetch_sender
            .get()
            .ok_or(WorldAccessError::BackendFailure)
            .and_then(|sender| {
                sender
                    .send((region, result_sender.clone()))
                    .map_err(|_| WorldAccessError::BackendFailure)
            });
        if let Err(error) = sent {
            let _ = result_sender.send(Err(error));
        }
        request
    }

    pub fn pin_regions(
        &self,
        regions: &[BoundedRegion],
    ) -> Result<DiskResidentGuard, WorldAccessError> {
        if regions.is_empty() {
            return Err(WorldAccessError::BackendFailure);
        }
        let mut keys = Vec::new();
        for region in regions {
            let validated =
                BoundedRegion::new(region.minimum, region.maximum_exclusive, self.inner.extent)?;
            keys.extend(self.inner.keys_for_region(validated));
        }
        keys.sort_unstable();
        keys.dedup();
        {
            let mut cache = self
                .inner
                .cache
                .lock()
                .map_err(|_| WorldAccessError::BackendFailure)?;
            for key in &keys {
                *cache.pins.entry(*key).or_insert(0) += 1;
            }
        }
        for key in &keys {
            if let Err(error) = self.inner.load(*key) {
                let mut cache = self
                    .inner
                    .cache
                    .lock()
                    .map_err(|_| WorldAccessError::BackendFailure)?;
                for rollback in &keys {
                    if let Some(count) = cache.pins.get_mut(rollback) {
                        *count -= 1;
                        if *count == 0 {
                            cache.pins.remove(rollback);
                        }
                    }
                }
                return Err(error);
            }
        }
        let mut cache = self
            .inner
            .cache
            .lock()
            .map_err(|_| WorldAccessError::BackendFailure)?;
        self.inner.refresh_metrics_and_evict(&mut cache);
        drop(cache);
        let minimum = StorageCoord {
            x: regions.iter().map(|region| region.minimum.x).min().unwrap(),
            y: regions.iter().map(|region| region.minimum.y).min().unwrap(),
            z: regions.iter().map(|region| region.minimum.z).min().unwrap(),
        };
        let maximum_exclusive = StorageCoord {
            x: regions
                .iter()
                .map(|region| region.maximum_exclusive.x)
                .max()
                .unwrap(),
            y: regions
                .iter()
                .map(|region| region.maximum_exclusive.y)
                .max()
                .unwrap(),
            z: regions
                .iter()
                .map(|region| region.maximum_exclusive.z)
                .max()
                .unwrap(),
        };
        Ok(DiskResidentGuard {
            inner: self.inner.clone(),
            keys,
            region: BoundedRegion {
                minimum,
                maximum_exclusive,
            },
        })
    }

    pub fn metrics(&self) -> DiskCacheMetrics {
        self.inner
            .cache
            .lock()
            .expect("disk cache mutex poisoned")
            .metrics
    }
}

impl WorldRead for DiskBackedWorld {
    fn extent(&self) -> WorldExtent {
        self.inner.extent
    }

    fn get_block_value(&self, coord: StorageCoord) -> Result<Option<Block>, WorldAccessError> {
        validate_coordinate(self.inner.extent, coord)?;
        let key = self.inner.chunk_key(coord);
        if !self.inner.locations.contains_key(&key) {
            return Ok(None);
        }
        let chunk = self.inner.load(key)?;
        Ok(chunk
            .occupied(self.inner.local_coord(coord))
            .then(Block::default))
    }

    fn blocks_in_region(
        &self,
        region: BoundedRegion,
    ) -> Result<Vec<(StorageCoord, Block)>, WorldAccessError> {
        let region =
            BoundedRegion::new(region.minimum, region.maximum_exclusive, self.inner.extent)?;
        let mut values = Vec::new();
        for key in self.inner.keys_for_region(region) {
            let chunk = self.inner.load(key)?;
            let origin = StorageCoord {
                x: key.0 * self.inner.chunk_shape.x,
                y: key.1 * self.inner.chunk_shape.y,
                z: key.2 * self.inner.chunk_shape.z,
            };
            for x in 0..chunk.shape.x {
                for y in 0..chunk.shape.y {
                    for z in 0..chunk.shape.z {
                        let local = StorageCoord { x, y, z };
                        let coordinate = StorageCoord {
                            x: origin.x + x,
                            y: origin.y + y,
                            z: origin.z + z,
                        };
                        if region.contains(coordinate) && chunk.occupied(local) {
                            values.push((coordinate, Block::default()));
                        }
                    }
                }
            }
        }
        values.sort_by_key(|(coordinate, _)| coordinate.global_key());
        Ok(values)
    }

    fn block_count(&self) -> u64 {
        self.inner.block_count
    }
}

impl WorldResidency for DiskBackedWorld {
    type Guard<'a> = DiskResidentGuard;

    fn is_region_resident(&self, region: BoundedRegion) -> bool {
        if BoundedRegion::new(region.minimum, region.maximum_exclusive, self.inner.extent).is_err()
        {
            return false;
        }
        let cache = self.inner.cache.lock().expect("disk cache mutex poisoned");
        self.inner
            .keys_for_region(region)
            .iter()
            .all(|key| cache.values.contains_key(key))
    }

    fn pin_region(&self, region: BoundedRegion) -> Result<Self::Guard<'_>, WorldAccessError> {
        let region =
            BoundedRegion::new(region.minimum, region.maximum_exclusive, self.inner.extent)?;
        self.pin_regions(&[region])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::world::{World, WorldState};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture() -> (PathBuf, DiskBackedWorld) {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("theseo-disk-world-{suffix}"));
        std::fs::create_dir_all(&root).unwrap();
        let first = [
            b"AWC1".as_slice(),
            &[2],
            &0u32.to_le_bytes(),
            &7u32.to_le_bytes(),
        ]
        .concat();
        let second = [b"AWC1".as_slice(), &[2], &1u32.to_le_bytes()].concat();
        let first_sha = format!("{:x}", Sha256::digest(&first));
        let second_sha = format!("{:x}", Sha256::digest(&second));
        std::fs::write(
            root.join("world.pack"),
            [first.as_slice(), second.as_slice()].concat(),
        )
        .unwrap();
        let manifest = serde_json::json!({
            "schema_version": 1, "coordinate_type": "u32",
            "extent": {"x": 4, "y": 2, "z": 2},
            "chunk_shape": {"x": 2, "y": 2, "z": 2},
            "chunks": [
                {"coordinate": {"x": 0, "y": 0, "z": 0}, "pack_offset": 0, "byte_length": first.len(), "occupied_voxels": 2, "sha256": first_sha},
                {"coordinate": {"x": 1, "y": 0, "z": 0}, "pack_offset": first.len(), "byte_length": second.len(), "occupied_voxels": 1, "sha256": second_sha}
            ]
        });
        std::fs::write(
            root.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let world = DiskBackedWorld::open(&root, 1).unwrap();
        (root, world)
    }

    fn region(minimum: StorageCoord, maximum_exclusive: StorageCoord) -> BoundedRegion {
        BoundedRegion::new(minimum, maximum_exclusive, WorldExtent { x: 4, y: 2, z: 2 }).unwrap()
    }

    #[test]
    fn cold_reads_fault_once_and_hot_reads_do_not_touch_the_pack() {
        let (root, world) = fixture();
        let coordinate = StorageCoord { x: 0, y: 0, z: 0 };
        assert!(world.get_block_value(coordinate).unwrap().is_some());
        assert!(world.get_block_value(coordinate).unwrap().is_some());
        let metrics = world.metrics();
        assert_eq!(metrics.pack_reads, 1);
        assert_eq!(metrics.cache_misses, 1);
        assert_eq!(metrics.cache_hits, 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn pinning_union_allows_reported_overcommit_then_evicts_on_drop() {
        let (root, world) = fixture();
        let first = region(
            StorageCoord { x: 0, y: 0, z: 0 },
            StorageCoord { x: 2, y: 2, z: 2 },
        );
        let second = region(
            StorageCoord { x: 2, y: 0, z: 0 },
            StorageCoord { x: 4, y: 2, z: 2 },
        );
        let guard = world.pin_regions(&[first, second]).unwrap();
        assert_eq!(guard.region().maximum_exclusive.x, 4);
        let pinned = world.metrics();
        assert_eq!(pinned.pinned_chunks, 2);
        assert_eq!(pinned.pinned_overcommit_bytes, 1);
        drop(guard);
        let released = world.metrics();
        assert_eq!(released.pinned_chunks, 0);
        assert!(released.decoded_bytes <= 1);
        assert!(released.evictions > 0);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn clones_share_the_same_process_cache_and_async_prefetch() {
        let (root, world) = fixture();
        let region = region(
            StorageCoord { x: 2, y: 0, z: 0 },
            StorageCoord { x: 4, y: 2, z: 2 },
        );
        world.request_prefetch(region).wait().unwrap();
        let clone = world.clone();
        assert!(clone
            .get_block_value(StorageCoord { x: 2, y: 0, z: 1 })
            .unwrap()
            .is_some());
        assert_eq!(clone.metrics().pack_reads, 1);
        assert_eq!(clone.metrics().cache_hits, 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn independent_opens_share_one_process_cache() {
        let (root, first) = fixture();
        let second = DiskBackedWorld::open(&root, 1).unwrap();
        assert!(first
            .get_block_value(StorageCoord { x: 0, y: 0, z: 0 })
            .unwrap()
            .is_some());
        assert!(second
            .get_block_value(StorageCoord { x: 0, y: 0, z: 0 })
            .unwrap()
            .is_some());
        assert_eq!(second.metrics().pack_reads, 1);
        assert_eq!(second.metrics().cache_hits, 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cold_cross_chunk_mutations_fault_before_overlay_commit() {
        let (root, _) = fixture();
        let mut world = WorldState::from_compiled_pack(&root, 1).unwrap();
        world
            .set_block(
                (2, 0, 1),
                Block {
                    kind: 9,
                    ..Block::default()
                },
            )
            .unwrap();
        assert_eq!(world.get_block((2, 0, 1)).unwrap().kind, 9);
        world.remove_block((2, 0, 1)).unwrap();
        assert!(world.get_block((2, 0, 1)).is_none());
        world.set_block((3, 1, 1), Block::default()).unwrap();
        assert!(world.get_block((3, 1, 1)).is_some());
        assert!(world.disk_cache_metrics().unwrap().pack_reads >= 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn chunk_corruption_after_open_is_rejected_on_cold_fault() {
        let (root, world) = fixture();
        let mut payload = std::fs::read(root.join("world.pack")).unwrap();
        payload[5] ^= 1;
        std::fs::write(root.join("world.pack"), payload).unwrap();

        assert_eq!(
            world.get_block_value(StorageCoord { x: 0, y: 0, z: 0 }),
            Err(WorldAccessError::BackendFailure)
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn short_read_surfaces_through_async_prefetch() {
        let (root, world) = fixture();
        std::fs::write(root.join("world.pack"), b"short").unwrap();
        let region = region(
            StorageCoord { x: 2, y: 0, z: 0 },
            StorageCoord { x: 4, y: 2, z: 2 },
        );

        assert_eq!(
            world.request_prefetch(region).wait(),
            Err(WorldAccessError::BackendFailure)
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}

fn decode_chunk(payload: &[u8], shape: WorldExtent) -> Result<DecodedChunk, WorldAccessError> {
    if payload.len() < 5 || &payload[..4] != b"AWC1" {
        return Err(WorldAccessError::BackendFailure);
    }
    let size = shape
        .voxel_count()
        .ok_or(WorldAccessError::BackendFailure)? as usize;
    let mut occupied = vec![0u8; (size + 7) / 8];
    match payload[4] {
        1 if payload.len() == 5 => occupied.fill(0xff),
        2 if (payload.len() - 5) % 4 == 0 => {
            for raw in payload[5..].chunks_exact(4) {
                let index = u32::from_le_bytes(raw.try_into().unwrap()) as usize;
                if index >= size {
                    return Err(WorldAccessError::BackendFailure);
                }
                occupied[index / 8] |= 1 << (index % 8);
            }
        }
        3 => {
            let mut decoder = ZlibDecoder::new(&payload[5..]);
            let mut decoded = Vec::new();
            decoder
                .read_to_end(&mut decoded)
                .map_err(|_| WorldAccessError::BackendFailure)?;
            if decoded.len() != (size + 7) / 8 {
                return Err(WorldAccessError::BackendFailure);
            }
            occupied = decoded;
        }
        _ => return Err(WorldAccessError::BackendFailure),
    }
    if size % 8 != 0 {
        let mask = (1u16 << (size % 8)) as u8 - 1;
        if let Some(last) = occupied.last_mut() {
            *last &= mask;
        }
    }
    Ok(DecodedChunk { occupied, shape })
}

fn parse_sha256(value: &str) -> Result<[u8; 32], WorldAccessError> {
    if value.len() != 64 {
        return Err(WorldAccessError::BackendFailure);
    }
    let mut bytes = [0u8; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| WorldAccessError::BackendFailure)?;
    }
    Ok(bytes)
}

fn start_prefetch_worker(inner: &Arc<DiskInner>) -> Result<(), WorldAccessError> {
    let (sender, receiver) = mpsc::channel::<(BoundedRegion, PrefetchResultSender)>();
    inner
        .prefetch_sender
        .set(sender)
        .map_err(|_| WorldAccessError::BackendFailure)?;
    let weak = Arc::downgrade(inner);
    std::thread::Builder::new()
        .name("theseo-world-prefetch".to_owned())
        .spawn(move || {
            while let Ok((region, result)) = receiver.recv() {
                let Some(inner) = weak.upgrade() else {
                    let _ = result.send(Err(WorldAccessError::BackendFailure));
                    break;
                };
                let world = DiskBackedWorld { inner };
                let _ = result.send(world.prefetch_region(region));
            }
        })
        .map_err(|_| WorldAccessError::BackendFailure)?;
    Ok(())
}
