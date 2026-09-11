use serde::{Deserialize, Serialize};

pub const WORLD_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct WorldExtent {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl WorldExtent {
    pub const fn cubic(size: u32) -> Self {
        Self {
            x: size,
            y: size,
            z: size,
        }
    }

    pub const fn contains_storage(self, coord: StorageCoord) -> bool {
        coord.x < self.x && coord.y < self.y && coord.z < self.z
    }

    pub fn voxel_count(self) -> Option<u64> {
        u64::from(self.x)
            .checked_mul(u64::from(self.y))?
            .checked_mul(u64::from(self.z))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct StorageCoord {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl StorageCoord {
    pub const fn global_key(self) -> (u32, u32, u32) {
        (self.x, self.y, self.z)
    }

    pub fn checked_scalar_id(self, extent: WorldExtent) -> Option<u64> {
        if !extent.contains_storage(self) {
            return None;
        }
        u64::from(self.z)
            .checked_mul(u64::from(extent.y))?
            .checked_add(u64::from(self.y))?
            .checked_mul(u64::from(extent.x))?
            .checked_add(u64::from(self.x))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct TaskCoord {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl TaskCoord {
    pub fn manhattan_distance(self, other: Self) -> u64 {
        u64::from(self.x.abs_diff(other.x))
            + u64::from(self.y.abs_diff(other.y))
            + u64::from(self.z.abs_diff(other.z))
    }

    pub fn checked_squared_distance(self, other: Self) -> Option<u64> {
        [
            u64::from(self.x.abs_diff(other.x)),
            u64::from(self.y.abs_diff(other.y)),
            u64::from(self.z.abs_diff(other.z)),
        ]
        .into_iter()
        .try_fold(0_u64, |total, delta| {
            total.checked_add(delta.checked_mul(delta)?)
        })
    }

    pub fn euclidean_distance(self, other: Self) -> f64 {
        let dx = f64::from(self.x.abs_diff(other.x));
        let dy = f64::from(self.y.abs_diff(other.y));
        let dz = f64::from(self.z.abs_diff(other.z));
        dx.hypot(dy).hypot(dz)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoordinateError {
    BelowTaskMinimum(TaskCoord),
    OutOfBounds(TaskCoord),
    Overflow(StorageCoord),
}

pub fn task_to_storage(
    coord: TaskCoord,
    extent: WorldExtent,
) -> Result<StorageCoord, CoordinateError> {
    let storage = StorageCoord {
        x: coord
            .x
            .checked_sub(1)
            .ok_or(CoordinateError::BelowTaskMinimum(coord))?,
        y: coord
            .y
            .checked_sub(1)
            .ok_or(CoordinateError::BelowTaskMinimum(coord))?,
        z: coord
            .z
            .checked_sub(1)
            .ok_or(CoordinateError::BelowTaskMinimum(coord))?,
    };
    if !extent.contains_storage(storage) {
        return Err(CoordinateError::OutOfBounds(coord));
    }
    Ok(storage)
}

pub fn storage_to_task(
    coord: StorageCoord,
    extent: WorldExtent,
) -> Result<TaskCoord, CoordinateError> {
    if !extent.contains_storage(coord) {
        return Err(CoordinateError::OutOfBounds(TaskCoord {
            x: coord.x.saturating_add(1),
            y: coord.y.saturating_add(1),
            z: coord.z.saturating_add(1),
        }));
    }
    Ok(TaskCoord {
        x: coord
            .x
            .checked_add(1)
            .ok_or(CoordinateError::Overflow(coord))?,
        y: coord
            .y
            .checked_add(1)
            .ok_or(CoordinateError::Overflow(coord))?,
        z: coord
            .z
            .checked_add(1)
            .ok_or(CoordinateError::Overflow(coord))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_public_minimum_and_maximum() {
        let extent = WorldExtent { x: 8, y: 4, z: 2 };
        assert_eq!(
            task_to_storage(TaskCoord { x: 1, y: 1, z: 1 }, extent),
            Ok(StorageCoord { x: 0, y: 0, z: 0 })
        );
        assert_eq!(
            task_to_storage(TaskCoord { x: 8, y: 4, z: 2 }, extent),
            Ok(StorageCoord { x: 7, y: 3, z: 1 })
        );
    }

    #[test]
    fn rejects_zero_and_per_axis_overflow() {
        let extent = WorldExtent { x: 8, y: 4, z: 2 };
        assert!(matches!(
            task_to_storage(TaskCoord { x: 0, y: 1, z: 1 }, extent),
            Err(CoordinateError::BelowTaskMinimum(_))
        ));
        assert!(matches!(
            task_to_storage(TaskCoord { x: 8, y: 5, z: 2 }, extent),
            Err(CoordinateError::OutOfBounds(_))
        ));
    }

    #[test]
    fn voxel_count_uses_checked_u64_math() {
        assert_eq!(
            WorldExtent::cubic(100_000).voxel_count(),
            Some(1_000_000_000_000_000)
        );
        assert_eq!(WorldExtent::cubic(u32::MAX).voxel_count(), None);
    }

    #[test]
    fn tuple_keys_and_scalar_ids_do_not_use_u32_flattening() {
        let extent = WorldExtent {
            x: 100_000,
            y: 50_000,
            z: 10_000,
        };
        let coord = StorageCoord {
            x: 99_999,
            y: 49_999,
            z: 9_999,
        };
        assert_eq!(coord.global_key(), (99_999, 49_999, 9_999));
        assert_eq!(coord.checked_scalar_id(extent), Some(49_999_999_999_999));
    }

    #[test]
    fn large_distances_are_widened_or_checked() {
        let minimum = TaskCoord { x: 1, y: 1, z: 1 };
        let maximum = TaskCoord {
            x: u32::MAX,
            y: u32::MAX,
            z: u32::MAX,
        };
        assert_eq!(minimum.manhattan_distance(maximum), 12_884_901_882);
        assert_eq!(minimum.checked_squared_distance(maximum), None);
        assert!(minimum.euclidean_distance(maximum).is_finite());
    }
}
