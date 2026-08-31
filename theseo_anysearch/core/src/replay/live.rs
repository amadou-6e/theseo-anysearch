use std::{
    collections::{BTreeMap, HashMap},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{SyncSender, TrySendError},
        Arc,
    },
};

use crate::voxel::world::StorageCoord;

use super::regional::ReplayMutation;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiveIdentity {
    pub episode_id: String,
    pub world_identity: String,
    pub reset_generation: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LivePayload {
    Delta(Vec<ReplayMutation>),
    Snapshot(Vec<ReplayMutation>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct LiveUpdate {
    pub identity: LiveIdentity,
    pub revision: u64,
    pub cursor: StorageCoord,
    pub payload: LivePayload,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LiveFrame {
    pub identity: LiveIdentity,
    pub revision: u64,
    pub cursor: StorageCoord,
    pub overlay: Vec<ReplayMutation>,
    pub changed_coordinates: Vec<StorageCoord>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LiveMetrics {
    pub queued_updates: usize,
    pub applied_updates: u64,
    pub stale_updates: u64,
    pub duplicate_updates: u64,
    pub dropped_updates: u64,
    pub revision_lag: u64,
}

pub struct RevisionAssembler {
    identity: Option<LiveIdentity>,
    revision: u64,
    overlay: HashMap<StorageCoord, bool>,
    pending: BTreeMap<u64, LiveUpdate>,
    maximum_pending: usize,
    metrics: LiveMetrics,
    latest_frame: Option<LiveFrame>,
}

impl RevisionAssembler {
    pub fn new(maximum_pending: usize) -> Self {
        assert!(
            maximum_pending > 0,
            "live queue must have positive capacity"
        );
        Self {
            identity: None,
            revision: 0,
            overlay: HashMap::new(),
            pending: BTreeMap::new(),
            maximum_pending,
            metrics: LiveMetrics::default(),
            latest_frame: None,
        }
    }

    pub fn ingest(&mut self, update: LiveUpdate) -> Option<&LiveFrame> {
        if self.identity.as_ref() != Some(&update.identity) {
            if self.identity.as_ref().is_some_and(|current| {
                current.episode_id == update.identity.episode_id
                    && current.reset_generation > update.identity.reset_generation
            }) {
                self.metrics.stale_updates += 1;
                return self.latest_frame.as_ref();
            }
            self.identity = Some(update.identity.clone());
            self.revision = 0;
            self.overlay.clear();
            self.pending.clear();
            self.latest_frame = None;
        }
        if update.revision <= self.revision {
            self.metrics.stale_updates += 1;
            return self.latest_frame.as_ref();
        }
        if self.pending.contains_key(&update.revision) {
            self.metrics.duplicate_updates += 1;
            return self.latest_frame.as_ref();
        }
        self.pending.insert(update.revision, update);
        self.recover_from_snapshot_if_needed();
        while let Some(update) = self.pending.remove(&(self.revision + 1)) {
            self.apply(update);
        }
        while self.pending.len() > self.maximum_pending {
            self.pending.pop_last();
            self.metrics.dropped_updates += 1;
        }
        self.refresh_metrics();
        self.latest_frame.as_ref()
    }

    fn recover_from_snapshot_if_needed(&mut self) {
        if self.pending.contains_key(&(self.revision + 1)) {
            return;
        }
        let snapshot_revision = self.pending.iter().rev().find_map(|(revision, update)| {
            matches!(update.payload, LivePayload::Snapshot(_)).then_some(*revision)
        });
        let Some(snapshot_revision) = snapshot_revision else {
            return;
        };
        let update = self
            .pending
            .remove(&snapshot_revision)
            .expect("snapshot exists");
        self.metrics.dropped_updates += snapshot_revision.saturating_sub(self.revision + 1);
        self.pending
            .retain(|revision, _| *revision > snapshot_revision);
        self.apply(update);
    }

    fn apply(&mut self, update: LiveUpdate) {
        let mut changed_coordinates = Vec::new();
        match update.payload {
            LivePayload::Snapshot(mutations) => {
                self.overlay.clear();
                for mutation in mutations {
                    changed_coordinates.push(mutation.coordinate);
                    self.overlay.insert(mutation.coordinate, mutation.occupied);
                }
            }
            LivePayload::Delta(mutations) => {
                for mutation in mutations {
                    changed_coordinates.push(mutation.coordinate);
                    self.overlay.insert(mutation.coordinate, mutation.occupied);
                }
            }
        }
        changed_coordinates.sort_by_key(|coordinate| coordinate.global_key());
        changed_coordinates.dedup();
        self.revision = update.revision;
        self.metrics.applied_updates += 1;
        let mut overlay = self
            .overlay
            .iter()
            .map(|(coordinate, occupied)| ReplayMutation {
                coordinate: *coordinate,
                occupied: *occupied,
            })
            .collect::<Vec<_>>();
        overlay.sort_by_key(|mutation| mutation.coordinate.global_key());
        self.latest_frame = Some(LiveFrame {
            identity: update.identity,
            revision: update.revision,
            cursor: update.cursor,
            overlay,
            changed_coordinates,
        });
    }

    fn refresh_metrics(&mut self) {
        self.metrics.queued_updates = self.pending.len();
        self.metrics.revision_lag = self
            .pending
            .last_key_value()
            .map(|(revision, _)| revision.saturating_sub(self.revision))
            .unwrap_or(0);
    }

    pub fn metrics(&self) -> LiveMetrics {
        self.metrics
    }
}

#[derive(Clone)]
pub struct LivePublisher {
    sender: SyncSender<LiveUpdate>,
    dropped: Arc<AtomicU64>,
}

impl LivePublisher {
    pub fn new(sender: SyncSender<LiveUpdate>) -> Self {
        Self {
            sender,
            dropped: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn publish(&self, update: LiveUpdate) -> bool {
        match self.sender.try_send(update) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    pub fn dropped_updates(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::sync_channel;

    use super::*;

    fn identity(generation: u64) -> LiveIdentity {
        LiveIdentity {
            episode_id: "episode".to_string(),
            world_identity: "world".to_string(),
            reset_generation: generation,
        }
    }

    fn mutation(x: u32, occupied: bool) -> ReplayMutation {
        ReplayMutation {
            coordinate: StorageCoord { x, y: 1, z: 1 },
            occupied,
        }
    }

    fn update(revision: u64, payload: LivePayload) -> LiveUpdate {
        LiveUpdate {
            identity: identity(1),
            revision,
            cursor: StorageCoord {
                x: revision as u32,
                y: 1,
                z: 1,
            },
            payload,
        }
    }

    #[test]
    fn reordered_deltas_apply_only_at_complete_revision_boundaries() {
        let mut assembler = RevisionAssembler::new(8);
        assembler.ingest(update(2, LivePayload::Delta(vec![mutation(2, true)])));
        assert!(assembler.latest_frame.is_none());
        let frame = assembler
            .ingest(update(1, LivePayload::Delta(vec![mutation(1, true)])))
            .unwrap();
        assert_eq!(frame.revision, 2);
        assert_eq!(frame.overlay, vec![mutation(1, true), mutation(2, true)]);
    }

    #[test]
    fn duplicated_and_late_updates_are_discarded() {
        let mut assembler = RevisionAssembler::new(8);
        assembler.ingest(update(1, LivePayload::Delta(vec![mutation(1, true)])));
        assembler.ingest(update(1, LivePayload::Delta(vec![mutation(1, false)])));
        assert_eq!(assembler.metrics().stale_updates, 1);
        assert_eq!(
            assembler.latest_frame.as_ref().unwrap().overlay,
            vec![mutation(1, true)]
        );
    }

    #[test]
    fn snapshot_recovers_from_dropped_delta_without_mixing_revisions() {
        let mut assembler = RevisionAssembler::new(2);
        let frame = assembler
            .ingest(update(
                3,
                LivePayload::Snapshot(vec![mutation(1, true), mutation(3, true)]),
            ))
            .unwrap();
        assert_eq!(frame.revision, 3);
        assert_eq!(assembler.metrics().dropped_updates, 2);
    }

    #[test]
    fn reset_discards_pending_work_and_old_generation_updates() {
        let mut assembler = RevisionAssembler::new(8);
        assembler.ingest(update(2, LivePayload::Delta(vec![mutation(2, true)])));
        let mut reset = update(1, LivePayload::Snapshot(vec![mutation(8, true)]));
        reset.identity = identity(2);
        assert_eq!(
            assembler.ingest(reset).unwrap().overlay,
            vec![mutation(8, true)]
        );
        let old = update(1, LivePayload::Snapshot(vec![mutation(9, true)]));
        assert_eq!(
            assembler.ingest(old).unwrap().overlay,
            vec![mutation(8, true)]
        );
        assert_eq!(assembler.metrics().stale_updates, 1);
    }

    #[test]
    fn slow_or_disconnected_viewer_never_blocks_publisher() {
        let (sender, receiver) = sync_channel(1);
        let publisher = LivePublisher::new(sender);
        assert!(publisher.publish(update(1, LivePayload::Delta(Vec::new()))));
        assert!(!publisher.publish(update(2, LivePayload::Delta(Vec::new()))));
        drop(receiver);
        assert!(!publisher.publish(update(3, LivePayload::Delta(Vec::new()))));
        assert_eq!(publisher.dropped_updates(), 2);
    }
}
