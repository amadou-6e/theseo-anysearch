# Live visualization revision protocol

Live visualization is an optional, lossy observer of training. It must never
participate in an environment step, learner update, or Ray object-store read.

Each update carries an episode identifier, immutable world identity, reset
generation, monotonically increasing revision, cursor, and either an overlay
delta or a complete overlay snapshot. The viewer applies deltas only in
contiguous revision order. A later complete snapshot may skip a missing range
and becomes the next atomic frame; skipped revisions are counted as dropped.

Reset generation changes discard all queued fetch, delta, and mesh work from
the previous generation. Results are accepted only when their complete identity
and revision still match the viewer's active request. Cursor teleports require
no special correctness path: the cursor belongs to the atomic revision and the
bounded world loader may synchronously miss while preparing the next complete
frame. The last complete frame stays visible during that work.

The local producer uses a bounded synchronous channel with non-blocking
`try_send`. Full or disconnected channels drop visualization updates and
increment a counter; they never block training. Periodic snapshots permit a
viewer to recover after backpressure drops. The same envelope can later be
carried over a local socket or remote transport, provided that transport keeps
the non-blocking producer boundary and does not route world reads through Ray
learners or EnvRunners.

Viewer metrics report queued, applied, stale, duplicate, and dropped updates,
plus revision lag. Each completed frame exposes its changed coordinates so the
render cache can invalidate the affected chunk and both sides of any changed
chunk boundary.
