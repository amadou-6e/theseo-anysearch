# Imported 3D routing dataset contract

Status: foundation for [issue #411](https://github.com/amadou-6e/theseo-anysearch/issues/411),
not a benchmark result or authorization to train. The governing draft roadmap is
[specs@766461e](https://github.com/amadou-6e/specs/blob/766461e21d4a9c92e254557ca4a3f3e6c52eb9e5/projects/theseo-anysearch/routing-geometry-integration-roadmap.md).
Replace that pin with the merged spec SHA before integration.

The contracts live in
[`environments/routing_manifests.py`](../theseo_anysearch/environments/routing_manifests.py).
Each record is a strict, version-1 JSON sidecar with a canonical SHA-256
identity. `write_sidecar` creates an immutable envelope containing the record
type, identity and payload. `read_sidecar` rejects unexpected fields, duplicate
JSON keys, wrong types and payload tampering.

| Record | Required distinction |
| --- | --- |
| `SourceRecord` | Original revision, relative member paths, file hashes, role and rights review. No use is allowed by default. `verify_artifact` checks an extracted local file's actual bytes inside its source root. |
| `ConversionRecord` | Converter version and parameters, input source identity, output occupancy hash and optional parent world. A roof parameter change is a new conversion. |
| `RoutingWorldRecord` | Complete full-truth occupancy hash, three-axis extent, metric frame, source/conversion links, site, topology family and root geometry ID. Roofed and cropped descendants keep the same root ID but get new world identities. Partial observations belong in observation sidecars, not a world-pack free-space complement. |
| `RoutingTaskRecord` | Source query evidence versus explicit derivation; zero-based start/goal, movement model, body radius and ceiling. Any changed endpoint, radius, ceiling or world changes its task identity. |
| `RoutingObservationRecord` | Three independent masks: observed occupied, observed free and unknown. The occupancy world pack is not a sensor observation. |
| `RoutingReferenceRecord` | Unverified, feasible, independently validated or certified-optimal claim. Feasibility claims need a route; independent validation and optimality need separate verification evidence. |
| `RoutingSplitRecord` | One partition per world, with root-geometry and site groups kept together. Topology family is recorded so a later protocol can define a family holdout. |

`validate_routing_bundle` checks references across those sidecars, task bounds,
conversion-output hashes, source query evidence and exact split coverage. Its
dataset identity includes rights and split records; the source **content**
identity deliberately does not change when a rights decision changes. A
rights change therefore changes the dataset record, not the underlying
geometry identity. Neither a manifest nor a public download grants a use:
call `RightsRecord.require_allowed` before evaluation, training or
redistribution.

## Coordinate and visibility rules

Storage coordinates are zero-based. The existing live task ABI is one-based;
`storage_to_task` and `task_to_storage` are checked inverses within the
world extent. `GridFrame` stores a source-metric origin, isotropic meters per
voxel and three orthonormal storage-axis directions in the source frame.
Voxel-center conversion is reversible for both right- and left-handed frames.
Only exact voxel centers are accepted when converting source positions back to
integer storage coordinates; an importer must state its own snapping policy
for arbitrary continuous endpoints.

`validate_observation_masks` requires occupied, free and unknown masks to be
binary, disjoint and exhaustive. When full-truth occupancy is available, an
observed label must agree with it. Unknown is never implicitly converted to
free. `validate_task_endpoints` checks a full-truth occupancy array for shape,
binary values and unoccupied point endpoints; it is **not** a path search or a
finite-radius clearance checker.

These array validators are for compact/offline data. A large compiled world
must be checked through bounded or chunked reads rather than materializing a
whole-world Boolean array only for validation.

## Next integration step

Issue #412 can verify `plant01` source files, parse their map and scenario
formats, compile the occupancy through the existing `.npy` world-pack path,
create these sidecars and run an independently replayed point-path baseline.
This foundation intentionally ships no third-party archives, converted
corpora, scenario parser, simulator dependency, training code or route score.
Its tests generate small boxes, an occluded volume and a tunnel in memory.
