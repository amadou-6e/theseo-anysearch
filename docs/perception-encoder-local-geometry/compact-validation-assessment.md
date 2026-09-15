# Frozen compact encoder: robustness and collision transfer

Issue405 / PR406; specs78 / PR79. User-authorized stack on package403/PR404.
Disposition: retain experimental evidence; no promotion or merging authorized.
Fixed input33, central17, frozen729 spatial code; no encoder retraining.

## Protocol and evidence

- Frozen specification: `191f883a24b67afe75a5ce9b0783f53351802423` in
  `amadou-6e/specs`, `perception-encoder-compact-robustness-transfer.md`.
- Frozen implementation: `e16fbc6` on `exp/405`; manifests committed as `a40a7ec`.
- Package manifest: `2ddf188ecedff4699dba434fec05a851c5649119c07dcf30c9d9afd1292d4ed1`.
- Robustness report: `6f3036bd4f6a83443324ca44a7ac9c234b82748a31b89c71be4052349de09b7e`.
- Collision report: `3c058b813540b08c7eb8f7499a13c2c6b08b44549689a2fc1d9f2a76ec7f8d67`.
- Robustness runtime artifacts: `runtime/compact-robustness-v1-run1`.
- Transfer runtime artifacts: `runtime/compact-collision-transfer-v1-run1`.

Both datasets/protocols were frozen before either study executed. The same three
packaged seeds401/402/403 are evaluated; none is selected using these results.
Weights/corpora/prediction tensors are runtime-only. Compact JSON reports and replay
evidence accompany this completed assessment.

## Robustness result

Completed2016 paired observations across seven generators and twelve conditions,
with three frozen encoders/native geometry heads. All252 prediction groups and
artifacts replayed on the original CUDA path; all encoder/head state hashes match.

All four original generators pass all four original targets at baseline for all
three seeds. Among unseen generators, capsule forests pass baseline; ellipsoids
miss boundary F1 for seed401 (.6954 vs .70). Gyroid walls miss hidden occupancy IoU
for all seeds (.4513-.5363 vs .60), with boundary failures for401/403 as well.

| Condition | Seed/family groups failing any available original target (out of21) |
| --- | ---: |
| Baseline (.16 density, .20 unknown, no noise) | 4 |
| Density .03 | 21 |
| Density .08 | 18 |
| Density .28 | 6 |
| Density .45 | 18 |
| Missing .10 | 3 |
| Missing .40 | 11 |
| Missing .60 | 19 |
| Visible bit-flip noise .01 | 19 |
| Visible bit-flip noise .05 | 21 |
| Seven-slice contiguous unknown slab | 21 |

Zero missing cells leave hidden occupancy/boundary/recovery metrics unavailable
for all21 groups. Available clearance passes, but this is **not** a full-target pass.

The slab is the strongest observed failure: hidden IoU .0379-.1494 and hidden
boundary F1 .0602-.2198. At5% noise, hidden F1 .4366-.6895 misses the unchanged .70
bar in every group. At density .03, hidden IoU .0278-.5505 misses .60 throughout.
Good occupancy alone is insufficient: dense solids can retain high IoU while their
boundary F1 deteriorates. These results do not support broad distributional robustness.

The report includes support counts, per-geometry scores, all-cell diagnostics, and
231 paired mean negative-distance-error comparisons with1000 geometry bootstrap
resamples. Conditions share scene fields; density shifts change truth and are
paired-field comparisons, not repeated measurements of identical occupancy.
There are24 independent scene fields per family, not2016 independent parents.

Runtime accounting1321.922s includes600s validation reserve, freeze preparation,
and a conservative interrupted-phase charge. One Windows file-lock failure during
atomic progress replacement resumed without source/protocol changes. Original
device verification took24.735s, within the reserved validation allowance.

## Collision transfer result

Completed all12 heads,4096 updates each: compact/raw/spatial/null for seeds401/402/403.
All twelve calibration thresholds locked before ID/OOD test predictions. Fixed-head
shuffled/zeroed controls did not refit or recalibrate. Encoder and normalization
remained frozen. Fresh data:1536 train,384 calibration,384 ID test,288 OOD geometries,
32 paths each. Each head sampled524288 path examples with replacement, not that many
distinct paths. The report contains36 test evaluations,12 calibration records,
24 stratified paired comparisons, and164 hashed runtime artifacts.

| Input to new head | Parameters | ID test AUPRC, range over3 seeds | OOD AUPRC, range over3 seeds |
| --- | ---: | ---: | ---: |
| Frozen729 compact code | 4521 | .9010-.9055 | .9000-.9089 |
| Raw observed occupancy + unknown | 4737 | .9929-.9946 | .9917-.9942 |
| Frozen full spatial features | 6033 | .9888-.9897 | .9843-.9875 |
| Trained zero-input compact control | 4521 | .3359-.3363 | .2640-.2674 |
| Shuffled code, unchanged compact head | 4521 | .2367-.2498 | .1867-.1904 |

**Useful transfer, but not competitive under the frozen screen.** Every compact
seed exceeds the null margin and passes the overall false-safe/false-alarm bars,
but every seed fails the .03 AUPRC margin against both raw and spatial controls.
There are six failed comparisons, not a selected passing seed. Zeroing the trained
compact head's code produces97.9-100% missed ID collisions; it is not equivalent
to training a null-input head and is not evidence of safety.

Primary per-geometry negative-log-loss comparisons agree: all95% bootstrap intervals
are below zero against raw/spatial, and above zero against null/shuffled, for every
seed and both test splits. For example, seed401 ID compact-minus-raw is -.12554
[-.13731,-.11382]; compact-minus-spatial is -.11090 [-.12260,-.09968]. These are
paired geometry-cluster results with1000 resamples, not independent path samples.

At the frozen calibration thresholds, compact ID missed-collision rates are
4.07-5.22%, with12.93-13.68% false alarms. Raw heads have4.85-5.05% missed collisions
but only .39-.65% false alarms. This overall view does not describe hidden-space
safety: see the explicitly preregistered uncertain-path stratum below.

## Hidden-space warning

Uncertain means at least one unknown path voxel and no visible occupied path voxel.
The ID subset has4796 paths including257 collisions; OOD has3772 including153.

| Input | Uncertain ID AUPRC | Uncertain ID missed collisions | Uncertain ID false alarms |
| --- | ---: | ---: | ---: |
| Compact | .4434-.4895 | 8.17-13.62% | 14.74-15.53% |
| Raw | .6888-.7071 | 45.91-48.25% | .84-1.34% |
| Spatial | .7472-.7539 | 31.52-37.35% | 1.96-2.16% |

The raw and spatial heads rank uncertain cases better than compact, but the single
overall95%-recall calibration threshold permits many missed hidden collisions.
Compact's lower missed-collision rate here comes with many more false alarms; it
does not overturn the ranking/log-loss result. OOD repeats the warning: uncertain
missed-collision rates are8.50-15.69% compact,46.41-49.67% raw,35.29-35.95% spatial.
Do not recalibrate those thresholds on these now-used test sets.

The visible-grid rule answers60.97% of ID and59.07% of OOD paths with zero errors on
its definite subset, and abstains on the rest. This is not100% collision coverage.
Observed free/occupied checks must remain distinguishable from inferred hidden
occupancy when considering any later integration.

Additional compact weaknesses: ID oblique-sheet AUPRC .8337-.8450 with31.4-32.4%
false alarms; length2 ID AUPRC .6876-.6976 with10.8-14.4% missed collisions. OOD
gyroid AUPRC .8548-.8670 trails capsules .9605-.9628. These are reported strata,
not retrospectively added selection rules.

## Evidence-directed next direction

1. Keep the small33/central17 field and the frozen spatial backbone. Its uncertain
   ID ranking is encouraging; this is a diagnostic, not a replacement primary gate.
2. Test the compact readout before blaming the encoder: compare the current
   trilinear query head with learned spatial upsampling and an existing native
   geometry-decoder-derived collision control. Match head budgets and then test
   whether compact-code training itself needs a collision-aware objective.
3. In a separately frozen followup, calibrate/evaluate uncertain-path operating
   points explicitly, rather than relying on pooled95% collision recall. Preserve
   visible-grid definite checks and a declared abstention policy.
4. For a future retraining study, include density variation, visible noise, and
   contiguous missing regions. Keep its validation/test identities separate from
   this no-retraining robustness screen. Do not scale the field to avoid these gaps.

These are directions, not experiments already executed or changes to this protocol.
One lightweight head family was tested without HPO. Different first-layer parameter
counts, receptive fields and upsampling paths remain limitations; this experiment
does not prove an information-theoretic limit of729 numbers. Three seeds share one
fresh synthetic corpus, and OOD generators retain the same quantile-density mechanism.
No RLlib, navigation, global topology, finite-body collision, or real-sensor claim.

## Validation and resources

Local garden suite559 passed; published implementation CI was green at3959df2.
Transfer replay passed:164 artifact hashes, all12 fitted heads,48 calibration/test
prediction groups,486 original-device encoder batches,82944 resampled queries,
calibration locks, all metrics/paired comparisons and unchanged encoder states.
The shared spatial caches were checked against identical packaged backbones.
Verification took83.5s, within the900s reserve. Robustness replay passed all252 groups.
Compact fits took29.1-30.1s; raw167.0-168.8s; spatial235.4-244.1s; null28.4-29.4s.
Maximum recorded CUDA allocation was2,132,763,136 bytes; observed GPU utilization
was86-93%. Fixed data caches and GPU-preloaded training representations were used.

Transfer accounted2384.922s including900s validation reserve and preparation, within
its14400s tranche. Both requested studies together account3706.844s (1.030h);
the campaign ledger totals11.38357 of48 authorized hours, leaving36.61643h. No
additional budget, merges, promotion, or encoder retraining was performed.
