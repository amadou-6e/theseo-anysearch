# Compact classification controls assessment

Issue #381; source3987482, authorized unmerged stack on #379. Spec
amadou-6e/specs@97e205793e6e443199f118b528c0c3fc9181e76c,
projects/theseo-anysearch/python/perception-encoder-compact-controls.md.
Run compact-controls-v1-run1. Disposition retain diagnostic and controls.
No encoder updates, HPO selection, certification, promotion or integration merge.

## Result: threshold effects exist, but are not the main limitation

All60 probe fits completed on fresh48 probe/24 selection/48 development parents.
The six compact encoders are the96-update128-dimensional smoke checkpoints, not
fully trained candidates. Same queries for all controls. Thresholds selected on
selection F1 only; below are plain1024 results, not best-over-recipes selection.

|Representation|Occupied IoU selected|Occupied AUPRC|Boundary F1 selected|Boundary AUPRC|
|---|---:|---:|---:|---:|
|Attention frozen|.225|.264|.320|.246|
|Attention joint|.230|.254|.322|.243|
|Grid frozen|.190|.233|.307|.223|
|Grid joint|.198|.248|.330|.233|
|Strided frozen|.205|.247|.305|.232|
|Strided joint|.220|.255|.311|.234|
|Random grid|.201|.245|.289|.212|
|Pretrained spatial|.690|.911|.777|.843|
|Raw3-cubed neighborhood|.633|.866|.638|.637|
|Coordinates only|.177|.185|.257|.148|

Development prevalence is.177 occupancy and.147 boundary. Many compact fixed.5
scores are zero while selected thresholds give nonzero scores: the earlier zero
values did not prove complete absence of signal. However their AUPRC remains low
and close to the random compact control, whereas the spatial and raw controls
recover much stronger signal. Moving128->1024 probe updates does not close this
gap. Balanced BCE improves fixed.5 decisions but yields little ranking improvement;
for compact candidates it worsens development log loss to about.630-.686 versus
.398-.474 for plain1024. Weighted logits are not calibrated probabilities.

Training AUPRC for compact plain1024 remains about.254-.282, unlike spatial
.851-.914: this is not simply good training performance failing to generalize.
Raw-neighborhood boundary AUPRC drops from.906 train to.637 development, exposing
a separate overfitting issue in that control. Do not infer causes solely from a
single score. Full probabilities, curves, family breakdowns and all recipes are
retained in the report/artifacts.

## Direction and limits

Do not spend the full search budget optimizing thresholds of these checkpoints.
Investigate compact training and decoder learnability first. A small coordinate
MLP must infer location-dependent geometry from a global vector; a spatial head
receives location-aligned features directly. Equal hidden width is not equivalent
task difficulty, and low compact scores do not prove128 dimensions are impossible.

Next bounded foundation experiment: demonstrate tiny-corpus overfitting through
the compact vector, and test a known low-dimensional geometric code with the
same query decoder. Compare plain versus position-enriched coordinates/stronger
decoder under an explicit new protocol. This separates decoder expressiveness
from representation training before broad HPO. Preserve independent frozen-probe
evaluation afterward; a stronger training decoder must not bypass the vector.

This diagnostic uses one checkpoint seed, two synthetic families, central17
queries, fixed masking and only classification. No uncertainty or HPO-winner
claim. Random grid is not matched to every aggregation. Control input dimensions
and head parameter counts differ: compact131/4257, spatial11/417, raw57/1889,
coordinates3/161. Raw control is a local neighborhood, not a full-grid baseline.
Distance quality, richer corruptions, calibrated utility floors and actual campaign
resource allocation remain unresolved. Scaling remains paused.

## Evidence and cost

Elapsed79.032 seconds including preparation; feature extraction cached once per
representation/split.60 probe states, train normalization and predictions retained
outside Git, with hashes. Report
docs/perception-encoder-local-geometry/compact-controls-report.json;
registration compact-controls-preregistration.json in the same directory.
Payload05385b70d722dcfa75222e1f62f82b3682bdddd0a46622eba478d357da4bad86.
442 garden tests passed before adding the report replay test.23 focused tests
passed before execution. Data disjointness checked against smoke/context/tiling
parents. No final assessment data accessed or48 GPU-hour search launched.
