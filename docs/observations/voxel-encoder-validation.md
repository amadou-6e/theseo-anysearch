# Garden 3D voxel encoder validation

This validation determines whether the existing pretrained Garden residual 3D
encoders preserve useful voxel geometry before they are integrated into RLlib.
The historical checkpoint metadata only reports reconstruction loss, which can
be misleading for sparse occupancy grids.

## Protocol

The reproducible validation command is:

```powershell
python scripts/validate_voxel_encoder.py --preset r2 --epochs 30
python scripts/validate_voxel_encoder.py --preset r3 --epochs 30
python scripts/validate_voxel_encoder.py --model-dir runtime/garden/models/voxel_resnet_b0_tuned/gpu-v3 --epochs 30
python scripts/validate_voxel_encoder.py --model-dir runtime/garden/models/voxel_resnet_b0_spatial/gpu-v4 --epochs 30
python scripts/validate_voxel_encoder.py --model-dir runtime/garden/models/voxel_resnet_b0_spatial/gpu-v6 --epochs 30
```

The encoder is frozen. A small decoder is trained on its latent vectors, then
evaluated on held-out axis-aligned boxes and out-of-distribution spheres and
planes. The same protocol runs against an identically shaped random encoder.
Linear probes measure occupied-volume retention and the six center-face
occupancy values relevant to immediate movement. A one-voxel perturbation tests
whether the latent reacts to small geometric changes.

The dataset contains 2,400 training, 600 held-out, and 600 out-of-distribution
grids. Seeds, generation rules, thresholds, and decoder training are fixed in
the script.

## Results

| Metric | Pretrained r2 | Random r2 | Pretrained r3 | Random r3 |
|---|---:|---:|---:|---:|
| Grid shape | `5³` | `5³` | `7³` | `7³` |
| Parameters | 426,880 | 426,880 | 886,080 | 886,080 |
| Latent dimensions | 64 | 64 | 128 | 128 |
| Held-out reconstruction IoU | **0.869** | 0.229 | **0.735** | 0.205 |
| Held-out occupied recall | **0.976** | 0.719 | **0.958** | 0.681 |
| OOD reconstruction IoU | **0.745** | 0.328 | **0.689** | 0.279 |
| Fill-fraction linear-probe R² | **0.990** | 0.935 | **0.996** | 0.976 |
| Center-face probe accuracy | **99.58%** | 91.53% | **99.58%** | 93.78% |
| Relative latent change from one voxel | **0.296** | 0.004 | **0.263** | 0.006 |

### Tuned B0 pilot

The first GPU search ran eight B0 optimization trials for up to 20 epochs on
the same cached 50,000-sample random-box dataset. ASHA pruned four trials at
epoch 4. The selected configuration used Adam, batch size 128, learning rate
`7.388e-4`, and weight decay `5.916e-5`. Clean retraining reached validation
loss `3.395e-5` with the corrected `positive_bce_v2` objective.

| Metric | Tuned B0 | Random B0 | Pretrained r2 |
|---|---:|---:|---:|
| Parameters | **327,576** | 327,576 | 426,880 |
| Held-out reconstruction IoU | 0.761 | 0.385 | **0.869** |
| Held-out occupied recall | 0.955 | 0.793 | **0.976** |
| OOD reconstruction IoU | 0.683 | 0.440 | **0.745** |
| Fill-fraction linear-probe R² | 0.988 | 0.615 | **0.990** |
| Center-face probe accuracy | 98.89% | 92.83% | **99.58%** |
| Relative latent change from one voxel | 0.253 | 0.252 | **0.296** |

The B0 encoder learned substantially better features than its random control,
but minimizing the autoencoder validation objective did not produce the best
frozen latent representation. Its much smaller training loss therefore must
not be interpreted as beating `r2:v1`. The next search should rank checkpoints
by representation metrics or a combined validation score rather than raw focal
reconstruction loss alone.

### Spatial B0 representation tuning

The follow-up search preserves the final `2x2x2` spatial feature map, combines
positive BCE with Dice and center-face auxiliary losses, and selects trials by
a frozen linear-probe score containing both held-out and deterministic OOD
geometry. Eight GPU trials selected Adam with batch size 128, learning rate
`2.874e-4`, weight decay `2.435e-4`, Dice weight `0.25`, and face-loss weight
`0.25`.

| Metric | Spatial B0 | Random spatial B0 | Pretrained r2 |
|---|---:|---:|---:|
| Parameters | **356,248** | 356,248 | 426,880 |
| Held-out reconstruction IoU | **0.933** | 0.508 | 0.869 |
| Held-out occupied recall | **0.990** | 0.876 | 0.976 |
| OOD reconstruction IoU | **0.777** | 0.530 | 0.745 |
| OOD occupied recall | 0.936 | 0.781 | **0.976** |
| Fill-fraction linear-probe R² | 0.977 | 0.606 | **0.990** |
| Center-face probe accuracy | **100.00%** | 92.94% | 99.58% |
| Relative latent change from one voxel | **0.634** | 0.433 | 0.296 |

Spatial B0 clears the representation gate and improves on `r2:v1` by 0.064
held-out IoU and 0.032 OOD IoU with 17% fewer encoder parameters. The result
also validates the representation-oriented search metric: it selected a model
that generalizes better even though raw reconstruction loss was no longer the
optimization criterion.

### Compact S0 search

The 100-trial width, depth, and latent search selected the architecture now
named `voxel_resnet_s0`: channels `[20, 40, 56]`, one block per stage, spatial
`2x2x2` pooling, and a 48-dimensional latent. At 257,188 parameters it is 28%
smaller than spatial B0 and 40% smaller than `r2:v1`.

| Metric | S0 clean retrain | S0 trial epoch 14 | Spatial B0 |
|---|---:|---:|---:|
| Parameters | **257,188** | **257,188** | 356,248 |
| Held-out reconstruction IoU | 0.802 | **0.843** | **0.933** |
| OOD reconstruction IoU | 0.661 | **0.696** | **0.777** |
| Fill-fraction linear-probe R² | **0.969** | 0.966 | **0.977** |
| Center-face probe accuracy | 99.81% | **100.00%** | **100.00%** |
| Relative latent change from one voxel | 0.598 | **0.600** | **0.634** |

S0 is adopted as the smallest named Garden model, but not as the high-fidelity
default. Its strong in-search probe score did not transfer fully to the
independent reconstruction protocol. This also shows that clean retraining must
select checkpoints by the representation metric rather than reconstruction
validation loss alone.

Single-threaded warm inference on the validation machine measured approximately
2.4 ms for one r2 grid and 2.8 ms for one r3 grid on CPU. Warm GPU batches of
128 took approximately 3.0 ms for r2 and 2.2 ms for r3, excluding host/device
transfer. Single-grid GPU latency remained around 2 ms because kernel-launch
overhead dominates at this scale.

Latency is hardware- and runtime-dependent and must be measured again inside an
RLlib EnvRunner/Learner pipeline. Weight values do not alter the architecture's
operation count, so timing differences between pretrained and random instances
are treated as benchmark noise.

## Decision

Pretraining is effective and necessary. Both checkpoints retain substantially
more occupied structure than a random encoder and respond strongly to a
single-voxel change. The movement-relevant face probe is nearly perfect.

The spatial B0 checkpoint remains the strongest 5x5x5 representation and should
be the primary frozen-policy candidate. Its 0.933 held-out IoU and 0.777 OOD IoU
surpass `r2:v1`, while using fewer parameters. Keep `r2:v1` as the established
baseline for policy ablations.

The r3 checkpoint is useful but does not yet meet a strict faithful-reconstruction
standard: held-out IoU is 0.735. It should remain experimental until it is
pretrained on more task-representative navigation and construction patches or
its reconstruction IoU improves.

RLlib integration should therefore compare frozen spatial B0 with `r2:v1`, keep the raw
`face_neighbors` or action mask available outside the latent for safety-critical
local feasibility, and compare frozen, gradually unfrozen, and randomly
initialized encoders. End-to-end success and sample efficiency remain separate
gates; this validation establishes representation fidelity, not policy quality.

The original globally pooled B0 pilot does not replace `r2:v1`; spatial pooling
and representation-aligned selection are the changes that close and reverse
that gap. Policy learning remains the next required validation gate.

## Limitations

- Synthetic validation is independent but resembles the random-box pretraining
  distribution. Real imported geometry must be added before declaring broad
  generalization.
- The probe decoder is not the original discarded training decoder.
- Reconstruction does not prove that a policy can exploit the latent.
- GPU timings exclude observation transfer and distributed synchronization.
