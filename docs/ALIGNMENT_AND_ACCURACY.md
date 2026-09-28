# Alignment and accuracy safeguards

Updated 28 September 2026. No measured distance, GPS, RTK or ground-control reference is available for the current video. Outputs remain in arbitrary reconstruction units. Absolute dimensions and geographic orientation are not verified.

## Connected stage coordinates

1. COLMAP estimates intrinsics, camera poses and sparse geometry; MVS uses that camera model.
2. Open3D fuses depth in the same coordinate system. Depth-only scale changes are rejected: multiplying depth without changing camera translations makes overlapping surfaces disagree.
3. Open3D saves the exact raw-to-world rigid transform and a SHA-256 fingerprint of the calibrated sparse-model files in its manifest. The fingerprint survives a folder move.
4. Gaussian preparation verifies both the source video and reconstruction fingerprint. Camera extrinsics are transformed using the inverse of the mesh transform, preserving image projections. Missing legacy transform files now cause an error rather than silently substituting identity.
5. The server selects the dense run that matches the mesh, not merely the most recent run of the same video. Semantic completion verifies the exported camera dataset against the same transform and fingerprint.

Legacy outputs without enough provenance must be regenerated. Do not edit their metadata to bypass these checks.

## Surface and texture safeguards

- Alignment transforms must be finite proper rotations: no shear, reflection or hidden scale.
- Surface cleanup fails if it leaves an empty or non-finite mesh.
- Flat-field deformation and procedural solid objects are disabled by default in the optional AI preview, including requests from the interface.
- Planar semantic completion requires at least 50% support for its fitted ground plane. If it fails, retain the observed Open3D surface. Even a passing plane fit does not establish that an entire mountainous scene is flat.
- Photographic triangle textures require compatible depths at all three vertices, in addition to the existing centre and sky checks. Inconsistent triangles retain interpolated color instead of accepting a visibly incorrect projection.
- Detail-preserving settings remain available; smaller voxels cannot recover detail absent from the original stereo depths.

## Measurement safeguards

GPS fitting rejects non-finite or degenerate point sets, mismatched array shapes, duplicate image references and missing altitude. Mesh-only georeferencing no longer requires a Gaussian file. Fitting GPS positions is explicitly marked as unvalidated geometry: residuals on fitted positions are not independent accuracy measurements.

For verified metres, collect a measured scale reference or surveyed controls, then reserve independent checkpoints for testing. Report distance error, horizontal/vertical error and surface completeness against those references. GPS alone does not establish roof, vehicle or tree accuracy.

## Validation scope

Regression tests exercise projection and distance preservation after a rigid transform, rejection of scale/shear/reflection, mismatched videos and camera solutions, relocated artifacts, missing transforms, invalid metric references, and server selection of the matching dense run. These tests verify implementation behavior; they do not establish improved real-video geometry. A rebuilt scene and independent measurements are still required for that claim.

## Executed validation — 29 September 2026

- Full local suite: **55 passed, no skipped tests**; one upstream FastAPI/httpx deprecation warning.
- Real Open3D box export preserved 2 × 3 × 5 reconstruction-unit extents through PLY, GLB and viewer JSON.
- Synthetic calibrated reconstruction registered all 32 views. Alignment used 16 camera centres; the other 16 were held out. Held-out camera RMSE: **0.0019213 synthetic scene units**; maximum: **0.0039056**. These are ideal synthetic camera results, not metres or drone surface accuracy.
- Local evidence: `outputs/synthetic/synthetic_validation.json`, `metrics.json`, and `data/alignment-validation/ground_truth.json`. An initial run in `outputs/alignment-validation` was interrupted after an incorrectly entered principal point was noticed; its output is not used for validation. The successful run uses the generator's actual intrinsics: 700,400,300,0.
- Open3D 0.20.0, Torch 2.14.0+cpu and Transformers 5.17.0 are installed in the combined `.venv`. PyTorch CUDA is unavailable in this environment; the GPU reconstruction/training environments and model weights still require restoration.
- No updated real-video mesh was produced by this validation.
