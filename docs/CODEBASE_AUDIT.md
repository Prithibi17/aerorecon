# AeroRecon codebase audit

Audit date: 2026-09-12  
Baseline audited: `8cfb98b`  
Scope: every first-party source, test, script, dependency manifest, launcher, and document in the repository. Vendored Three.js and OrbitControls were treated as upstream dependencies rather than application code.

## Executive finding

AeroRecon is a working local research prototype with COLMAP structure from motion and dense multi-view stereo, semantic sky masks, Open3D TSDF mesh refinement, gsplat/PyTorch 3D Gaussian training, held-out image validation, browser viewing, and optional GPS/IMU CSV georeferencing. It is not yet a survey-grade mapping system. Camera geometry and MVS depth are evidence-based, while weakly observed surfaces remain incomplete. The current example video has no telemetry, so its output still has relative scale. The system also lacks GCP/checkpoint accuracy reporting, a multi-flight project model, and defensible cross-flight merge logic.

The safest development path is to make the classical photogrammetry path reliable and observable first, then add scale/georeferencing, dense MVS, cleanup, semantic assistance, and multi-flight alignment. AI depth should fill weak areas and aid masking; it should not define camera geometry or override consistent photogrammetry.

## Repository map

| Area | Current responsibility | Assessment |
|---|---|---|
| `pipeline/cli.py` | CLI orchestration and run manifest | Real and useful; was too monolithic before Phase 1 extraction. |
| `pipeline/video_analysis.py` | Content-aware video analysis and keyframe selection | Added in Phase 1; real image measurements, configurable thresholds, diagnostic artifacts. |
| `pipeline/sfm.py` | Camera configuration, COLMAP SIFT/matching/mapping, sparse exports | Added in Phase 1 by extracting and strengthening existing real COLMAP work. |
| `pipeline/progress.py` | Atomic, machine-readable stage progress | Added in Phase 1; replaces server log-string guessing for new sparse runs. |
| `pipeline/server.py` | Loopback API, upload validation, worker control, artifact serving | Working. Sensible local-origin checks and artifact allowlist. Process registry remains memory-local. |
| `pipeline/pose_tracking.py` | Optical feature tracks and PnP poses for non-keyframes | Real estimation anchored to COLMAP. Fallback interpolation is explicitly counted. |
| `pipeline/semantic_fusion.py` | MoGe depth, SegFormer labels, depth scaling, field leveling, TSDF, object/texture hooks | Functional experimental pipeline. Geometry is not metric and several priors can dominate weak evidence. |
| `pipeline/ai_reconstruct.py` | Earlier AI depth reconstruction path | Experimental/legacy relative to semantic fusion. Keep for comparison until dense MVS replaces it. |
| `pipeline/video_fusion.py` | Earlier full-video fusion path | Experimental/legacy. Duplicates responsibilities now present in semantic fusion. |
| `pipeline/object_models.py` | Turns semantic clusters into closed house/tree solids | Procedural approximation, not reconstruction of actual roofs, walls, branches, or trunks. |
| `pipeline/texture_atlas.py` | Bakes top-down photographic color evidence | Useful preview. Not a calibrated multi-view texture unwrap; vertical and hidden surfaces are inferred. |
| `pipeline/train_aerial_corrector.py` | Trains a small depth residual corrector | Real training code, but current training/validation evidence is synthetic-domain only. |
| `apps/web/` | Upload, run list, progress, Three.js/Gaussian viewer, layers, exports | Working interface. It can view and launch Gaussian training, and exports a GPS/IMU template. Project/session, map, GCP, and multi-flight controls remain absent. |
| `scripts/` | Synthetic generation/checks, calibration audit, model download, run naming | Useful developer utilities. Synthetic validation does not establish real-world accuracy. |
| `tests/` | Geometry math, tracking, API controls, extraction, object helpers | Good regression base. Optional AI tests depend on the AI environment/models. No real aerial benchmark suite. |

## What already works

1. Video files are validated and decoded locally with PyAV.
2. Run directories retain manifests, logs, configurations, warnings, errors, and exports.
3. COLMAP/PyCOLMAP performs actual SIFT extraction, geometric matching, incremental mapping, triangulation, and bundle adjustment.
4. Sparse camera centers and colored sparse points are exported and rendered in the browser.
5. Camera intrinsics can be supplied and frozen. Phase 1 validates supported models and parameter counts and records whether intrinsics came from the user or COLMAP.
6. Per-frame optical tracking and PnP extend registered keyframe poses through the source video, with explicit tracked/fallback counts.
7. MoGe, semantic segmentation, sparse-depth alignment, correction gating, field-plane fitting, and Open3D TSDF code execute as a coherent experimental dense path.
8. The server is loopback-only, checks origins for mutations, limits upload type/size, and exposes only allowlisted artifacts.
9. The UI supports source video, keyframes, point cloud/surface/camera toggles, color modes, logs, and downloads.
10. Automated tests cover core transformations, server controls, upload behavior, and failure evidence.

## What was incomplete or misleading

### Video preparation before Phase 1

Selection used a fixed sampling rate plus blur, clipped-pixel, and ORB-count gates. It did not evaluate visual redundancy or measured inter-frame motion. The frame cap could therefore spend the reconstruction budget on nearly identical images and stop before the end of a long flight.

Phase 1 adds similarity, verified ORB displacement, temporal-gap retention, a contact sheet, per-candidate CSV metrics, and JSON selection statistics. The current implementation still uses thresholds rather than a globally optimized coverage solver, and a hard frame cap can still truncate long videos. Segment-aware budgeting is a later improvement.

### Progress reporting before Phase 1

The server inferred stages by searching worker log text. That was brittle and could display an incorrect stage when COLMAP wording changed. New sparse runs now write `progress.json` atomically with stage index, percentage, message, counts, status, timestamp, and the complete stage list. Old runs retain the log fallback for compatibility.

### Dense geometry

There is no classical dense stereo stage. Current dense geometry comes primarily from per-frame monocular depth fused with tracked cameras. Sparse COLMAP points constrain scale locally, but they do not turn monocular predictions into independently accurate surfaces. Weak poses, textureless fields, foliage, motion, exposure shifts, or depth-domain mismatch can create ridges, sheets, arcs, and warped terrain.

### Ground orientation and dimensions

Field leveling estimates planes from semantic ground and applies a flat-field prior. This can suppress the severe bending seen in earlier outputs, but it is an assumption and can erase real terrain slope. Coordinates remain relative. Without telemetry, GCPs, RTK/PPK, or known distances, the system cannot claim metres, altitude, north/east/up, or geographic position.

### Houses and trees

The object module detects semantic clusters and generates bounded closed solids. These are visualization proxies. A box/roof generated from a building cluster is not a recovered building mesh; a generated trunk/crown is not a reconstructed tree. Labels such as “object models” in the UI describe the output form, not geometric fidelity.

### Texture

The atlas transfers observed colors and creates a useful photographic preview. It does not yet perform calibrated visibility testing, per-face best-view selection, occlusion-aware projection, seam optimization, exposure compensation, or UV chart packing. Top-facing field areas receive the strongest evidence. Walls, roof sides, and hidden regions can receive stretched, repeated, or inferred color.

### AI training and model claims

MoGe-2 and SegFormer are pretrained models; the repository does not train either model on the user's site. The aerial residual corrector has a real training script and acceptance gate, but the documented run was trained on synthetic aerial data. That is insufficient evidence for real farms and buildings. SegFormer ADE20K labels are general-scene labels rather than aerial mapping classes. Model licenses and dataset rights must be reviewed before distribution or commercial training.

### Multi-flight reconstruction

There is no first-class project/session database, session transform, global reconstruction graph, cross-session verified matching, global bundle adjustment, semantic identity store, incremental project update, or “Add another flight” workflow. Existing runs are independent folders. Combining their vertices would be an unjustified merge and is prohibited by the specification.

## Architectural risks and bottlenecks

- `semantic_fusion.py` contains model loading, pose resolution, segmentation, depth correction, leveling, fusion, meshing, object generation, texturing, metrics, and manifest updates in one long worker. It should be split only as those phases are replaced and covered by tests.
- Job state is stored in files plus an in-memory subprocess dictionary. Restart recovery can detect dead PIDs, but there is no durable queue, cancellation protocol, resource scheduler, or resumable stage cache.
- Classical feature extraction is CPU-only and PyCOLMAP currently selects many CPU threads internally. Large keyframes can consume substantial RAM.
- Sequential matching is efficient for one continuous flight but misses loop closures and cross-flight overlap. Vocabulary-tree/retrieval and spatial prior matching are needed later.
- The single-camera assumption is valid only when a video keeps one unchanged camera/lens mode. Digital zoom, stabilization crops, lens changes, or mixed inputs violate it.
- AI workers assume a configured NVIDIA/CUDA environment and local model paths. Capability checks are coarse.
- Output schemas are versioned only at the top-level run manifest. Artifacts need explicit schemas and migrations before project/session persistence.
- Viewer payloads are JSON and browser memory becomes a bottleneck for large point clouds/meshes. Tiled or binary streaming will be needed.

## Components to preserve

- Immutable run directories and failure evidence.
- COLMAP as the camera-geometry source of truth.
- Explicit warnings for arbitrary scale, suspect calibration, partial registration, and disconnected models.
- Per-frame pose tracking only when anchored to registered geometry, with fallback counts visible.
- Sparse-evidence gates that accept AI corrections only when held-out geometry improves.
- Local-only upload and origin controls.
- Existing viewer/layer controls and artifact allowlist.
- Synthetic scene generator as a deterministic smoke test, while clearly separating it from accuracy validation.

## Components to replace or demote

- Demote monocular depth from the primary dense reconstruction method to a confidence-gated hole-filling or initialization aid after dense MVS exists.
- Replace procedural house/tree solids with observed dense geometry plus semantic labels. Keep procedural proxies only as an explicitly optional visualization mode.
- Replace the top-down atlas with calibrated multi-view texture projection and seam handling.
- Replace flat-field enforcement with a selectable terrain prior backed by telemetry/GCP/plane evidence and confidence.
- Replace log parsing with structured state throughout every worker, including the AI worker.
- Consolidate `ai_reconstruct.py`, `video_fusion.py`, and `semantic_fusion.py` after the replacement stages have parity and tests.

## Dependency and license audit

| Dependency/model | Role | Current concern |
|---|---|---|
| PyCOLMAP/COLMAP | Sparse SfM, camera calibration, and CUDA dense MVS | Correct geometric foundation; output quality still depends on overlap, texture, and camera calibration. |
| OpenCV | Video/image metrics, features, PnP, geometry | Appropriate; algorithm thresholds need aerial validation. |
| PyAV/FFmpeg | Video decoding | Appropriate; metadata/telemetry parsing remains minimal. |
| Open3D | TSDF and mesh operations | Appropriate for prototypes; memory and scale limits need profiling. |
| MoGe-2 | Monocular geometry prior | Useful assistive evidence; not metric truth. Review model terms for the intended deployment. |
| SegFormer ADE20K | Semantic masks | Domain mismatch for nadir/oblique aerial imagery; model card restrictions require review. |
| Transformers/PyTorch | AI inference/training | CUDA/environment size and reproducibility need tighter checks. |
| Trimesh | Mesh export/manipulation | Appropriate. |
| Three.js/OrbitControls | Browser rendering | Appropriate; upstream license files are retained. |
| FastAPI/Uvicorn | Local API | Appropriate. Production remote hosting is outside the current threat model. |

No repository license has been selected. That blocks clear downstream reuse even where individual dependencies are permissively licensed.

## Implementation order

### Phase 1 — reliable sparse reconstruction and observability

Status: implemented in this change.

- Content-aware candidate analysis and keyframe selection.
- Blur, exposure, feature, similarity, motion, and temporal-gap evidence.
- Configurable camera models and validated intrinsics.
- COLMAP SIFT, temporal matching, mapping, bundle adjustment, sparse export, and diagnostics in a dedicated module.
- Structured stage/progress state consumed by the existing API and UI.
- Contact sheet, video metadata, selection metrics, camera configuration, metrics, report, and failure evidence.
- Automated tests plus an end-to-end synthetic COLMAP validation.

### Phase 2 — telemetry, scale, and coordinates

- Parse SRT, CSV, subtitle telemetry, container metadata, and supported drone logs.
- Normalize timestamp/frame/latitude/longitude/altitude/yaw/pitch/roll and record source/confidence.
- Estimate a robust similarity transform from SfM to telemetry or GCP coordinates.
- Introduce explicit scale states: relative, approximate metric, metric, georeferenced.
- Add CRS metadata and ENU/projected-coordinate exports. Never claim precision beyond the evidence.

### Phase 3 — dense photogrammetry

- Add undistortion and classical dense MVS through COLMAP dense stereo or OpenMVS.
- Add depth-map filtering, fusion, normals, confidence, and visibility-consistency diagnostics.
- Produce dense point clouds before meshing. Use AI depth only when photogrammetric confidence is weak.

### Phase 4 — cleanup, mesh, and calibrated texture

- Statistical/radius outlier removal, connected-component filtering, normal consistency, and conservative hole handling.
- Poisson or Delaunay/ball-pivot mesh options chosen by scene type and density.
- Occlusion-aware multi-view texture projection, exposure compensation, seam optimization, UV atlas, and orthomosaic/DSM/DTM paths.

### Phase 5 — aerial semantics and dynamics

- Fine-tune an aerial segmentation model on licensed classes such as field, road, roof, tree, water, vehicle, person, and sky/background.
- Mask moving vehicles, people, propellers, shadows when harmful, and unstable vegetation regions before matching/fusion.
- Fuse labels in 3D using visibility, confidence, reprojection quality, and repeated observations. Keep geometry and semantics as separate data.

### Phase 6 — validation and confidence

- Add held-out image reprojection, completeness, density, track length, view count, depth consistency, telemetry residual, checkpoint/GCP error, and coverage maps.
- Build real aerial benchmark projects with independent measurements and train/validation/test site separation.
- Generate machine-readable and human-readable reports that distinguish measured, estimated, inferred, and missing values.

### Phase 7 — multi-flight project model

- Add project and session records with immutable original frames, intrinsics, timestamps, telemetry, poses, local geometry, and per-session provenance.
- Generate session retrieval descriptors and candidate cross-flight pairs; verify them geometrically with inlier count, spatial distribution, transform stability, and residuals.
- Build a global reconstruction graph and initialize session transforms only from accepted visual, telemetry, GCP, or known-distance constraints.
- Run joint/global bundle adjustment rather than concatenating clouds. Preserve disconnected components when evidence is insufficient.
- Fuse duplicate geometry and semantic observations with source-session lists, timestamps, confidence, and conflict records.
- Support incremental flight addition without erasing prior evidence. Expose “Add another flight,” per-flight visibility, merge confidence, connected/disconnected state, and re-optimization status in the existing interface.

## Phase 1 validation evidence

Automated tests: 22 passed, 5 optional tests skipped in the classical environment.

End-to-end deterministic synthetic video:

- 24 selected keyframes.
- 24/24 registered cameras.
- 12,683 sparse points.
- 0.243 px mean reprojection error.
- One connected model.
- Fixed known `SIMPLE_RADIAL` intrinsics.
- Structured progress reached `Quality report`, 100%, complete.

This validation proves that the Phase 1 code path executes and recovers the designed synthetic scene. It does not prove accuracy on real drone footage.
