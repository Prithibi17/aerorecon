# AeroRecon AI

Experimental, local-first reconstruction of a 3D scene from drone video.

AeroRecon extracts camera motion from a video, predicts per-frame geometry and semantic classes, fuses the observations into a shared surface, levels agricultural ground, creates approximate house/tree solids, and exports a browser-viewable map, point cloud, textured GLB, evidence files, and quality report.

> [!WARNING]
> AeroRecon is a research prototype. Its current output is **not survey-grade, metrically calibrated, georeferenced, or independently validated**. Do not use it for engineering measurements, navigation, boundaries, construction, inspection, safety decisions, or legal land records.

## Current state

Phase 1 of the production upgrade is implemented: content-aware keyframe selection, validated camera-model configuration, modular COLMAP SfM, structured progress, and diagnostic artifacts. See [the full codebase audit](docs/CODEBASE_AUDIT.md) for an evidence-based description of what is real, approximate, missing, and planned.

The end-to-end workflow runs on the supplied 28.72-second, 1920×1080, 25 FPS field video and processes all 718 decoded frames.

| Capability | State | What it means |
| --- | --- | --- |
| Local upload and project UI | Working | Upload, process, inspect, orbit, select layers, and download artifacts locally. |
| Classical camera reconstruction | Working with assumptions | COLMAP/PyCOLMAP supplies 54 registered anchor views. Intrinsics are assumed rather than independently calibrated. |
| Per-frame camera tracking | Working experimentally | Optical flow and PnP track the remaining 664 frames; this sample used zero interpolated poses. |
| Monocular depth | Working experimentally | MoGe-2 Base predicts depth/validity for every frame. Sparse features align relative scale. |
| Semantic understanding | Working with errors | SegFormer identifies sky, field, buildings, and vegetation, but was trained for general scenes and misclassifies this aerial domain. |
| Full-video fusion | Working experimentally | Open3D TSDF integrates valid depth from every frame. |
| Flat-field orientation | Working as a scene prior | A shared ground plane fixes the false-mountain failure for this agricultural clip. It must be disabled for real hills or mountains. |
| Houses and trees | Prototype | Compact semantic clusters become closed procedural solids. Their placement is evidence-based; their shape and hidden sides are inferred. |
| Texture | Prototype | A 1024×1024 UV atlas uses registered video views. Unobserved and weakly aligned regions remain coarse or filled. |
| Metric scale | Not implemented | Coordinates use arbitrary relative units. |
| GPS/GCP/IMU alignment | Not implemented | The map has no latitude/longitude, surveyed control, gravity measurement, or north heading. |
| Real-world accuracy validation | Not completed | Internal consistency numbers are available; no LiDAR, RTK, surveyed distance, or withheld real-scene ground truth has been used. |

## Latest experimental result

The latest accepted run is `outputs/drone-photo-textured-ground-clean` (generated artifacts are ignored by Git).

| Measurement | Result |
| --- | ---: |
| Frames decoded and fused | 718 / 718 |
| Registered COLMAP anchors | 54 |
| Optical-flow/PnP tracked frames | 664 |
| Interpolated fallback poses | 0 |
| Median tracking support | 1,387.5 inlier 3D features |
| Frames with a measured local field plane | 689 |
| Frames using the shared field pose | 718 |
| Median field-plane support | 72.9% |
| Final vertices | 69,049 |
| Final triangles | 126,568 |
| Approximate house solids | 29 |
| Approximate tree solids | 32 |
| Texture atlas | 1024×1024 |
| Texture evidence | 254,428 samples from 54 registered frames |
| Median sparse-depth discrepancy | 5.83% |
| Locally refined depth frames | 606 |
| Aerial corrector accepted | 260 / 718 frames |
| Internal held-out depth discrepancy | 6.57% → 5.91% |

The depth discrepancy is measured against feature geometry derived from the same assumed-camera reconstruction. It is an internal consistency check, **not an independent accuracy measurement**. The house/tree counts are accepted semantic clusters, not verified real object counts.

## What happened during development

1. The first classical reconstruction registered only part of the video and recovered suspicious camera intrinsics.
2. A fixed-intrinsics COLMAP run registered 54 stable keyframes, but the scale and lens model remained assumptions.
3. Early single-frame AI surfaces produced one depth sheet rather than a full map.
4. Full-video depth fusion was added so every decoded frame contributes.
5. Per-frame optical-flow/PnP tracking replaced interpolation between most keyframes: 54 anchors + 664 tracked frames.
6. MoGe-2 replaced the earlier single-view surface as the main depth engine. SegFormer added sky/ground/building/vegetation labels.
7. Sparse-track scale alignment and held-out depth refinement were added per frame.
8. The real clip incorrectly formed mountains. A field-specific plane model now estimates local ground and maps it onto one shared Y=0 plane.
9. Building and tree pixels initially became stretched sheets. Compact clusters now become closed wall/roof and trunk/canopy solids.
10. The browser had a legacy Y-axis inversion. Ground-aligned output now displays +Y upward.
11. Vertex colors were insufficient for realistic appearance. The current exporter creates UVs and embeds a photographic atlas assembled from registered video views.
12. Texture trials showed that tracked poses can be adequate for geometry but still smear photographs. The current atlas therefore uses the 54 bundle-adjusted anchors and verified ground pixels; weakly observed regions remain limited.

## Pipeline

```mermaid
flowchart LR
    A[Drone video] --> B[Decode and score frames]
    B --> C[COLMAP SIFT reconstruction]
    C --> D[Registered camera anchors]
    D --> E[Optical flow + PnP for every frame]
    A --> F[MoGe-2 depth and validity]
    A --> G[SegFormer semantic classes]
    D --> H[Sparse feature depth]
    E --> I[Per-frame pose]
    F --> J[Scale alignment and gated refinement]
    H --> J
    G --> K[Sky rejection and field/object masks]
    I --> L[Local field-plane estimation]
    J --> M[Open3D TSDF fusion]
    K --> M
    L --> M
    M --> N[Flat ground and bounded relief]
    N --> O[Approximate house/tree solids]
    D --> P[Registered-view texture samples]
    P --> Q[Ground-clean UV atlas]
    O --> R[GLB / PLY / JSON / reports]
    Q --> R
    R --> S[Three.js local viewer]
```

### Main components

- `pipeline/cli.py` — video validation, keyframe extraction, COLMAP feature extraction/matching/mapping, run manifests, and sparse exports.
- `pipeline/pose_tracking.py` — dense frame trajectory using optical flow, 3D feature tracks, robust PnP, and registered anchors.
- `pipeline/semantic_fusion.py` — main experimental pipeline: MoGe-2, SegFormer, sparse scale alignment, depth refinement, field leveling, TSDF fusion, object solidification, and exports.
- `pipeline/object_models.py` — connected semantic clusters, procedural houses/trees, closed meshes, and video-color transfer.
- `pipeline/texture_atlas.py` — top-down atlas rasterization, UV generation, gap filling, and object-footprint painting.
- `pipeline/train_aerial_corrector.py` — small residual depth corrector trained with frozen MoGe-2 predictions.
- `pipeline/ai_reconstruct.py` — optional Depth Anything 3 multi-view preview retained for comparison.
- `pipeline/video_fusion.py` — earlier DA3/Open3D fusion experiment retained for comparison.
- `pipeline/server.py` — loopback-only FastAPI worker API and artifact server.
- `apps/web/` — dependency-local Three.js reconstruction interface.
- `tests/` — camera, geometry, depth gating, object, texture, server, and pipeline tests.

## Models and libraries

Model weights are intentionally excluded from Git.

| Component | Current use | License note |
| --- | --- | --- |
| [MoGe-2 Base Normal](https://huggingface.co/Ruicheng/moge-2-vitb-normal) | Per-frame depth, validity, and geometry | MIT model repository |
| [SegFormer B0 ADE20K](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) | General semantic segmentation | Model card reports `other`; review NVIDIA/ADE20K terms before commercial use |
| [Depth Anything 3 Small](https://huggingface.co/depth-anything/DA3-SMALL) | Optional comparison preview | Apache-2.0 code; review model-card terms |
| COLMAP / PyCOLMAP | Sparse structure and camera anchors | See upstream project license |
| Open3D | TSDF fusion and mesh processing | MIT |
| OpenCV | Optical flow, PnP, and image processing | Apache-2.0 |
| Trimesh | Object construction and GLB export | MIT |
| Three.js | Local interactive viewer | MIT |

The project also uses NumPy, SciPy, PyTorch, Transformers, Pillow, PyAV, FastAPI, Uvicorn, and psutil. Review all upstream licenses and the licenses of any datasets or weights before redistribution or commercial use.

## Requirements

The tested development machine used Windows 11, Python 3.11, an NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM, CUDA-enabled PyTorch, FFmpeg on `PATH`, and Git. The classical stage can run on CPU. The full-video AI stage is designed for CUDA and was tuned around the 8 GB laptop GPU. The latest 718-frame run at width 336 took about five minutes after initialization; timing varies.

## Installation

### 1. Clone and create the classical/web environment

```powershell
git clone https://github.com/Prithibi17/aerorecon.git
cd aerorecon
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e . pytest httpx
```

### 2. Create the AI environment

Install a CUDA build of PyTorch compatible with the installed NVIDIA driver, then install the AI packages and upstream model code:

```powershell
py -3.11 -m venv .venv-ai
.\.venv-ai\Scripts\python.exe -m pip install --upgrade pip
# Choose the correct CUDA command from https://pytorch.org/get-started/locally/
.\.venv-ai\Scripts\python.exe -m pip install torch torchvision
.\.venv-ai\Scripts\python.exe -m pip install -r requirements-ai.txt

git clone --depth 1 https://github.com/microsoft/MoGe.git third_party/MoGe
git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git third_party/Depth-Anything-3
.\.venv-ai\Scripts\python.exe -m pip install -e third_party/MoGe
.\.venv-ai\Scripts\python.exe -m pip install -e third_party/Depth-Anything-3
```

`requirements-lock.txt` records the tested classical environment. `requirements-ai-lock.txt` is a snapshot of the development AI environment and includes local editable paths, so it is evidence rather than a portable installer.

### 3. Download model weights

```powershell
.\.venv-ai\Scripts\python.exe scripts\download_models.py
```

The script downloads MoGe-2 Base Normal to `models/MoGe-2-Base`, SegFormer B0 ADE20K to `models/SegFormer-B0`, and DA3 Small to `models/DA3-SMALL`. These large downloads stay outside Git.

## Run the application

Double-click `Start-AeroRecon.cmd` or run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn pipeline.server:app --host 127.0.0.1 --port 8765
```

Open <http://127.0.0.1:8765>. Use **New reconstruction** to upload an MP4, MOV, M4V, AVI, or MKV file up to 1 GB. When the camera stage completes, select the run and use **Build full-video map**.

The server binds to loopback and rejects cross-origin mutation requests. Uploaded videos and outputs remain in ignored local folders.

### Command-line workflow

```powershell
# Sparse camera reconstruction
.\.venv\Scripts\python.exe -m pipeline.cli run "C:\path\flight.mp4" --out outputs\flight-camera

# Full semantic map
.\.venv-ai\Scripts\python.exe -m pipeline.semantic_fusion outputs\flight-camera `
  --out outputs\flight-map --resolution 336 --flat-field --solid-objects
```

Use `--no-flat-field` for terrain that is not expected to be planar. Never enable the field constraint merely to make an incorrect reconstruction look flatter.

## Output files

| File | Purpose |
| --- | --- |
| `surface.glb` | Final textured triangle mesh with the atlas embedded. |
| `surface_full.ply` | Final surface in PLY form. |
| `dense.ply` | Colored point representation of final vertices. |
| `texture.png` | Generated texture atlas. |
| `surface_viewer.json` | Browser mesh positions, colors, semantic colors, indices, UVs, and texture reference. |
| `viewer.json` | Display point cloud, camera centers, coordinate flags, and counts. |
| `camera_centres.csv` | Per-frame relative camera positions. |
| `frames.csv` | Per-frame time, masks, depth scale, refinement decisions, and residuals. |
| `metrics.json` | Machine-readable result statistics. |
| `REPORT.md` | Human-readable method, warnings, and measurements. |
| `run_manifest.json` | Input hash, configuration, status, warnings, and provenance. |
| `keyframes/` | Semantic evidence overlays. |
| `sparse/`, `database.db` | Inspectable COLMAP evidence for camera-stage runs. |

## Why the current map is inaccurate

1. **Camera calibration is assumed.** Incorrect focal length or distortion changes camera motion, depth scale, and object shape.
2. **The video is mostly forward-looking.** Houses and trees do not have complete side, rear, and roof coverage.
3. **No GPS, IMU, gimbal angles, RTK, or ground-control points were supplied.** The system cannot independently recover metric scale, geographic position, gravity, or north.
4. **Monocular depth is ambiguous.** MoGe-2 predicts plausible geometry from appearance; it does not measure distance.
5. **SegFormer is out of domain.** ADE20K is a general scene dataset, so small aerial houses, crops, vegetation, and horizon regions are confused.
6. **Pose errors accumulate.** PnP-tracked poses can fuse geometry acceptably while remaining too imprecise for seamless photography.
7. **The flat-field prior is scene-specific.** It fixes this field clip but would erase genuine terrain relief.
8. **Object geometry is procedural.** A building cluster becomes a box and roof; a tree cluster becomes a trunk and canopy.
9. **Texture coverage is incomplete.** Registered anchors align best, but unobserved regions remain coarse or filled.
10. **There is no independent real-world test set.** Internal residuals can improve while absolute geometry remains wrong.

More AI models alone will not resolve missing viewpoints or missing ground truth. Better capture, calibration, telemetry, and evaluation should precede larger-model training.

## How to improve and train it

See [TRAINING_AND_CAPTURE.md](TRAINING_AND_CAPTURE.md) for the existing residual-corrector experiment and capture checklist.

### Phase 1 — establish ground truth

Collect multiple independent sites and keep complete sites in only one of train, validation, or test. Retain original frames, exact camera/lens data, calibration, timestamps, GPS/RTK, IMU attitude, gimbal angle, altitude source, coordinate reference system, nadir and oblique views, surveyed control/check points, aligned LiDAR or independently verified meshes, semantic masks, building/tree instance IDs, measured object subsets, and licensing/privacy records.

Never train on an AeroRecon reconstruction and then report accuracy against that same reconstruction.

### Phase 2 — fix calibration and pose

1. Add a calibration profile for every drone/camera/lens.
2. Read and synchronize SRT/telemetry with exact frame timestamps.
3. Use GPS/IMU as priors and optimize cameras with bundle adjustment.
4. Detect rolling shutter and electronic stabilization.
5. Add loop closure and global pose-graph optimization.
6. Compare camera ATE/RPE against RTK/IMU before depth evaluation.

Texture and geometry cannot align reliably until camera calibration and pose are accurate.

### Phase 3 — aerial semantic fine-tuning

Fine-tune segmentation on aerial labels. Start with SegFormer B0 for a controlled baseline. Use class-balanced crops, hard negatives at roofs/roads/field edges/horizon, temporal consistency, separate building/tree instances, site-level splits, per-class IoU/F1, boundary F1, instance AP, and calibrated uncertainty.

Possible starting datasets: [UAVid](https://uavid.nl/), [ISPRS 2D Semantic Labeling](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx), and [SensatUrban](https://github.com/QingyongHu/SensatUrban). Verify scope and licensing; none exactly matches this rural forward-looking video.

### Phase 4 — depth and surface training

The current 0.21-million-parameter residual corrector keeps MoGe-2 frozen. It used 716 TartanAir training images and 84 images from a separate trajectory for validation. Best normalized-depth AbsRel changed from 0.1167 to 0.1020, a 12.6% relative improvement on one synthetic environment. This does not prove real-video improvement.

For real training:

1. Precompute base-model depth with fixed hashes.
2. Train on RGB, log depth, calibrated intrinsics, normals, and semantics.
3. Combine scale-invariant depth, normal, edge-aware, temporal reprojection, and multi-view losses.
4. Give independently measured depth more weight than synthetic or pseudo-label depth.
5. Validate on unseen sites, drones, seasons, altitudes, and lighting.
6. Promote only when real held-out geometry and dimensions improve without pose regressions.

Current experiment command:

```powershell
.\.venv-ai\Scripts\python.exe -m pipeline.train_aerial_corrector `
  --dataset data\training\tartanair\ArchVizTinyHouseDay\Data_easy\ArchVizTinyHouseDay\Data_easy `
  --cache data\training\cache --out models\Aerial-Depth-Corrector `
  --samples 800 --epochs 8 --batch 8
```

[TartanAir](https://tartanair.org/) is useful for synthetic depth, pose, segmentation, and optical flow, but synthetic results must remain separate from real-world accuracy claims.

### Phase 5 — replace procedural objects

Track individual building/tree instances across frames, fuse their silhouettes and depth separately, estimate roof/wall planes, optimize meshes against multi-view evidence, preserve uncertainty, and mark unseen surfaces. For vegetation, compare meshes with point-based or neural radiance representations rather than forcing every crown into a solid mesh.

### Phase 6 — calibrated texture reconstruction

Unwrap a non-overlapping UV atlas; rank views per triangle by visibility, angle, resolution, blur, and exposure; reject occlusion with the mesh; compensate exposure/white balance; optimize seams; label unseen texels; and evaluate held-out PSNR, SSIM, LPIPS, seam energy, and coverage.

### Phase 7 — independent evaluation

| Subsystem | Required metrics |
| --- | --- |
| Camera trajectory | Absolute Trajectory Error, Relative Pose Error, rotation error |
| Depth | AbsRel, RMSE, δ thresholds, normal angular error |
| Surface | Chamfer distance, thresholded F-score, completeness |
| Ground | plane-normal error, elevation RMSE, slope error |
| Buildings | footprint IoU, height MAE, roof accuracy, instance precision/recall |
| Trees | crown-center error, crown-diameter MAE, instance precision/recall |
| Texture | held-out PSNR/SSIM/LPIPS, seam score, observed coverage |
| Georeferencing | horizontal/vertical GCP and independent checkpoint RMSE |

The UI should fail closed below documented calibration/pose/depth thresholds and show spatial uncertainty rather than one project-wide number.

## Better capture protocol

1. Keep original camera files; disable digital zoom and post-stabilization.
2. Lock exposure, focus, white balance, resolution, and frame rate when possible.
3. Fly a nadir grid with 75–85% forward and 70–80% side overlap.
4. Add a slower 35–45° oblique cross-grid around buildings and trees.
5. Keep motion smooth, shutter speed high, and subjects static.
6. Revisit the start area for loop closure.
7. Preserve telemetry/SRT and exact timestamps.
8. Measure visible scale/control and reserve independent checkpoints.
9. Avoid sky-dominant footage, heavy compression, blur, abrupt rolling-shutter turns, reflective water, and moving vegetation where practical.

See OpenDroneMap's [flight planning guidance](https://docs.opendronemap.org/flying/).

## Tests

```powershell
# Classical/server suite
.\.venv\Scripts\python.exe -m pytest -q

# AI geometry suite
.\.venv-ai\Scripts\python.exe -m unittest `
  tests.test_object_models tests.test_depth_refinement tests.test_pose_tracking
```

Current classical/server result: 22 tests passed and 5 optional AI-environment tests were skipped. The Phase 1 synthetic end-to-end validation registered 24/24 cameras with 12,683 sparse points and 0.243 px mean reprojection error. The previously documented AI geometry suite passed 10 tests in the AI environment.

## Repository policy

Python environments, model weights, original/uploaded videos, datasets, caches, generated reconstructions, logs, and cloned upstream repositories remain local and are ignored. Do not commit videos, checkpoints, GPS tracks, control coordinates, or generated maps without explicit license, privacy, and size review.

## License status

No project license has been selected yet. Until one is added, copyright law reserves rights to the project source. Every model, dataset, library, and third-party repository keeps its own license and usage terms.
