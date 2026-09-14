# AeroRecon: SIH presentation content and judge preparation

This document is a factual briefing for the team and for an AI presentation or interview assistant. It is not an official SIH template. Use the current template supplied by your college SPOC or SIH portal. The user supplied problem ID **SIH26158**. Its exact official title and sponsoring organisation could not be verified in the public search results. Team name, institution and exact stakeholder requirements remain placeholders, not invented facts.

## One-sentence pitch

AeroRecon is a local prototype that turns overlapping video views into an inspectable 3D reconstruction, with semantic sky filtering, surface and appearance reconstruction, and explicit labels for inferred or unvalidated geometry.

## Problem and proposed use

Ordinary video is easy to view but difficult to use for inspecting spatial relationships across an area. A local workflow can help a user explore the observed landscape and export a model without uploading the source video. Candidate users include terrain visualisation teams, researchers, and education/demo users. Disaster assessment, infrastructure inspection, and planning are future validation targets, not established deployments or approved safety uses.

The final SIH pitch must connect these capabilities to a specific official problem statement and stakeholder workflow. Do not claim a match until the team provides that statement.

## Suggested presentation content

Adapt this content to the required slide count and headings; these are content blocks, not a claim about the current official template.

### 1. Title and problem

- Project: AeroRecon AI.
- Team / institution: [TEAM] / [INSTITUTION].
- Problem statement: SIH26158 — [verified official title and organisation pending].
- User pain: video provides many images but no immediately inspectable shared 3D scene.
- Scope: local visual reconstruction; dimensions and geographic position require external control data.

Speaker note: Say what the selected stakeholder needs to decide, and what a 3D preview would help them inspect. Avoid promising reliable measurements from an arbitrary clip.

### 2. Proposed solution and demonstration

- Upload a video and preserve the original locally.
- Select useful overlapping keyframes across its duration.
- Recover camera positions, estimate consistent depth, and reconstruct a shared scene.
- Inspect video, keyframes, mesh, points, Gaussian appearance, and camera path.
- Export geometry, checkpoints, quality reports, and a GPS/IMU input template.
- Switch inferred completion off to distinguish measured image evidence from approximate filled regions.

Demo: source video → reconstruction → orbit → observed/inferred comparison → quality report → export. Keep local artifacts ready if live processing is too slow.

### 3. Technical architecture

Video → PyAV/OpenCV frame analysis → COLMAP/PyCOLMAP SfM → undistorted cameras/images → CUDA PatchMatch MVS → semantic sky masks → dense fusion → Open3D TSDF → mesh.

Camera poses + source images + initial mesh → gsplat/PyTorch scene optimisation → Gaussian PLY/checkpoint + withheld-view renders.

Optional matched GPS camera positions → similarity alignment → local metric ENU coordinates. IMU fields are currently preserved as evidence; they are not fused in a full visual-inertial optimiser.

Optional field completion → semantic object clusters and ground patches → inferred mesh layer. Disable field-plane and house/tree completion assumptions on mountain terrain.

### 4. What is distinctive and what is reused

- Our contribution is the integrated local workflow, artifact provenance, scene inspection, semantic filtering, failure reporting, and experiments with completion and validation.
- COLMAP, Open3D, gsplat and the pretrained networks are existing research/software components. We did not invent their underlying algorithms.
- Relative to a plain image viewer, this adds shared camera geometry and spatial inspection. Relative to mature photogrammetry products, quality, robustness, automation, and validation remain gaps.
- A side-by-side benchmark against a documented baseline is needed before claiming superior quality, speed, cost, or novelty.

### 5. Feasibility, evidence and limits

- Demonstrated locally on an RTX 4060 Laptop GPU; the mountain-specific outcome is recorded in the README evidence table.
- Mountain test: 27 sampled views and 351 exhaustive verified pairs still produced no connected camera model; every pair was planar-or-panoramic. No mountain Gaussian training was completed. See [the experiment report](MOUNTAIN_EXPERIMENT.md).
- The agricultural MVS experiment registered 75/75 selected images, with about 1.04 px sparse reprojection error and 486,360 fused points.
- The original field Gaussian run used 80,000 primitives and 2,000 optimisation steps, scoring 15.47 dB on 10 withheld images. That result was visibly blurry and is not a high-quality accuracy claim.
- A later field completion run rejected 7,354 steep faces labelled as ground and added illustrative ground patches. Low ground-plane support (about 25%) remains a concern.
- No ground-truth surface, independently surveyed checkpoints, validated metre-scale dimensions, or georeferenced output has been demonstrated with the supplied videos.
- No proven cloud cost, production throughput, stakeholder adoption, or percentage accuracy exists yet.

### 6. Impact, development plan and references

- Potential benefit: easier inspection of observed areas and local handling of source video.
- Next milestone: a controlled capture with calibrated camera, adequate translation/overlap, matched telemetry, and independent checkpoints.
- Evaluate geometry accuracy, completeness, sky contamination, texture fit, runtime, peak memory, and failure rate across several held-out scenes.
- Add robust outlier handling, semantic confidence maps, better occlusion checks, camera synchronisation, and explicit terrain modes.
- Any operational use requires validation against the stakeholder's accuracy, safety, and geographic requirements.

## What the system actually uses

| Component | Role | Learned here? |
|---|---|---|
| PyAV / FFmpeg | Video decoding, timestamps and metadata | No |
| OpenCV | Sharpness, motion, image transforms and texture operations | No |
| COLMAP / PyCOLMAP, SIFT | Feature matching, triangulation, camera recovery and bundle adjustment | Estimates scene geometry; not neural-network training |
| CUDA PatchMatch MVS | Multi-view depth and geometric consistency | No general AI training |
| Open3D | Dense RGBD/TSDF fusion, cleanup and mesh operations | No |
| SegFormer-B0 | Pretrained semantic labels including sky, terrain and vegetation | Pretrained model inference; not fine-tuned on these videos |
| MoGe-2 / earlier DA3 experiments | Predicted monocular depth in experimental paths | Pretrained inference; these are not the primary measured MVS path |
| gsplat + PyTorch | Optimise Gaussian position, size, rotation, opacity and RGB to one scene | Yes, scene-specific optimisation |
| Custom aerial depth corrector | Experimental residual correction | Earlier synthetic-data experiment, not proof of real aerial accuracy |
| Three.js / GaussianSplats3D | Browser rendering, orbit, layers and exports | No |
| GPS alignment module | Match camera centres to WGS84 positions and fit a similarity transform | No neural training; requires real telemetry |
| Completion module | Close compact object clusters and fill illustrative field ground | Geometric heuristics and local colours, not a generative world model |

## Training: what can and cannot be claimed

Training a Gaussian scene is fitting the appearance of that particular video. It does not teach SegFormer what mountains are, train MoGe, or produce a general model that understands all future flights. A high image-fit score may still correspond to wrong geometry, especially with a nearly stationary camera.

General improvement would require multiple legally usable scenes and useful supervision: masks for sky/terrain, calibrated poses, depth/LiDAR or reference meshes, camera metadata, and telemetry. Split entire locations/flights into training, validation and test sets; splitting neighbouring video frames alone leaks almost identical content. Fine-tune only a specific subsystem with an explicit failure mode, and compare against the unchanged baseline. Keep a final unseen-location test set.

Suggested experiments:

1. Sky segmentation: annotate snow/cloud/ridge boundaries; report sky IoU, terrain falsely rejected, and sky falsely kept. Snow must not automatically become sky.
2. Depth/cameras: compare with calibrated, translated captures and reference geometry; measure depth error and pose error, not only image reprojection.
3. Mesh: measure Chamfer/F-score and completeness against reference surfaces; record unsupported filled area separately.
4. Appearance: withhold viewpoints or full trajectory segments; report PSNR/SSIM and visible seams. Use novel viewpoints sufficiently different from training views.
5. Terrain modes: test steep slopes without any flattening; evaluate vegetation separately from rigid buildings.
6. Georeferencing: align on some controls and report errors on different, independently held-out checkpoints.

## Likely judge questions and defensible answers

These are practice questions, not predictions of an actual jury's script.

| Question | Answer direction / evidence to show |
|---|---|
| What exact SIH problem are you solving? | State the official ID, stakeholder and workflow. This information still needs to be supplied. |
| What did your team build instead of downloading? | Show orchestration, viewer, reports, filtering, completion and evaluation code; credit upstream reconstruction engines and models. |
| Why use video instead of photographs? | Convenience and temporal overlap; disadvantages include compression, redundancy, rolling shutter and poor capture paths. Photographs can be better inputs. |
| Can any video produce a reliable map? | No. Translation, overlap, texture, calibration and static content matter; rotation-only or distant scenes may be unobservable in depth. |
| Why is the mountain clip difficult? | Distant terrain, similar views, snow, clouds and potentially little camera translation. Use the actual run diagnostics, not an assumption of successful reconstruction. |
| How do you avoid flattening mountains? | No flat-field completion or forced ground plane in the mountain reconstruction. Coordinate orientation is separate from terrain height variation. |
| Why did the earlier model contain walls/sky? | Semantic mistakes, unstable stereo near the horizon, and texture visibility errors. Show stricter mask support and triangle checks; do not claim perfect exclusion. |
| How are missing backs generated? | Approximate closed envelopes from local clusters with transferred colours, presented as an optional inferred layer. They are not recovered unseen details. |
| What does 1 px reprojection error prove? | Camera-feature image consistency; it does not establish geographic or surface accuracy. |
| What does PSNR prove? | Image appearance fit on the specified held-out views; it does not prove correct depth or structure. |
| What was trained on the new video? | If the run completes: scene-specific Gaussians. If camera/depth reconstruction fails: report that no valid Gaussian training was completed. |
| How many datasets have you tested? | Only report the documented local scenes/experiments. Two clips are not a robust benchmark. |
| Is the map georeferenced or in metres? | Not for the supplied clips. The alignment code needs matched GPS/control points; no real-world accuracy result is available. |
| Does it use IMU fusion? | CSV fields exist, but full time-synchronised visual-inertial estimation is future work. |
| How long does it take and what does it cost? | Quote measured stage timings and hardware. Local software cost does not mean zero hardware, energy, capture or engineering cost. |
| What happens without an NVIDIA GPU? | Some classical stages can run on CPU; the configured dense CUDA/gsplat path requires compatible GPU/software. No equivalent CPU throughput claim exists. |
| Can it work during a disaster or offline? | Local processing can run with dependencies/models already installed. Operational disaster suitability has not been validated. |
| What about moving trees, snow, water or clouds? | They can violate static-scene assumptions; mask/reject uncertain evidence and report missing areas. |
| How do you handle privacy and data rights? | Preserve local source data, exclude media/telemetry from Git, confirm capture/reuse rights, and check each model's license. |
| Is everything commercially open source? | No blanket claim. Libraries/models have different terms; the repository currently has no selected project license. Review the README license table. |
| How will you validate improvements? | Same inputs/settings and independent test scenes; geometry, completeness, appearance, runtime and rejection-rate comparisons. |
| What would you demonstrate if live reconstruction fails? | A saved local example plus its source and report; openly show the failed run and its cause instead of substituting a fabricated result. |

## Copy-paste prompt for an AI mock judge

```text
Act as a technically demanding Smart India Hackathon mock judge. Use only the
AeroRecon README, this briefing, and run reports I supply. First list the project
facts, measured evidence, limitations, and missing team/problem-statement details.
Do not invent accuracy, deployment, licenses, novelty, users, test results, or a
successful training run. Distinguish inference, scene-specific Gaussian fitting,
general model fine-tuning, photogrammetric evidence, and inferred completion.

Generate 30 likely questions across problem fit, originality, architecture,
computer vision, sky/snow separation, mountain slopes, geometry validation,
texture, GPS/IMU, hardware, feasibility, privacy, economics, and adoption.
Rank each easy/medium/hard and explain what evidence the judge is seeking.
For each, give a concise honest answer using the supplied evidence and a tougher
follow-up. Mark unknown facts as UNKNOWN and suggest how to measure them.

Then run a mock viva: ask one question at a time, wait for my answer, score clarity,
technical correctness and evidential support from 0–5, point out unsupported claims,
and offer an improved 30-second answer. Focus extra questions on our observed
failures: sky walls, missing ground, texture smearing, weak parallax, and uncertain
scale. Do not imply that these questions are known to be asked by the real jury.

Finally turn the verified facts into concise presentation content in the current
official template I provide. Leave team/PS placeholders where necessary. Include
a reproducible demo sequence, honest limitations, and next validation milestones.
```

## References and current-template caution

- [Official SIH college guidelines](https://sih.gov.in/letters/Guidelines-College-SPOC.pdf): published guidance includes idea novelty, feasibility and related evaluation themes. The document available at this URL refers to an older edition; it is not confirmation of the current year's dates or slide rules.
- [SIH portal](https://sih.gov.in/): obtain the current official problem statement and submission format through the portal/SPOC.
- [COLMAP](https://colmap.github.io/), [Open3D](https://www.open3d.org/docs/release/), [gsplat](https://docs.gsplat.studio/): upstream technical references; AeroRecon uses these components.
- Project evidence: README result tables and local `outputs/*/run_manifest.json`, `metrics.json`, `REPORT.md`, and validation renders. Generated artifacts are not included in the repository.
