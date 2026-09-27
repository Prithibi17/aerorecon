# Small-object and vegetation detail

Implemented 27 September 2026. This update is code-tested, but has not yet been visually validated on a regenerated drone scene. The previous local environment, weights and generated depth/mesh files were missing when work resumed. The source video remains available.

## Changes

- Semantic completion reads images up to 1280 pixels wide instead of shrinking them to the Gaussian training camera resolution (often 640). Camera intrinsics are rescaled with the images.
- A full-image SegFormer prediction supplies context. Overlapping 512-pixel crops supply local detail; their probabilities are blended with tapered weights to avoid hard tile seams. This reuses the existing model, not a newly trained detector.
- Separate surface classes: unknown, ground, building, tree, sky, vehicle, low vegetation, field. Cars, trucks and buses belong to vehicle; plants/bushes are a low-vegetation proxy. These are semantic labels, not individual object detections or counts. Crop type and bush species are not identified.
- Labels require at least two depth-consistent supporting views and 60% vote agreement. Existing five-view geometry support and sky exclusion remain active. Inferred backs have unknown labels.
- The observed-mesh budget increases from 60,000 to 120,000 faces. Set it to zero to retain all supported faces, at a potentially substantial memory cost. Existing 16-pixel per-face texture tiles remain a limitation.
- The viewer's **Detected surface classes** mode colors meshes and point clouds: red vehicles, orange buildings, dark green trees, light green low vegetation, yellow fields, brown ground, gray unknown. Existing results need regeneration to acquire these labels.
- `semantic_details.json` reports observed vertex counts per class, not object counts.

## Reconstruction settings

For an existing calibrated dense run, use the Open3D detail preset:

```powershell
python -m pipeline.open3d_refine outputs/drone-photogrammetry-dense --out outputs/drone-detail-mesh --detail
```

The preset halves the default relative-scale voxel length and truncation distance, lowers component removal from 150 to 20 triangles, limits smoothing to one iteration and raises the mesh budget to at least 500,000 triangles. This preserves more small geometry but also retains more noise and costs memory. These lengths are in reconstruction units, not metres. It cannot increase the resolution of the source MVS depth maps.

Semantic completion requires the source mesh, original dense run, and camera dataset exported for that exact mesh. Do not reuse cameras transformed for another mesh. With those matching artifacts:

```powershell
python -m pipeline.complete_scene --source MATCHING_MESH_RUN --dense MATCHING_DENSE_RUN --cameras MATCHING_CAMERA_RUN --out outputs/scene-details --semantic-width 1280 --semantic-tile 512 --semantic-overlap 128 --observed-face-budget 120000
```

This completion stage still assumes approximately flat ground. Do not apply its ground flattening or inferred ground fill to mountains. The Open3D detail preset alone does not impose a ground plane.

## What remains unproven

More labels and triangles do not establish better reconstruction accuracy. Vehicles moving between frames violate static-scene reconstruction assumptions; this update does not implement motion tracking or guarantee their geometry. Tiny objects may occupy too few pixels, and wind-blown crops may have inconsistent stereo depth. Unseen detail cannot be recovered from these settings.

Next validation: regenerate calibrated depths and mesh, inspect the same parked cars, roofs, bushes and field boundaries in source frames and rendered camera views, measure semantic errors against manual labels, compare held-out renders and geometric reference measurements, and report missing objects and false positives. Do not claim improved accuracy or a successful detailed-video reconstruction until that comparison is complete.
