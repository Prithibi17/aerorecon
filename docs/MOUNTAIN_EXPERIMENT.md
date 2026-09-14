# Mountain capture and training attempt

Date: 14 September 2026. Project: AeroRecon. User-supplied SIH problem ID: SIH26158; exact statement title remains unverified.

## Input

- File: `12438510_3840_2160_24fps.mp4` (43,553,873 bytes).
- SHA-256: `721f778c7b4ec5d3159650c047a44ff7903b4a454e71fc145305768e9d9aaec5`.
- Video: 3840×2160 H.264, 24000/1001 FPS, 13.0964 seconds.
- Scene: distant snowy mountains and vegetation; foreground terrain visible.
- Inspected container tags include format/encoder information, with no GPS/IMU telemetry found. No sidecar telemetry was supplied.
- Source media is local and not committed to Git. Reuse rights have not been independently established.

## Attempts and evidence

| Item | Initial attempt | Denser recovery attempt |
|---|---:|---:|
| Run | mountain-sfm-v1 | mountain-sfm-v2 |
| Candidate frames | 40 | 27 |
| Selected frames | 8 | 27 |
| Selected time range | 0–11.3447 s | 0–13.013 s |
| Matching | Sequential overlap 10 | Exhaustive |
| Verified image pairs | 17 | 351 |
| PLANAR_OR_PANORAMIC pairs | 17 | 351 |
| Matches per verified pair | 6,792–10,249 | 9,350–10,764 |
| Connected sparse model | None | None |
| Elapsed seconds | 106.31 | 760.92 |

Both runs used estimated SIMPLE_RADIAL intrinsics. COLMAP could not find a suitable initial image pair even after its initialization relaxation attempts. The second run ensures that the first selector's redundancy rejection was not the only reason for failure.

## Interpretation and training outcome

There is abundant repeatable image texture, but the recovered pair geometry is planar/panoramic. This is consistent with insufficient useful depth parallax, for example distant terrain observed from nearly the same location or a rotation-dominant camera. It does not prove the camera was completely stationary. A separately calibrated capture would help disambiguate the cause.

No connected camera model means there is no valid calibrated input to this pipeline's MVS and mesh-seeded Gaussian trainer. **No mountain neural training was completed, and no mountain mesh, Gaussian checkpoint or held-out PSNR exists.** The earlier agricultural scene's scores must not be copied into the mountain result.

Flat-field leveling, planar gap fill, procedural house/tree completion and invented geographic coordinates were not applied to the mountain video. The code also fixes previously hard-coded ground-aligned metadata in Gaussian exports, preserving unaligned coordinates for future terrain runs.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pipeline.cli run PATH_TO_VIDEO.mp4 --out outputs\mountain-retry --fps 2 --max-frames 30 --max-keyframe-gap 0.45 --exhaustive --name "Mountain capture test"
```

Use a new output folder. Future runs cap CPU feature extraction at four threads and 2000-pixel working resolution; this differs from the uncapped extraction used for the logged attempts. Original keyframes remain full resolution.

For the original evidence, inspect local `outputs/mountain-sfm-v1` and `outputs/mountain-sfm-v2`: `run_manifest.json`, `keyframe_selection.json`, `database.db`, `capture_diagnostics.json`, `error.log`, and the generated report.

## Next input needed

A video that translates across the terrain while maintaining strong overlapping views, with foreground and middle-distance features visibly shifting relative to the background. Preserve original lens/camera metadata, avoid post-stabilization and digital zoom, and retain telemetry if available. The next test should verify camera recovery before committing time to dense reconstruction and scene-specific training.

## SIH talking point

“We tested two captures/domains, including a mountain clip. The mountain test failed at camera initialization despite hundreds of matching image pairs. We report that failure because image similarity is not enough to establish depth. Our next milestone is a controlled translated capture and independent geometry validation.”

This demonstrates diagnostics and an honest limitation. It does not demonstrate successful mountain mapping or a generally trained terrain AI.
