# Improving your drone reconstruction

The uploaded MP4 matches the original example byte-for-byte. Its repeat sparse reconstruction completed but produced a suspicious focal length (about 323 pixels for a 1920-pixel-wide image). Registering all frames therefore did not establish correct 3D geometry. The app now finds the matching camera run and checks calibration before dense processing. That existing camera run still uses assumed intrinsics, not independently calibrated ones.

## What to provide first

1. Original drone photos or video files, before editing, cropping or stabilization; include the drone and camera model if known.
2. Images looking downward and images from several angles around buildings. A single forward-looking pass may not show enough of their sides or roofs.
3. Any flight telemetry, original SRT sidecar, camera calibration or IMU/gimbal orientation. Gravity/orientation information directly helps determine down. GPS positions alone do not identify which surface is level.
4. A few independently measured visible distances and, if available, surveyed control points. Reserve some measurements for checking results rather than fitting them.

For a capture reference, OpenDroneMap describes combining downward views with a 45-degree cross-grid and 70–80% overlap for the oblique pass: [official capture guidance](https://docs.opendronemap.org/flying/). Follow applicable operating rules when collecting new footage.

## Can this one video train the model?

Not as reliable supervised geometry training by itself. It has images, but no trusted depth, gravity or building dimensions. Feeding our distorted mesh back as truth can reinforce its mistakes. More frames of the same view do not supply independent ground truth.

Semantic training and geometry training solve different problems. Hand-labeling sky, buildings and ground can improve class recognition; it does not teach correct depth or height. Depth training needs trustworthy 3D supervision, such as aligned depth/laser scans, independently verified reconstructions or synthetic scenes with known geometry.

## A practical fine-tuning sequence

1. Build a small, carefully checked pilot dataset spanning multiple sites, camera angles and lighting conditions. Keep entire sites separate between training and evaluation; adjacent frames must not cross that split.
2. Include RGB images, aligned depth, validity masks and known camera intrinsics. Add semantic labels and surface normals when available. Record the source and license of every sample.
3. Convert the data to the model's format. MoGe uses one folder per sample, an image, encoded depth, and metadata containing normalized camera intrinsics. Use its depth I/O helpers instead of saving an arbitrary grayscale depth PNG.
4. Fine-tune an existing checkpoint in a separate training environment. Start with a small memory test; your 8 GB GPU's inference success does not guarantee full training fits. Adjust accumulation/checkpointing and model size after measuring memory. Do not replace the working inference environment.
5. Compare the original and fine-tuned models on withheld sites using depth error, ground-normal angle, camera consistency and independently measured dimensions. Promote the new model only if these improve.

The official guide explains checkpoint initialization, configuration and the training launcher: [MoGe training and fine-tuning](https://github.com/microsoft/MoGe/blob/main/docs/train.md). The example launcher uses multiple GPUs and must be adapted; it is not a tested command for this laptop.

## Useful data resource

[TartanAir V2](https://tartanair.org/) provides synthetic images with depth, segmentation and camera poses. It can help test geometry and training infrastructure, but performance on simulation does not establish accuracy on your drone footage. Download a small relevant sample first, not the entire dataset.

## Training run completed

A small residual depth corrector was trained locally on the TartanAir V2 `ArchVizTinyHouseDay/Data_easy` subset. The download contains 866 paired RGB/depth images in seven trajectories. P000–P005 supplied 716 training samples; all 84 selected samples from P006 were held out for validation. The corrector has about 0.21 million parameters and leaves MoGe-2 frozen.

The best checkpoint reduced scale-normalized held-out absolute relative depth error from 0.1167 to 0.1020. This is a 12.6% relative improvement on one synthetic environment. It does not prove improvement on the uploaded real video. During real-video reconstruction, the corrector is accepted per frame only when sparse feature depths withheld from its scale fit improve by at least 3%; otherwise the original MoGe depth is used.

Training artifacts are in `models/Aerial-Depth-Corrector`; exact metrics are in `metrics.json`. Cached model inputs and the downloaded dataset are retained under `data/training` for inspection and reproducibility.

To repeat this experiment from the project folder after installing the AI environment, run:

```powershell
.venv-ai\Scripts\python.exe -m pipeline.train_aerial_corrector --dataset data\training\tartanair\ArchVizTinyHouseDay\Data_easy\ArchVizTinyHouseDay\Data_easy --cache data\training\cache --out models\Aerial-Depth-Corrector --samples 800 --epochs 8 --batch 8
```

The random seeds and whole-trajectory split are fixed in the script. The saved checkpoint SHA-256 for this run is `59BBA81B4B77E40113183ECACBB32980E38212774678517D060825274638DF88`.
