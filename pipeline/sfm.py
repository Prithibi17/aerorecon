"""COLMAP structure-from-motion with calibration checks and diagnostics."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from pipeline.cli_support import save_json


PARAMETER_COUNTS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
}


@dataclass(frozen=True)
class CameraConfig:
    model: str = "SIMPLE_RADIAL"
    parameters: str = ""
    fix_intrinsics: bool = False

    def validate(self) -> None:
        if self.model not in PARAMETER_COUNTS:
            raise ValueError(f"Unsupported camera model {self.model}; choose {', '.join(PARAMETER_COUNTS)}")
        if self.fix_intrinsics and not self.parameters:
            raise ValueError("Fixed intrinsics require explicit camera parameters")
        if self.parameters:
            try:
                values = [float(value.strip()) for value in self.parameters.split(",")]
            except ValueError as exc:
                raise ValueError("Camera parameters must be comma-separated numbers") from exc
            expected = PARAMETER_COUNTS[self.model]
            if len(values) != expected:
                raise ValueError(f"{self.model} requires {expected} parameters, received {len(values)}")
            if values[0] <= 0 or (self.model in {"PINHOLE", "OPENCV"} and values[1] <= 0):
                raise ValueError("Camera focal length must be positive")


def reconstruct(
    output: Path,
    camera: CameraConfig | None = None,
    *,
    exhaustive: bool = False,
    progress=None,
) -> dict:
    import pycolmap as pc

    camera = camera or CameraConfig()
    camera.validate()
    database = output / "database.db"
    reader = pc.ImageReaderOptions()
    reader.camera_model = camera.model
    if camera.parameters:
        reader.camera_params = camera.parameters
    save_json(output / "camera_configuration.json", {
        "model": camera.model,
        "parameters": camera.parameters or None,
        "intrinsics_fixed": camera.fix_intrinsics,
        "source": "user" if camera.parameters else "COLMAP estimate",
    })

    print("Extracting SIFT features...", flush=True)
    if progress:
        progress.update("Feature extraction", .05, "Extracting local SIFT features")
    try:
        pc.extract_features(database, output / "keyframes", camera_mode=pc.CameraMode.SINGLE,
                            reader_options=reader, device=pc.Device.cpu)
    except Exception as exc:
        raise RuntimeError(f"COLMAP feature extraction failed: {exc}") from exc
    if progress:
        progress.update("Feature extraction", 1.0, "SIFT feature database ready")

    if exhaustive:
        print("Matching all image pairs...", flush=True)
        matching_label = "exhaustive"
        if progress:
            progress.update("Frame matching", .05, "Matching every selected image pair")
        try:
            pc.match_exhaustive(database, device=pc.Device.cpu)
        except Exception as exc:
            raise RuntimeError(f"COLMAP exhaustive matching failed: {exc}") from exc
    else:
        print("Matching sequential frames...", flush=True)
        matching_label = "sequential overlap 10"
        pairs = pc.SequentialPairingOptions()
        pairs.overlap = 10
        if progress:
            progress.update("Frame matching", .05, "Matching nearby keyframes with temporal overlap")
        try:
            pc.match_sequential(database, pairing_options=pairs, device=pc.Device.cpu)
        except Exception as exc:
            raise RuntimeError(f"COLMAP sequential matching failed: {exc}") from exc
    if progress:
        progress.update("Frame matching", 1.0, "Feature correspondences verified")

    options = pc.IncrementalPipelineOptions()
    options.num_threads = 4
    options.random_seed = 0
    options.init_num_trials = 40
    options.max_num_models = 5
    options.max_runtime_seconds = 240
    if camera.fix_intrinsics:
        options.ba_refine_focal_length = False
        options.ba_refine_extra_params = False
        options.mapper.abs_pose_refine_focal_length = False
        options.mapper.abs_pose_refine_extra_params = False
    sparse = output / "sparse"
    sparse.mkdir()
    print("Recovering cameras and sparse geometry...", flush=True)
    if progress:
        progress.update("Sparse reconstruction", .05, "Initializing camera poses and triangulation")
    try:
        models = pc.incremental_mapping(database, output / "keyframes", sparse, options=options)
    except Exception as exc:
        raise RuntimeError(f"COLMAP sparse reconstruction failed: {exc}") from exc
    if not models:
        raise RuntimeError("No connected camera model was recovered. Check the contact sheet for blur, ensure 70–85% overlap, and provide calibrated intrinsics when available.")
    best_id, model = max(models.items(), key=lambda pair: pair[1].num_reg_images())
    model.export_PLY(output / "sparse.ply")
    text_dir = output / "model_text"
    text_dir.mkdir()
    model.write_text(text_dir)
    with (output / "camera_centres.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "x", "y", "z"])
        for image in sorted(model.images.values(), key=lambda item: item.name):
            writer.writerow([image.name, *image.projection_center()])
    selected = len(list((output / "keyframes").glob("*.jpg")))
    metrics = {
        "registered_images": model.num_reg_images(),
        "selected_images": selected,
        "registered_ratio": model.num_reg_images() / selected,
        "points": model.num_points3D(),
        "mean_reprojection_error_px": model.compute_mean_reprojection_error(),
        "mean_track_length": model.compute_mean_track_length(),
        "best_model_id": int(best_id),
        "model_count": len(models),
        "matching_strategy": matching_label,
        "units": "arbitrary",
        "scale_state": "relative",
        "georeferenced": False,
        "geometry_provenance": "COLMAP structure from motion",
        "pycolmap_version": pc.__version__,
    }
    metrics["cameras"] = [{
        "model": item.model_name,
        "parameters": item.params.tolist(),
        "width": item.width,
        "height": item.height,
        "focal_ratio": float(item.focal_length / max(item.width, item.height)),
    } for item in model.cameras.values()]
    metrics["calibration_suspect"] = any(item["focal_ratio"] < .2 or item["focal_ratio"] > 5 for item in metrics["cameras"])
    metrics["intrinsics_fixed"] = camera.fix_intrinsics
    metrics["intrinsics_source"] = "user" if camera.parameters else "COLMAP estimate"
    metrics["geometry_validated"] = False
    save_json(output / "metrics.json", metrics)
    (output / "REPORT.md").write_text(
        "# AeroRecon sparse reconstruction\n\n"
        f"Registered {metrics['registered_images']}/{selected} keyframes; {metrics['points']} sparse points.\n\n"
        f"Mean reprojection error: {metrics['mean_reprojection_error_px']:.3f} pixels.\n\n"
        f"Matching: {matching_label}. Intrinsics source: {metrics['intrinsics_source']}. "
        f"Calibration plausibility: {'FAILED' if metrics['calibration_suspect'] else 'no extreme focal estimate detected'}.\n\n"
        "Coordinates have relative scale and are not georeferenced. Reprojection error measures image fit, not ground accuracy. "
        "Dense surfaces, telemetry alignment, and independent accuracy validation are later stages.\n",
        encoding="utf-8",
    )
    if progress:
        progress.update("Sparse reconstruction", 1.0, f"Registered {model.num_reg_images()} of {selected} keyframes",
                        current=model.num_reg_images(), total=selected)
    return metrics
