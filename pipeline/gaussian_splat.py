"""Prepare calibrated views and orchestrate GPU Gaussian Splatting training."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import pycolmap

from pipeline.cli_support import save_json
from pipeline.progress import ProgressWriter


STAGES = ("Calibrated image set", "Open3D mesh initialization", "PyTorch Gaussian training", "Quality validation", "Georeference check")


def aligned_world_matrix(transform_path: Path | None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if not transform_path or not transform_path.is_file():
        return result
    data = json.loads(transform_path.read_text(encoding="utf-8"))
    rotation = np.asarray(data["rotation"], dtype=np.float64)
    origin = np.asarray(data["origin"], dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = -rotation @ origin
    return result


def prepare_dataset(dense: Path, mesh_run: Path, output: Path, width: int, max_points: int) -> dict:
    workspace = dense / "workspace"
    reconstruction = pycolmap.Reconstruction(workspace / "sparse")
    mesh_manifest = json.loads((mesh_run / "run_manifest.json").read_text(encoding="utf-8"))
    transform_value = mesh_manifest.get("configuration", {}).get("ground_transform")
    transform_path = Path(transform_value).resolve() if transform_value else None
    aligned_from_raw = aligned_world_matrix(transform_path)
    raw_from_aligned = np.linalg.inv(aligned_from_raw)

    image_dir = output / "training" / "images"
    mask_dir = output / "training" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    cameras = []
    for image in sorted(reconstruction.images.values(), key=lambda item: item.name):
        source_image = workspace / "images" / image.name
        if not source_image.is_file():
            continue
        rgb = np.asarray(Image.open(source_image).convert("RGB"))
        height = max(1, round(rgb.shape[0] * width / rgb.shape[1]))
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        Image.fromarray(resized).save(image_dir / image.name, quality=92)
        source_mask = dense / "fusion_masks" / f"{image.name}.png"
        if source_mask.is_file():
            mask = np.asarray(Image.open(source_mask).convert("L"))
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.full((height, width), 255, np.uint8)
        Image.fromarray(mask).save(mask_dir / f"{image.name}.png")
        camera = reconstruction.cameras[image.camera_id]
        calibration = np.asarray(camera.calibration_matrix(), dtype=np.float64)
        calibration[0] *= width / camera.width
        calibration[1] *= height / camera.height
        view_raw = np.eye(4, dtype=np.float64)
        view_raw[:3] = image.cam_from_world().matrix()
        view_aligned = view_raw @ raw_from_aligned
        cameras.append({"name": image.name, "width": width, "height": height,
                        "K": calibration.tolist(), "world_to_camera": view_aligned.tolist(),
                        "centre": np.linalg.inv(view_aligned)[:3, 3].tolist()})

    mesh = o3d.io.read_triangle_mesh(str(mesh_run / "surface_open3d.ply"))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
    if len(colors) != len(vertices):
        colors = np.full_like(vertices, .6)
    if len(vertices) > max_points:
        # Evenly retain the whole surface rather than truncating one spatial region.
        take = np.linspace(0, len(vertices) - 1, max_points, dtype=np.int64)
        vertices, colors = vertices[take], colors[take]
    np.savez_compressed(output / "training" / "initial_gaussians.npz", means=vertices, colors=colors)
    dataset = {"schema": "aerorecon.gsplat-dataset/v1", "cameras": cameras,
               "initial_points": len(vertices), "coordinate_system": "ground-aligned-relative",
               "source_mesh": str(mesh_run / "surface_open3d.ply")}
    save_json(output / "training" / "dataset.json", dataset)
    return dataset


def telemetry_template(output: Path, cameras: list[dict]) -> None:
    rows = ["image,latitude,longitude,altitude_m,yaw_deg,pitch_deg,roll_deg,accuracy_m"]
    rows.extend(f"{item['name']},,,,,,," for item in cameras)
    (output / "telemetry_template.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def run(args) -> None:
    dense = Path(args.dense_run).resolve(strict=True)
    mesh_run = Path(args.mesh_run).resolve(strict=True)
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_manifest = json.loads((dense / "run_manifest.json").read_text(encoding="utf-8"))
    started = time.time()
    progress = ProgressWriter(output, STAGES)
    manifest = {"schema": "aerorecon.run/v1", "status": "running", "pid": os.getpid(),
                "started_at": started, "input": source_manifest["input"],
                "input_sha256": source_manifest["input_sha256"], "engine": "gsplat-pytorch",
                "display_name": "3D Gaussian Splatting · calibrated video appearance",
                "units": "arbitrary", "georeferenced": False,
                "configuration": {"dense_run": str(dense), "mesh_run": str(mesh_run),
                                  "width": args.width, "steps": args.steps,
                                  "max_gaussians": args.max_gaussians},
                "warnings": ["Coordinates remain relative because the MP4 contains no GPS or IMU telemetry.",
                             "Gaussian appearance is optimized only for surfaces and viewing directions observed by the video."]}
    save_json(output / "run_manifest.json", manifest)
    try:
        progress.update(STAGES[0], .1, "Reading calibrated COLMAP cameras")
        dataset = prepare_dataset(dense, mesh_run, output, args.width, args.max_gaussians)
        progress.update(STAGES[0], 1, f"Prepared {len(dataset['cameras'])} calibrated, sky-masked views")
        progress.update(STAGES[1], 1, f"Initialized {dataset['initial_points']:,} Gaussians from the Open3D surface")

        trainer = Path(args.trainer_python).resolve(strict=True)
        command = [str(trainer), "-m", "pipeline.gsplat_train", str(output / "training"),
                   "--out", str(output), "--steps", str(args.steps), "--eval-every", str(args.eval_every)]
        log_path = output / "training.log"
        progress.update(STAGES[2], .02, f"Training {dataset['initial_points']:,} Gaussians with PyTorch/CUDA")
        with log_path.open("w", encoding="utf-8", errors="replace") as stream:
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], stdout=stream,
                                    stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise RuntimeError(f"Gaussian trainer failed; see {log_path.name}")
        train_metrics = json.loads((output / "gsplat_metrics.json").read_text(encoding="utf-8"))
        progress.update(STAGES[2], 1, f"Optimized {train_metrics['gaussians']:,} Gaussians on the RTX GPU")
        progress.update(STAGES[3], 1, f"Held-out PSNR {train_metrics['holdout_psnr_db']:.2f} dB")

        shutil.copy2(mesh_run / "surface.glb", output / "surface.glb")
        shutil.copy2(mesh_run / "surface_open3d.ply", output / "surface_open3d.ply")
        shutil.copy2(mesh_run / "surface_viewer.json", output / "surface_viewer.json")
        shutil.copy2(mesh_run / "keyframe_contact_sheet.jpg", output / "keyframe_contact_sheet.jpg")
        with (output / "camera_centres.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(["image", "x", "y", "z"])
            writer.writerows([item["name"], *item["centre"]] for item in dataset["cameras"])
        telemetry_template(output, dataset["cameras"])
        progress.update(STAGES[4], 1, "Telemetry template ready; GPS/IMU not present in source video")
        metrics = {"engine": "gsplat-pytorch", "reconstruction": "3D Gaussian Splatting",
                   "registered_images": len(dataset["cameras"]), "selected_images": len(dataset["cameras"]),
                   "registered_ratio": 1.0, "points": train_metrics["gaussians"],
                   "displayed_points": train_metrics["viewer_gaussians"], "surface_available": True,
                   "ground_aligned": True, "gaussian_splats": True, **train_metrics,
                   "geometry_validated": False, "units": "arbitrary", "scale_state": "relative",
                   "georeference_state": "awaiting telemetry"}
        save_json(output / "metrics.json", metrics)
        manifest["status"] = "complete"; manifest["metrics"] = metrics
        (output / "REPORT.md").write_text(
            "# Gaussian Splatting reconstruction\n\n"
            f"{len(dataset['cameras'])} calibrated views trained {train_metrics['gaussians']:,} 3D Gaussians. "
            "The Open3D mesh initializes geometry; gsplat/PyTorch optimizes position, covariance, opacity, and color "
            "against the video frames while semantic masks exclude sky.\n\n"
            f"Held-out PSNR: {train_metrics['holdout_psnr_db']:.2f} dB. This measures novel-view image fit, not survey accuracy.\n\n"
            "Georeferencing is waiting for real per-frame GPS/IMU. Fill telemetry_template.csv and run the "
            "georeference command; coordinates are not fabricated.\n", encoding="utf-8")
        progress.complete("Gaussian Splat map ready · telemetry needed for geographic coordinates")
    except Exception as error:
        manifest["status"] = "failed"; manifest["error"] = str(error)
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        manifest["elapsed_s"] = time.time() - started
        save_json(output / "run_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dense_run")
    parser.add_argument("--mesh-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trainer-python", default=".venv-gsplat/Scripts/python.exe")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--max-gaussians", type=int, default=80000)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=250)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
