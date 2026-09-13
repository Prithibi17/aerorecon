"""Classical CUDA multi-view stereo and evidence-based texture reconstruction."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
import trimesh
from PIL import Image

from pipeline.cli_support import save_json
from pipeline.progress import ProgressWriter


STAGES = (
    "Image undistortion",
    "PatchMatch stereo",
    "Depth-map fusion",
    "Surface reconstruction",
    "Multi-view texturing",
    "Viewer export",
)


def execute(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise RuntimeError(f"COLMAP command failed with exit code {result.returncode}: {command[1]}")


def load_transform(path: Path | None):
    if not path:
        return np.eye(3), np.zeros(3), False
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data["rotation"], float), np.asarray(data["origin"], float), True


def transform_points(points, rotation, origin):
    return (np.asarray(points) - origin) @ rotation.T


def export_textured_surface(textured_mesh: Path, texture_path: Path, output: Path,
                            rotation: np.ndarray, origin: np.ndarray,
                            ground_aligned: bool) -> tuple[int, int]:
    """Export COLMAP's per-face UV mesh for Three.js and as an embedded GLB."""
    mesh = trimesh.load(textured_mesh, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Expected one textured mesh in {textured_mesh}")
    if ground_aligned:
        mesh.vertices = transform_points(mesh.vertices, rotation, origin)
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.asarray(mesh.visual.uv), image=Image.open(texture_path).convert("RGB"))
    mesh.export(output / "surface.glb")
    shutil.copy2(texture_path, output / "texture.png")
    payload = {
        "positions": np.asarray(mesh.vertices, np.float32).ravel().tolist(),
        "colors": np.full((len(mesh.vertices), 3), 255, np.uint8).ravel().tolist(),
        "indices": np.asarray(mesh.faces, np.uint32).ravel().tolist(),
        "uv": np.asarray(mesh.visual.uv, np.float32).ravel().tolist(),
        "texture": "texture.png",
        "geometry_provenance": "COLMAP calibrated geometric multi-view stereo",
    }
    target = output / "surface_viewer.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temp.replace(target)
    return len(mesh.vertices), len(mesh.faces)


def create_non_sky_masks(images: Path, output: Path, model_path: Path) -> tuple[int, float]:
    """Create COLMAP fusion masks with semantic sky pixels set to zero."""
    import cv2
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    processor = SegformerImageProcessor.from_pretrained(str(model_path))
    model = SegformerForSemanticSegmentation.from_pretrained(str(model_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    sky_id = next(int(i) for i, name in model.config.id2label.items() if name == "sky")
    image_paths = sorted(path for path in images.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sky_fractions = []
    kernel = np.ones((7, 7), np.uint8)
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        with torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            labels = torch.nn.functional.interpolate(
                logits, size=(image.height, image.width), mode="bilinear", align_corners=False
            ).argmax(1)[0].cpu().numpy()
        sky = cv2.dilate((labels == sky_id).astype(np.uint8), kernel)
        mask = np.where(sky, 0, 255).astype(np.uint8)
        target = output / image_path.relative_to(images)
        target = target.with_name(target.name + ".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(target)
        sky_fractions.append(float(sky.mean()))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return len(image_paths), float(np.mean(sky_fractions)) if sky_fractions else 0.0


def run(args):
    source = Path(args.source_run).resolve(strict=True)
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    progress = ProgressWriter(output, STAGES)
    started = time.time()
    source_manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    source_metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    model = source / "sparse" / str(source_metrics["best_model_id"])
    colmap = Path(args.colmap).resolve(strict=True)
    manifest = {
        "schema": "aerorecon.run/v1", "status": "running", "pid": os.getpid(),
        "started_at": started, "input": source_manifest["input"],
        "input_sha256": source_manifest["input_sha256"], "engine": "colmap-mvs",
        "display_name": "Photogrammetric dense map · real image texture",
        "units": "arbitrary", "georeferenced": False,
        "configuration": {"source_run": str(source), "max_image_size": args.max_image_size,
                          "colmap": str(colmap), "ground_transform": args.ground_transform},
        "warnings": ["Relative scale: no telemetry, GCP, or known distance was supplied.",
                     "Only surfaces verified in overlapping views are reconstructed; sky and unseen faces are intentionally absent."],
    }
    save_json(output / "run_manifest.json", manifest)
    active = STAGES[0]
    try:
        progress.update(active, 0, "Undistorting registered keyframes")
        execute([str(colmap), "image_undistorter", "--image_path", str(source / "keyframes"),
                 "--input_path", str(model), "--output_path", str(workspace),
                 "--output_type", "COLMAP", "--max_image_size", str(args.max_image_size),
                 "--num_patch_match_src_images", str(args.source_images)])
        progress.update(active, 1, "Undistorted images and cameras ready")

        active = STAGES[1]
        progress.update(active, .02, "Estimating geometrically consistent depth maps on the GPU")
        execute([str(colmap), "patch_match_stereo", "--workspace_path", str(workspace),
                 "--workspace_format", "COLMAP", "--PatchMatchStereo.gpu_index", "0",
                 "--PatchMatchStereo.max_image_size", str(args.max_image_size),
                 "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.filter", "1",
                 "--PatchMatchStereo.num_iterations", str(args.iterations),
                 "--PatchMatchStereo.cache_size", str(args.cache_gb)])
        progress.update(active, 1, "Geometrically filtered depth maps ready")

        active = STAGES[2]
        fusion_command = [str(colmap), "stereo_fusion", "--workspace_path", str(workspace),
                          "--workspace_format", "COLMAP", "--input_type", "geometric"]
        semantic_mask_model = Path(args.semantic_mask_model).resolve() if args.semantic_mask_model else None
        if semantic_mask_model and semantic_mask_model.exists():
            progress.update(active, .02, "Removing semantic sky pixels before depth fusion")
            mask_dir = output / "fusion_masks"
            mask_count, sky_fraction = create_non_sky_masks(workspace / "images", mask_dir, semantic_mask_model)
            fusion_command.extend(["--StereoFusion.mask_path", str(mask_dir)])
            manifest["configuration"]["semantic_mask_model"] = str(semantic_mask_model)
            manifest["configuration"]["semantic_masks"] = mask_count
            manifest["configuration"]["mean_sky_fraction"] = sky_fraction
            save_json(output / "run_manifest.json", manifest)
        progress.update(active, .15, "Fusing consistent non-sky depth observations")
        raw_dense = output / "dense_raw.ply"
        fusion_command.extend(["--output_type", "PLY", "--output_path", str(raw_dense),
                               "--StereoFusion.max_image_size", str(args.max_image_size),
                               "--StereoFusion.min_num_pixels", str(args.min_consistent_pixels),
                               "--StereoFusion.max_reproj_error", "2.0",
                               "--StereoFusion.max_depth_error", "0.02"])
        execute(fusion_command)
        progress.update(active, 1, "Dense colored point cloud fused")

        active = STAGES[3]
        progress.update(active, .05, "Reconstructing a conservative Poisson surface")
        raw_mesh = output / "mesh_raw.ply"
        execute([str(colmap), "poisson_mesher", "--input_path", str(raw_dense),
                 "--output_path", str(raw_mesh), "--PoissonMeshing.depth", str(args.poisson_depth),
                 "--PoissonMeshing.trim", str(args.poisson_trim), "--PoissonMeshing.color", "1"])
        progress.update(active, 1, "Observed surface mesh reconstructed")

        active = STAGES[4]
        progress.update(active, .03, "Reducing the display mesh while preserving observed surfaces")
        viewer_mesh = output / "mesh_viewer_raw.ply"
        execute([str(colmap), "mesh_simplifier", "--input_path", str(raw_mesh),
                 "--output_path", str(viewer_mesh),
                 "--MeshSimplification.target_face_ratio", str(args.viewer_face_ratio)])
        progress.update(active, .2, "Selecting visible source images for each mesh region")
        texture_dir = output / "textured"
        texture_dir.mkdir()
        execute([str(colmap), "mesh_texturer", "--workspace_path", str(workspace),
                 "--input_path", str(viewer_mesh), "--output_path", str(texture_dir),
                 "--output_type", "BIN", "--MeshTextureMapping.apply_color_correction", "1",
                 "--MeshTextureMapping.texture_scale_factor", str(args.texture_scale)])
        progress.update(active, 1, "Calibrated multi-view texture atlas ready")

        active = STAGES[5]
        progress.update(active, .1, "Preparing browser and export artifacts")
        rotation, origin, ground_aligned = load_transform(Path(args.ground_transform).resolve() if args.ground_transform else None)
        cloud = trimesh.load(raw_dense, process=False)
        points = transform_points(cloud.vertices, rotation, origin) if ground_aligned else np.asarray(cloud.vertices)
        colors = np.asarray(cloud.colors)[:, :3] if getattr(cloud, "colors", None) is not None else np.full((len(points), 3), 180, np.uint8)
        trimesh.points.PointCloud(points, colors=colors).export(output / "dense.ply")
        stride = max(1, int(np.ceil(len(points) / args.viewer_points)))
        take = np.arange(0, len(points), stride)
        import pycolmap
        reconstruction = pycolmap.Reconstruction(model)
        cameras = []
        for image in sorted(reconstruction.images.values(), key=lambda item: item.name):
            centre = image.projection_center()
            if ground_aligned:
                centre = transform_points(np.asarray(centre)[None], rotation, origin)[0]
            cameras.append({"name": image.name, "centre": np.asarray(centre).tolist()})
        surface_vertices, surface_faces = export_textured_surface(
            texture_dir / "mesh.ply", texture_dir / "texture.png", output,
            rotation, origin, ground_aligned)
        save_json(output / "viewer.json", {
            "positions": points[take].astype(np.float32).ravel().tolist(),
            "colors": colors[take].ravel().tolist(), "cameras": cameras,
            "total_points": len(points), "displayed_points": len(take), "units": "arbitrary",
            "geometry_provenance": "COLMAP geometric multi-view stereo", "surface_available": True,
            "ground_aligned": ground_aligned,
        })
        shutil.copy2(source / "frames.csv", output / "frames.csv")
        shutil.copy2(source / "keyframe_contact_sheet.jpg", output / "keyframe_contact_sheet.jpg")
        with (output / "camera_centres.csv").open("w", encoding="utf-8", newline="") as stream:
            import csv
            writer = csv.writer(stream); writer.writerow(["image", "x", "y", "z"])
            writer.writerows([item["name"], *item["centre"]] for item in cameras)
        texture_files = [path for path in texture_dir.rglob("*") if path.is_file()]
        metrics = {
            "engine": "colmap-mvs", "reconstruction": "geometric multi-view stereo",
            "registered_images": source_metrics["registered_images"],
            "selected_images": source_metrics["selected_images"],
            "registered_ratio": source_metrics["registered_ratio"],
            "points": len(points), "displayed_points": len(take),
            "surface_vertices": surface_vertices, "surface_triangles": surface_faces,
            "mean_reprojection_error_px": source_metrics["mean_reprojection_error_px"],
            "ground_aligned": ground_aligned, "geometry_validated": False,
            "units": "arbitrary", "scale_state": "relative",
            "texture_files": [str(path.relative_to(output)) for path in texture_files],
            "surface_glb": "surface.glb",
            "max_image_size": args.max_image_size,
        }
        save_json(output / "metrics.json", metrics)
        (output / "REPORT.md").write_text(
            "# Photogrammetric dense reconstruction\n\n"
            f"COLMAP registered {source_metrics['registered_images']}/{source_metrics['selected_images']} keyframes, "
            f"then estimated geometrically consistent depth maps and fused {len(points):,} colored 3D points.\n\n"
            "The point cloud uses colors observed at verified 3D locations. Procedural houses, cylinders, tree crowns, "
            "monocular-depth surfaces, and top-down texture projection are not used in this result.\n\n"
            "The mesh and multi-view texture files are under `textured/`. Coordinate scale remains relative because no "
            "telemetry, GCPs, or known distance was supplied. Missing or weakly textured surfaces remain incomplete.\n",
            encoding="utf-8")
        manifest["status"] = "complete"; manifest["metrics"] = metrics
        progress.complete("Photogrammetric dense map ready")
    except BaseException as error:
        manifest["status"] = "failed"; manifest["error"] = str(error)
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        progress.fail(active, str(error)); raise
    finally:
        manifest["elapsed_s"] = time.time() - started
        save_json(output / "run_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description="COLMAP CUDA dense MVS for an AeroRecon sparse run")
    parser.add_argument("source_run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--colmap", default="third_party/colmap-4.0.4-cuda/bin/colmap.exe")
    parser.add_argument("--ground-transform", default="")
    parser.add_argument("--max-image-size", type=int, default=1200)
    parser.add_argument("--source-images", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--cache-gb", type=float, default=4)
    parser.add_argument("--min-consistent-pixels", type=int, default=4)
    parser.add_argument("--poisson-depth", type=int, default=10)
    parser.add_argument("--poisson-trim", type=float, default=8)
    parser.add_argument("--texture-scale", type=float, default=.75)
    parser.add_argument("--viewer-face-ratio", type=float, default=.2)
    parser.add_argument("--semantic-mask-model", default="models/SegFormer-B0")
    parser.add_argument("--viewer-points", type=int, default=300000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
