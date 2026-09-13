"""Fuse COLMAP geometric depth maps into a cleaned Open3D TSDF surface."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import pycolmap
import trimesh

from pipeline.cli_support import save_json
from pipeline.dense_mvs import load_transform, transform_points
from pipeline.progress import ProgressWriter


STAGES = ("Load calibrated depth", "Open3D TSDF fusion", "Mesh cleanup", "Viewer export")


def read_colmap_array(path: Path) -> np.ndarray:
    """Read COLMAP's ampersand-header dense array format."""
    with path.open("rb") as stream:
        header = b""
        while header.count(b"&") < 3:
            byte = stream.read(1)
            if not byte:
                raise ValueError(f"Invalid COLMAP dense array: {path}")
            header += byte
        width, height, channels = map(int, header[:-1].split(b"&"))
        values = np.fromfile(stream, np.float32)
    expected = width * height * channels
    if values.size != expected:
        raise ValueError(f"Expected {expected} values in {path}, found {values.size}")
    result = values.reshape((width, height, channels), order="F").transpose(1, 0, 2)
    return result[..., 0] if channels == 1 else result


def remove_small_components(mesh: o3d.geometry.TriangleMesh, minimum_triangles: int) -> int:
    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    counts = np.asarray(counts)
    remove = counts[clusters] < minimum_triangles
    removed = int(remove.sum())
    mesh.remove_triangles_by_mask(remove)
    mesh.remove_unreferenced_vertices()
    return removed


def write_surface(mesh: o3d.geometry.TriangleMesh, output: Path) -> tuple[int, int]:
    vertices = np.asarray(mesh.vertices, np.float32)
    faces = np.asarray(mesh.triangles, np.uint32)
    colors = np.clip(np.asarray(mesh.vertex_colors) * 255, 0, 255).astype(np.uint8)
    if len(colors) != len(vertices):
        colors = np.full((len(vertices), 3), 180, np.uint8)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255)
    o3d.io.write_triangle_mesh(str(output / "surface_open3d.ply"), mesh, write_vertex_colors=True)
    tri = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
    tri.export(output / "surface.glb")
    payload = {
        "positions": vertices.ravel().tolist(), "colors": colors.ravel().tolist(),
        "indices": faces.ravel().tolist(),
        "geometry_provenance": "Open3D TSDF fusion of COLMAP geometric depth maps",
    }
    target = output / "surface_viewer.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temporary.replace(target)
    return len(vertices), len(faces)


def run(args) -> None:
    source = Path(args.source_run).resolve(strict=True)
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    workspace = source / "workspace"
    model_path = workspace / "sparse"
    reconstruction = pycolmap.Reconstruction(model_path)
    manifest_source = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    rotation, origin, ground_aligned = load_transform(
        Path(args.ground_transform).resolve() if args.ground_transform else None)
    progress = ProgressWriter(output, STAGES)
    started = time.time()
    manifest = {
        "schema": "aerorecon.run/v1", "status": "running", "pid": os.getpid(),
        "started_at": started, "input": manifest_source["input"],
        "input_sha256": manifest_source["input_sha256"], "engine": "open3d-tsdf",
        "display_name": "Open3D TSDF · calibrated non-sky fusion",
        "units": "arbitrary", "georeferenced": False,
        "configuration": {"source_run": str(source), "voxel_length": args.voxel_length,
                          "sdf_trunc": args.sdf_trunc, "depth_scale": args.depth_scale,
                          "ground_transform": args.ground_transform},
        "warnings": ["Relative scale: no telemetry, GCP, or known distance was supplied.",
                     "Open3D fuses calibrated observed depth; weakly observed and occluded surfaces may remain incomplete."],
    }
    save_json(output / "run_manifest.json", manifest)
    active = STAGES[0]
    try:
        depth_dir = workspace / "stereo" / "depth_maps"
        mask_dir = source / "fusion_masks"
        views = []
        for image in sorted(reconstruction.images.values(), key=lambda item: item.name):
            depth_path = depth_dir / f"{image.name}.geometric.bin"
            image_path = workspace / "images" / image.name
            mask_path = mask_dir / f"{image.name}.png"
            if depth_path.is_file() and image_path.is_file():
                views.append((image, depth_path, image_path, mask_path))
        if not views:
            raise RuntimeError("No calibrated COLMAP depth maps were found")
        progress.update(active, 1, f"Loaded {len(views)} calibrated geometric depth maps")

        active = STAGES[1]
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=args.voxel_length, sdf_trunc=args.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        integrated = 0
        valid_pixels = 0
        for index, (image, depth_path, image_path, mask_path) in enumerate(views):
            depth = read_colmap_array(depth_path)
            rgb = np.asarray(Image.open(image_path).convert("RGB"))
            if rgb.shape[:2] != depth.shape:
                rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
            valid = np.isfinite(depth) & (depth > 0)
            if mask_path.is_file():
                mask = np.asarray(Image.open(mask_path).convert("L"))
                if mask.shape != depth.shape:
                    mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
                valid &= mask > 0
            if valid.sum() < args.min_valid_pixels:
                continue
            far = float(np.quantile(depth[valid], args.depth_quantile))
            valid &= depth <= far
            clean_depth = np.where(valid, depth * args.depth_scale, 0).astype(np.float32)
            camera = reconstruction.cameras[image.camera_id]
            scale_x = depth.shape[1] / camera.width
            scale_y = depth.shape[0] / camera.height
            calibration = camera.calibration_matrix()
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                depth.shape[1], depth.shape[0], calibration[0, 0] * scale_x,
                calibration[1, 1] * scale_y, calibration[0, 2] * scale_x,
                calibration[1, 2] * scale_y)
            extrinsic = np.eye(4)
            extrinsic[:3] = image.cam_from_world().matrix()
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(rgb)),
                o3d.geometry.Image(clean_depth), depth_scale=1.0,
                depth_trunc=max(float(clean_depth.max()) + 1, 100.0),
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic, extrinsic)
            integrated += 1
            valid_pixels += int(valid.sum())
            progress.update(active, (index + 1) / len(views),
                            f"Fused {integrated}/{len(views)} depth maps",
                            current=index + 1, total=len(views))
        if integrated < 3:
            raise RuntimeError("Too few depth maps passed the Open3D fusion checks")
        mesh = volume.extract_triangle_mesh()
        if not len(mesh.triangles):
            raise RuntimeError("Open3D TSDF produced an empty surface")
        progress.update(active, 1, f"Open3D fused {valid_pixels:,} non-sky depth pixels")

        active = STAGES[2]
        before_faces = len(mesh.triangles)
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()
        removed_components = remove_small_components(mesh, args.min_component_triangles)
        if args.smooth_iterations:
            mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smooth_iterations)
        if len(mesh.triangles) > args.target_triangles:
            mesh = mesh.simplify_quadric_decimation(args.target_triangles)
        vertices = np.asarray(mesh.vertices)
        if ground_aligned:
            vertices[:] = transform_points(vertices, rotation, origin)
        mesh.compute_vertex_normals()
        progress.update(active, 1, f"Cleaned {before_faces:,} to {len(mesh.triangles):,} triangles")

        active = STAGES[3]
        surface_vertices, surface_faces = write_surface(mesh, output)
        colors = np.clip(np.asarray(mesh.vertex_colors) * 255, 0, 255).astype(np.uint8)
        stride = max(1, int(np.ceil(surface_vertices / args.viewer_points)))
        take = np.arange(0, surface_vertices, stride)
        points = np.asarray(mesh.vertices)
        trimesh.points.PointCloud(points, colors=colors).export(output / "dense.ply")
        cameras = []
        for image in sorted(reconstruction.images.values(), key=lambda item: item.name):
            centre = np.asarray(image.projection_center())
            if ground_aligned:
                centre = transform_points(centre[None], rotation, origin)[0]
            cameras.append({"name": image.name, "centre": centre.tolist()})
        save_json(output / "viewer.json", {
            "positions": points[take].astype(np.float32).ravel().tolist(),
            "colors": colors[take].ravel().tolist(), "cameras": cameras,
            "total_points": surface_vertices, "displayed_points": len(take),
            "units": "arbitrary", "geometry_provenance": "Open3D calibrated TSDF",
            "surface_available": True, "ground_aligned": ground_aligned,
        })
        with (output / "camera_centres.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(["image", "x", "y", "z"])
            writer.writerows([item["name"], *item["centre"]] for item in cameras)
        for name in ("frames.csv", "keyframe_contact_sheet.jpg"):
            if (source / name).is_file():
                shutil.copy2(source / name, output / name)
        shutil.copytree(workspace / "images", output / "keyframes")
        metrics = {
            "engine": "open3d-tsdf", "fusion": "open3d-tsdf",
            "reconstruction": "calibrated TSDF fusion", "registered_images": integrated,
            "selected_images": len(views), "registered_ratio": integrated / len(views),
            "points": surface_vertices, "displayed_points": len(take),
            "surface_vertices": surface_vertices, "surface_triangles": surface_faces,
            "input_mesh_triangles": before_faces, "small_component_triangles_removed": removed_components,
            "valid_depth_pixels": valid_pixels, "semantic_sky_masks": sum(item[3].is_file() for item in views),
            "ground_aligned": ground_aligned, "geometry_validated": False,
            "units": "arbitrary", "scale_state": "relative",
        }
        save_json(output / "metrics.json", metrics)
        (output / "REPORT.md").write_text(
            "# Open3D TSDF refinement\n\n"
            f"Open3D fused {integrated}/{len(views)} calibrated geometric depth maps using "
            f"{valid_pixels:,} non-sky pixels. The cleaned surface contains {surface_vertices:,} vertices "
            f"and {surface_faces:,} triangles. Small disconnected components and non-manifold edges were removed, "
            "then Taubin smoothing reduced stereo noise while preserving larger edges.\n\n"
            "Vertex colors come from the calibrated RGBD integrations. Geometry remains relative-scale and is not "
            "surveyed or georeferenced. Missing overlap, occlusions, weak crop texture, and distant objects limit completeness.\n",
            encoding="utf-8")
        manifest["status"] = "complete"; manifest["metrics"] = metrics
        progress.complete("Open3D refined surface ready")
    except BaseException as error:
        manifest["status"] = "failed"; manifest["error"] = str(error)
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        progress.fail(active, str(error)); raise
    finally:
        manifest["elapsed_s"] = time.time() - started
        save_json(output / "run_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open3D TSDF refinement of a COLMAP dense run")
    parser.add_argument("source_run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ground-transform", default="")
    parser.add_argument("--voxel-length", type=float, default=.015)
    parser.add_argument("--sdf-trunc", type=float, default=.06)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--depth-quantile", type=float, default=.997)
    parser.add_argument("--min-valid-pixels", type=int, default=5000)
    parser.add_argument("--min-component-triangles", type=int, default=150)
    parser.add_argument("--smooth-iterations", type=int, default=3)
    parser.add_argument("--target-triangles", type=int, default=250000)
    parser.add_argument("--viewer-points", type=int, default=300000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
