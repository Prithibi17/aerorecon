"""Fit a metric WGS84 local frame from per-image GPS and transform mesh/splats."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def geodetic_to_ecef(latitude, longitude, altitude):
    lat, lon = np.radians(latitude), np.radians(longitude)
    normal = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return np.array([(normal + altitude) * np.cos(lat) * np.cos(lon),
                     (normal + altitude) * np.cos(lat) * np.sin(lon),
                     (normal * (1 - WGS84_E2) + altitude) * np.sin(lat)])


def ecef_to_enu_matrix(latitude, longitude):
    lat, lon = np.radians(latitude), np.radians(longitude)
    return np.array([[-np.sin(lon), np.cos(lon), 0],
                     [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                     [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]])


def similarity(source, target):
    """Umeyama least-squares similarity: target = scale * R * source + t."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if len(source) < 3: raise ValueError("At least three matched GPS camera positions are required")
    source_mean, target_mean = source.mean(0), target.mean(0)
    x, y = source - source_mean, target - target_mean
    if np.linalg.matrix_rank(x) < 2: raise ValueError("GPS matches must cover a path with more than one direction")
    covariance = y.T @ x / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0: sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    variance = (x * x).sum() / len(source)
    scale = float((singular * sign).sum() / variance)
    translation = target_mean - scale * rotation @ source_mean
    predicted = (scale * (rotation @ source.T)).T + translation
    residuals = np.linalg.norm(predicted - target, axis=1)
    return scale, rotation, translation, residuals


def rotation_quaternion(rotation):
    # Matrix to wxyz quaternion conversion for a proper rotation.
    r = np.asarray(rotation, float); trace = np.trace(r)
    if trace > 0:
        s = np.sqrt(trace + 1) * 2
        return np.array([.25*s, (r[2,1]-r[1,2])/s, (r[0,2]-r[2,0])/s, (r[1,0]-r[0,1])/s])
    i = int(np.argmax(np.diag(r))); j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1 + r[i,i] - r[j,j] - r[k,k]) * 2
    q = np.empty(4); q[0] = (r[k,j] - r[j,k]) / s; q[i+1] = .25 * s
    q[j+1] = (r[j,i] + r[i,j]) / s; q[k+1] = (r[k,i] + r[i,k]) / s
    return q


def quaternion_multiply(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0); bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], axis=-1)


def transform_gaussians(source, target, scale, rotation, translation):
    from plyfile import PlyData, PlyElement
    ply = PlyData.read(str(source)); vertex = ply["vertex"].data.copy()
    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")])
    xyz = (scale * (rotation @ xyz.T)).T + translation
    for i, name in enumerate(("x", "y", "z")): vertex[name] = xyz[:, i]
    for name in ("scale_0", "scale_1", "scale_2"): vertex[name] += np.log(scale)
    q = np.column_stack([vertex[f"rot_{i}"] for i in range(4)])
    q_align = np.repeat(rotation_quaternion(rotation)[None], len(q), axis=0)
    q = quaternion_multiply(q_align, q)
    q /= np.linalg.norm(q, axis=1, keepdims=True).clip(1e-8)
    for i in range(4): vertex[f"rot_{i}"] = q[:, i]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(target))


def run(args):
    import trimesh
    source = Path(args.run).resolve(strict=True); telemetry = Path(args.telemetry).resolve(strict=True)
    output = Path(args.out).resolve(); output.mkdir(parents=True, exist_ok=False)
    viewer = json.loads((source / "viewer.json").read_text(encoding="utf-8"))
    camera_map = {item["name"]: np.asarray(item["centre"], float) for item in viewer["cameras"]}
    rows = []
    with telemetry.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                if row.get("image") in camera_map and row.get("latitude") and row.get("longitude"):
                    rows.append({**row, "latitude": float(row["latitude"]), "longitude": float(row["longitude"]),
                                 "altitude_m": float(row.get("altitude_m") or 0)})
            except ValueError:
                continue
    if len(rows) < 3: raise ValueError("Telemetry must contain GPS for at least three reconstructed image names")
    anchor = rows[len(rows) // 2]
    anchor_ecef = geodetic_to_ecef(anchor["latitude"], anchor["longitude"], anchor["altitude_m"])
    enu_from_ecef = ecef_to_enu_matrix(anchor["latitude"], anchor["longitude"])
    gps_enu = np.array([enu_from_ecef @ (geodetic_to_ecef(r["latitude"], r["longitude"], r["altitude_m"]) - anchor_ecef) for r in rows])
    model = np.array([camera_map[r["image"]] for r in rows])
    scale, rotation, translation, residuals = similarity(model, gps_enu)
    if not np.isfinite(scale) or scale <= 0: raise ValueError("GPS alignment produced an invalid scale")
    if np.median(residuals) > args.max_median_error:
        raise ValueError(f"GPS alignment median error {np.median(residuals):.2f} m exceeds limit {args.max_median_error:.2f} m")

    matrix = np.eye(4); matrix[:3, :3] = scale * rotation; matrix[:3, 3] = translation
    mesh = trimesh.load(source / "surface.glb", force="scene")
    mesh.apply_transform(matrix); mesh.export(output / "surface_georeferenced.glb")
    transform_gaussians(source / "gaussians.ply", output / "gaussians_georeferenced.ply", scale, rotation, translation)
    with (output / "camera_georeference.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["image", "east_m", "north_m", "up_m", "gps_residual_m"])
        predicted = (scale * (rotation @ model.T)).T + translation
        writer.writerows([row["image"], *predicted[i], residuals[i]] for i, row in enumerate(rows))
    report = {"schema": "aerorecon.georeference/v1", "georeferenced": True, "units": "metres",
              "coordinate_system": "WGS84 local tangent ENU", "anchor": {"latitude": anchor["latitude"],
              "longitude": anchor["longitude"], "altitude_m": anchor["altitude_m"]},
              "model_to_enu": matrix.tolist(), "scale_metres_per_model_unit": scale,
              "matched_cameras": len(rows), "rmse_m": float(np.sqrt(np.mean(residuals ** 2))),
              "median_error_m": float(np.median(residuals)), "max_error_m": float(residuals.max()),
              "imu_fields_available": any(r.get("yaw_deg") for r in rows)}
    (output / "georeference.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run"); parser.add_argument("--telemetry", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--max-median-error", type=float, default=10.0)
    run(parser.parse_args())


if __name__ == "__main__": main()
