"""Conservative calibration plausibility labels for reconstructed cameras."""
from __future__ import annotations


def assess_cameras(cameras: list[dict]) -> str:
    """Return failed, wide_angle_review, telephoto_review, or estimated.

    Ratios use focal length divided by the longer image edge.  Values around
    0.1 occur on legitimate ultra-wide cameras, so they need review rather
    than being treated as a reconstruction failure.
    """
    ratios = [float(camera.get("focal_ratio", 0)) for camera in cameras]
    if not ratios or any(not 0.10 <= ratio <= 3.0 for ratio in ratios):
        return "failed"
    if any(ratio < 0.20 for ratio in ratios):
        return "wide_angle_review"
    if any(ratio > 2.0 for ratio in ratios):
        return "telephoto_review"
    return "estimated"


def failed(metrics: dict) -> bool:
    """Support historical result manifests that only stored focal_ratio."""
    status = metrics.get("calibration_status")
    if status is not None:
        return status == "failed"
    # Earliest manifests did not retain camera parameters. Preserve their
    # recorded decision instead of silently classifying an empty list as bad.
    if not metrics.get("cameras"):
        return bool(metrics.get("calibration_suspect", True))
    return assess_cameras(metrics.get("cameras", [])) == "failed"
