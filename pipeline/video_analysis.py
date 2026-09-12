"""Content-aware drone-video analysis and keyframe selection."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import av
import cv2
import numpy as np

from pipeline.cli_support import save_json


@dataclass(frozen=True)
class SelectionConfig:
    candidate_fps: float = 2.0
    max_frames: int = 160
    min_sharpness: float = 25.0
    max_clipped_fraction: float = 0.35
    min_features: int = 100
    max_similarity: float = 0.985
    min_motion_px: float = 1.5
    max_keyframe_gap_s: float = 2.0
    analysis_width: int = 640

    def validate(self) -> None:
        if not math.isfinite(self.candidate_fps) or self.candidate_fps <= 0:
            raise ValueError("Candidate FPS must be positive")
        if self.max_frames < 3:
            raise ValueError("Maximum frames must be at least 3")
        if not math.isfinite(self.min_sharpness) or self.min_sharpness < 0:
            raise ValueError("Minimum sharpness must be finite and nonnegative")
        if not 0 <= self.max_clipped_fraction <= 1:
            raise ValueError("Maximum clipped fraction must be between 0 and 1")
        if not 0 <= self.max_similarity <= 1:
            raise ValueError("Maximum similarity must be between 0 and 1")


def _analysis_gray(image: np.ndarray, width: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > width:
        scale = width / gray.shape[1]
        gray = cv2.resize(gray, (width, round(gray.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return gray


def _compare(previous_gray, previous_keypoints, previous_descriptors, gray, keypoints, descriptors):
    tiny_a = cv2.resize(previous_gray, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    tiny_b = cv2.resize(gray, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    similarity = float(1.0 - np.mean(np.abs(tiny_a - tiny_b)) / 255.0)
    motion = 0.0
    match_ratio = 0.0
    match_count = 0
    if previous_descriptors is not None and descriptors is not None:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(previous_descriptors, descriptors, k=2)
        good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
        match_count = len(good)
        match_ratio = match_count / max(1, min(len(previous_keypoints), len(keypoints)))
        if good:
            shifts = [np.linalg.norm(np.asarray(previous_keypoints[m.queryIdx].pt) - keypoints[m.trainIdx].pt) for m in good]
            motion = float(np.median(shifts))
    return similarity, motion, match_ratio, match_count


def create_contact_sheet(image_paths: list[Path], destination: Path, columns: int = 5) -> None:
    thumbs = []
    for path in image_paths[:30]:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = cv2.resize(image, (240, 135), interpolation=cv2.INTER_AREA)
        cv2.putText(image, path.stem[-8:], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(image)
    if not thumbs:
        return
    blank = np.zeros_like(thumbs[0])
    while len(thumbs) % columns:
        thumbs.append(blank)
    sheet = np.vstack([np.hstack(thumbs[i:i + columns]) for i in range(0, len(thumbs), columns)])
    cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def analyze_and_select(video: Path, output: Path, config: SelectionConfig, progress=None) -> list[dict]:
    config.validate()
    images = output / "keyframes"
    images.mkdir()
    rows: list[dict] = []
    selected_paths: list[Path] = []
    next_time = None
    previous_stamp = None
    last_selected_stamp = None
    previous_selected = None
    orb = cv2.ORB_create(2500)
    with av.open(str(video)) as container:
        if not container.streams.video:
            raise ValueError("Input contains no video stream")
        stream = container.streams.video[0]
        estimated_total = int(stream.frames or 0)
        metadata = {
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
            "codec": stream.codec_context.name,
            "declared_frames": estimated_total or None,
            "duration_s": float(stream.duration * stream.time_base) if stream.duration else None,
            "selection": asdict(config),
        }
        duration_s = metadata["duration_s"]
        candidate_interval = 1 / config.candidate_fps
        if duration_s:
            # Spread the frame budget across the complete clip instead of spending
            # it on the beginning of a long flight.
            candidate_interval = max(candidate_interval, duration_s / config.max_frames)
        metadata["effective_candidate_interval_s"] = candidate_interval
        save_json(output / "video_analysis.json", metadata)
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError("Frame lacks a presentation timestamp")
            stamp = float(frame.pts * frame.time_base)
            if previous_stamp is not None and stamp <= previous_stamp:
                raise ValueError("Video timestamps are not strictly increasing")
            previous_stamp = stamp
            if next_time is not None and stamp + 1e-9 < next_time:
                continue
            next_time = stamp + candidate_interval
            image = frame.to_ndarray(format="bgr24")
            gray = _analysis_gray(image, config.analysis_width)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            clipped = float(((gray < 5) | (gray > 250)).mean())
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            features = len(keypoints)
            similarity, motion, match_ratio, matches = (0.0, float("inf"), 0.0, 0)
            if previous_selected is not None:
                similarity, motion, match_ratio, matches = _compare(*previous_selected, gray, keypoints, descriptors)
            reasons = []
            if sharpness < config.min_sharpness:
                reasons.append("blur")
            if clipped > config.max_clipped_fraction:
                reasons.append("exposure")
            if features < config.min_features:
                reasons.append("few_features")
            forced_gap = last_selected_stamp is not None and stamp - last_selected_stamp >= config.max_keyframe_gap_s
            if previous_selected is not None and not forced_gap:
                if similarity > config.max_similarity and motion < config.min_motion_px:
                    reasons.append("redundant")
            name = f"frame_{index:08d}.jpg"
            keep = not reasons
            if keep and len(selected_paths) >= config.max_frames:
                keep = False
                reasons.append("frame_cap")
            if keep:
                path = images / name
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"Failed to write keyframe {name}")
                selected_paths.append(path)
                previous_selected = (gray, keypoints, descriptors)
                last_selected_stamp = stamp
            rows.append({
                "image": name, "source_frame": index, "pts": frame.pts,
                "time_base": str(frame.time_base), "time_s": stamp,
                "sharpness": sharpness, "clipped_fraction": clipped,
                "features": features, "similarity_to_previous": similarity,
                "motion_px": motion if math.isfinite(motion) else -1.0,
                "match_ratio": match_ratio, "matches": matches,
                "selected": keep, "reason": ",".join(reasons),
            })
            if progress and index % 10 == 0:
                fraction = index / estimated_total if estimated_total else min(.95, len(rows) / max(config.max_frames * 3, 1))
                progress.update("Video analysis", fraction, f"Analyzed source frame {index}", current=index, total=estimated_total or None)
    if rows:
        with (output / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    create_contact_sheet(selected_paths, output / "keyframe_contact_sheet.jpg")
    selected = len(selected_paths)
    rejected = len(rows) - selected
    save_json(output / "keyframe_selection.json", {
        "candidates": len(rows), "selected": selected, "rejected": rejected,
        "coverage_start_s": min((row["time_s"] for row in rows if row["selected"]), default=None),
        "coverage_end_s": max((row["time_s"] for row in rows if row["selected"]), default=None),
        "rejection_counts": {reason: sum(reason in row["reason"].split(",") for row in rows)
                             for reason in ("blur", "exposure", "few_features", "redundant", "frame_cap")},
        "configuration": asdict(config),
    })
    if progress:
        progress.update("Keyframe selection", 1.0, f"Selected {selected} of {len(rows)} candidates", current=selected, total=len(rows))
    if selected < 3:
        raise ValueError(f"Only {selected} usable keyframes; inspect frames.csv and keyframe_contact_sheet.jpg")
    return rows
