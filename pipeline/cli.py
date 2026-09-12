"""Initial local CLI. Every run uses a new directory and retains failure evidence."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import av
import cv2
import numpy as np


def save_json(path, data):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract(video, out, fps=2.0, max_frames=160, min_sharpness=25.0):
    if not math.isfinite(fps) or fps <= 0 or max_frames < 3:
        raise ValueError('FPS must be positive and max_frames must be at least 3')
    if not math.isfinite(min_sharpness) or min_sharpness < 0:
        raise ValueError('Minimum sharpness must be finite and nonnegative')
    images = out / 'keyframes'
    images.mkdir()
    rows = []
    selected = 0
    next_time = None
    previous = None
    orb = cv2.ORB_create(2000)
    with av.open(str(video)) as container:
        if not container.streams.video:
            raise ValueError('Input contains no video stream')
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError('Frame lacks a presentation timestamp')
            stamp = float(frame.pts * frame.time_base)
            if previous is not None and stamp <= previous:
                raise ValueError('Video timestamps are not strictly increasing')
            previous = stamp
            if next_time is not None and stamp + 1e-9 < next_time:
                continue
            next_time = stamp + 1 / fps
            im = frame.to_ndarray(format='bgr24')
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            clipped = float(((gray < 5) | (gray > 250)).mean())
            features = len(orb.detect(gray, None))
            reasons = []
            if sharp < min_sharpness:
                reasons.append('blur')
            if clipped > .35:
                reasons.append('exposure')
            if features < 100:
                reasons.append('few_features')
            name = f'frame_{index:08d}.jpg'
            keep = not reasons
            if keep:
                if not cv2.imwrite(str(images / name), im, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError('Failed to write frame')
                selected += 1
            rows.append(dict(image=name, source_frame=index, pts=frame.pts,
                             time_base=str(frame.time_base), time_s=stamp,
                             sharpness=sharp, clipped_fraction=clipped,
                             features=features, selected=keep, reason=','.join(reasons)))
            if selected >= max_frames:
                break
    if rows:
        with (out / 'frames.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if selected < 3:
        raise ValueError(f'Only {selected} usable frames; inspect frames.csv or use sharper footage')
    return rows


def reconstruct(out, camera_params='', exhaustive=False, fix_intrinsics=False):
    import pycolmap as pc
    db = out / 'database.db'
    reader = pc.ImageReaderOptions()
    reader.camera_model = 'SIMPLE_RADIAL'
    if camera_params:
        reader.camera_params = camera_params
    print('Extracting SIFT features...', flush=True)
    pc.extract_features(db, out / 'keyframes', camera_mode=pc.CameraMode.SINGLE,
                        reader_options=reader, device=pc.Device.cpu)
    print('Matching sequential frames...', flush=True)
    pairs = pc.SequentialPairingOptions()
    pairs.overlap = 10
    if exhaustive:
        print('Matching all image pairs...', flush=True)
        pc.match_exhaustive(db, device=pc.Device.cpu)
    else:
        pc.match_sequential(db, pairing_options=pairs, device=pc.Device.cpu)
    options = pc.IncrementalPipelineOptions()
    options.num_threads = 4
    options.random_seed = 0
    options.init_num_trials = 40
    options.max_num_models = 5
    options.max_runtime_seconds = 240
    if fix_intrinsics:
        if not camera_params:
            raise ValueError('Fixed intrinsics require explicit camera parameters')
        options.ba_refine_focal_length = False
        options.ba_refine_extra_params = False
        options.mapper.abs_pose_refine_focal_length = False
        options.mapper.abs_pose_refine_extra_params = False
    sparse = out / 'sparse'
    sparse.mkdir()
    print('Recovering cameras and sparse geometry...', flush=True)
    models = pc.incremental_mapping(db, out / 'keyframes', sparse, options=options)
    if not models:
        raise RuntimeError('No connected model. Inspect frame quality, overlap and camera calibration.')
    best_id, model = max(models.items(), key=lambda pair: pair[1].num_reg_images())
    model.export_PLY(out / 'sparse.ply')
    text_dir = out / 'model_text'
    text_dir.mkdir()
    model.write_text(text_dir)
    with (out / 'camera_centres.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'x', 'y', 'z'])
        for im in sorted(model.images.values(), key=lambda x: x.name):
            writer.writerow([im.name, *im.projection_center()])
    selected = len(list((out / 'keyframes').glob('*.jpg')))
    metrics = dict(registered_images=model.num_reg_images(), selected_images=selected,
                   registered_ratio=model.num_reg_images()/selected,
                   points=model.num_points3D(), mean_reprojection_error_px=model.compute_mean_reprojection_error(),
                   mean_track_length=model.compute_mean_track_length(), best_model_id=int(best_id),
                   model_count=len(models), units='arbitrary', georeferenced=False,
                   geometry_provenance='estimated', pycolmap_version=pc.__version__)
    metrics['cameras'] = [dict(model=cam.model_name, parameters=cam.params.tolist(),
                              width=cam.width, height=cam.height,
                              focal_ratio=float(cam.focal_length/max(cam.width,cam.height)))
                          for cam in model.cameras.values()]
    metrics['calibration_suspect'] = any(cam['focal_ratio'] < .2 or cam['focal_ratio'] > 5 for cam in metrics['cameras'])
    metrics['intrinsics_fixed'] = fix_intrinsics
    metrics['geometry_validated'] = False
    save_json(out / 'metrics.json', metrics)
    (out / 'REPORT.md').write_text(
        '# AeroRecon sparse reconstruction\n\n'
        f'Registered {metrics["registered_images"]}/{selected} frames; {metrics["points"]} points.\n\n'
        f'Mean reprojection error: {metrics["mean_reprojection_error_px"]:.3f} pixels.\n\n'
        'Coordinates have arbitrary scale. This is not yet a metric or georeferenced map.\n'
        f'Calibration plausibility check: {"FAILED — unreliable geometry" if metrics["calibration_suspect"] else "no extreme focal estimate detected; accuracy still unvalidated"}.\n'
        f'Fixed camera intrinsics: {fix_intrinsics}. Input parameters must be independently verified.\n'
        'Sparse geometry is an initial result; dense surfaces and telemetry alignment are later stages.\n', encoding='utf-8')
    return metrics


def run(args):
    video = Path(args.video).resolve(strict=True)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    manifest = dict(schema='aerorecon.run/v1', status='running', input=str(video),
                    input_sha256=sha256(video), configuration=vars(args),
                    units='arbitrary', georeferenced=False, pid=os.getpid(),
                    started_at=start, display_name=getattr(args, 'name', '') or video.name)
    manifest['configuration'] = {k: v for k, v in vars(args).items() if k != 'func'}
    save_json(out / 'run_manifest.json', manifest)
    try:
        print('Decoding and checking video frames...', flush=True)
        rows = extract(video, out, args.fps, args.max_frames, args.min_sharpness)
        manifest['selected_frames'] = sum(r['selected'] for r in rows)
        manifest['warnings'] = []
        if manifest['selected_frames'] == args.max_frames:
            manifest['warnings'].append('Frame cap reached; reconstruction may cover only the beginning of the video')
        if not args.prepare_only:
            manifest['metrics'] = reconstruct(out, args.camera_params, getattr(args, 'exhaustive', False), getattr(args, 'fix_intrinsics', False))
            if manifest['metrics']['calibration_suspect']:
                manifest['warnings'].append('UNRELIABLE GEOMETRY: implausible focal length estimated. This point cloud is a diagnostic result, not a usable map.')
            if getattr(args, 'fix_intrinsics', False):
                manifest['warnings'].append('Camera intrinsics were fixed to the supplied values. This is an experimental result unless those values are independently verified.')
            if manifest['metrics']['registered_ratio'] < .8:
                manifest['warnings'].append('Partial reconstruction: fewer than 80% of selected frames registered. The model does not cover the entire clip reliably.')
            if manifest['metrics']['model_count'] > 1:
                manifest['warnings'].append('Multiple disconnected models were found. The viewer shows the model with the most registered images.')
        manifest['status'] = 'prepared' if args.prepare_only else 'complete'
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = str(exc)
        (out / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        manifest['elapsed_s'] = time.time()-start
        save_json(out / 'run_manifest.json', manifest)
    print(f'Artifacts: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='AeroRecon video to sparse 3D')
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('run')
    command.add_argument('video')
    command.add_argument('--out', required=True, help='New output directory')
    command.add_argument('--fps', type=float, default=2)
    command.add_argument('--max-frames', type=int, default=160)
    command.add_argument('--min-sharpness', type=float, default=25)
    command.add_argument('--camera-params', default='', help='SIMPLE_RADIAL f,cx,cy,k in source pixels')
    command.add_argument('--prepare-only', action='store_true')
    command.add_argument('--exhaustive', action='store_true', help='Match all frame pairs; slower but useful for short difficult clips')
    command.add_argument('--name', default='', help='Display name in the local interface')
    command.add_argument('--fix-intrinsics', action='store_true', help='Freeze explicitly supplied camera calibration')
    command.set_defaults(func=run)
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
