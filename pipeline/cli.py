"""Initial local CLI. Every run uses a new directory and retains failure evidence."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback

from pipeline.cli_support import save_json
from pipeline.progress import ProgressWriter
from pipeline.sfm import CameraConfig, reconstruct as reconstruct_sfm
from pipeline.video_analysis import SelectionConfig, analyze_and_select


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract(video, out, fps=2.0, max_frames=160, min_sharpness=25.0):
    return analyze_and_select(Path(video), Path(out), SelectionConfig(
        candidate_fps=fps, max_frames=max_frames, min_sharpness=min_sharpness))


def reconstruct(out, camera_params='', exhaustive=False, fix_intrinsics=False):
    metrics = reconstruct_sfm(Path(out), CameraConfig(parameters=camera_params, fix_intrinsics=fix_intrinsics), exhaustive=exhaustive)
    selected = metrics['selected_images']
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
    progress = ProgressWriter(out)
    active_stage = 'Video analysis'
    try:
        print('Decoding and checking video frames...', flush=True)
        progress.update(active_stage, 0, 'Opening and validating the video')
        config = SelectionConfig(candidate_fps=args.fps, max_frames=args.max_frames,
                                 min_sharpness=args.min_sharpness,
                                 max_similarity=getattr(args, 'max_similarity', .985),
                                 min_motion_px=getattr(args, 'min_motion', 1.5),
                                 max_keyframe_gap_s=getattr(args, 'max_keyframe_gap', 2.0))
        rows = analyze_and_select(video, out, config, progress)
        manifest['selected_frames'] = sum(r['selected'] for r in rows)
        manifest['warnings'] = []
        if manifest['selected_frames'] == args.max_frames:
            manifest['warnings'].append('Keyframe budget reached; frames were spread across the clip, but finer scene changes may be omitted')
        if not args.prepare_only:
            active_stage = 'Feature extraction'
            camera = CameraConfig(model=getattr(args, 'camera_model', 'SIMPLE_RADIAL'),
                                  parameters=args.camera_params,
                                  fix_intrinsics=getattr(args, 'fix_intrinsics', False))
            manifest['metrics'] = reconstruct_sfm(out, camera, exhaustive=getattr(args, 'exhaustive', False), progress=progress)
            if manifest['metrics']['calibration_suspect']:
                manifest['warnings'].append('UNRELIABLE GEOMETRY: implausible focal length estimated. This point cloud is a diagnostic result, not a usable map.')
            if getattr(args, 'fix_intrinsics', False):
                manifest['warnings'].append('Camera intrinsics were fixed to the supplied values. This is an experimental result unless those values are independently verified.')
            if manifest['metrics']['registered_ratio'] < .8:
                manifest['warnings'].append('Partial reconstruction: fewer than 80% of selected frames registered. The model does not cover the entire clip reliably.')
            if manifest['metrics']['model_count'] > 1:
                manifest['warnings'].append('Multiple disconnected models were found. The viewer shows the model with the most registered images.')
        manifest['status'] = 'prepared' if args.prepare_only else 'complete'
        if args.prepare_only:
            progress.update('Keyframe selection', 1, 'Keyframes prepared', status='complete')
        else:
            active_stage = 'Quality report'
            progress.complete()
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = str(exc)
        (out / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        progress.fail(progress.last_stage, str(exc))
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
    command.add_argument('--max-similarity', type=float, default=.985, help='Reject near-duplicate candidates above this similarity')
    command.add_argument('--min-motion', type=float, default=1.5, help='Minimum median matched-feature displacement in analysis pixels')
    command.add_argument('--max-keyframe-gap', type=float, default=2.0, help='Keep a quality frame after this many seconds even with low motion')
    command.add_argument('--camera-model', default='SIMPLE_RADIAL', choices=['SIMPLE_PINHOLE','PINHOLE','SIMPLE_RADIAL','RADIAL','OPENCV'])
    command.add_argument('--camera-params', default='', help='Comma-separated parameters for the selected camera model, in source pixels')
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
