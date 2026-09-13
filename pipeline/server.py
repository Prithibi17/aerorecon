"""Loopback-only application serving local reconstruction jobs and artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'outputs'
WEB = ROOT / 'apps' / 'web'
app = FastAPI(title='AeroRecon AI')
processes = {}
lock = threading.Lock()


@app.middleware('http')
async def local_access(request: Request, call_next):
    # Prevent browser DNS rebinding and cross-origin requests to the local worker.
    from starlette.responses import JSONResponse
    if request.url.hostname not in {'127.0.0.1', 'localhost', 'testserver'}:
        return JSONResponse({'detail': 'Local access only'}, status_code=403)
    if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
        origin = request.headers.get('origin')
        if origin and origin != f'{request.url.scheme}://{request.url.netloc}':
            return JSONResponse({'detail': 'Cross-origin request rejected'}, status_code=403)
        if request.headers.get('x-aerorecon') != 'local':
            return JSONResponse({'detail': 'Application header required'}, status_code=403)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


def run_dir(identifier):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', identifier):
        raise HTTPException(404, 'Run not found')
    path = OUTPUTS / identifier
    if not path.is_dir():
        raise HTTPException(404, 'Run not found')
    return path


def read_json(path, fallback=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback or {}


def camera_source(source):
    """Find a usable camera reconstruction of the exact same video bytes."""
    digest = read_json(source/'run_manifest.json').get('input_sha256')
    if not digest:
        return None
    candidates=[]
    for path in OUTPUTS.iterdir():
        manifest=read_json(path/'run_manifest.json')
        metrics=read_json(path/'metrics.json')
        if (manifest.get('status')=='complete' and manifest.get('input_sha256')==digest
                and metrics.get('registered_ratio',0)>.95 and not metrics.get('calibration_suspect',True)
                and (path/'sparse'/str(metrics.get('best_model_id'))).is_dir()):
            candidates.append(path)
    return max(candidates,key=lambda p:p.stat().st_mtime) if candidates else None


def log_path(path):
    own = path / 'worker.log'
    return own if own.exists() else ROOT / f'{path.name}.log'


def log_tail(path):
    log = log_path(path)
    if not log.exists():
        return ''
    with log.open('rb') as stream:
        stream.seek(max(0, log.stat().st_size - 12000))
        return stream.read().decode('utf-8', errors='replace')


def describe(path):
    manifest = read_json(path/'run_manifest.json')
    job = read_json(path/'job.json')
    metrics = read_json(path/'metrics.json')
    progress = read_json(path/'progress.json')
    if manifest.get('fusion'):
        metrics = {'engine': manifest.get('engine', 'da3-small'), 'fusion': manifest['fusion'], **read_json(path/'progress.json'), **metrics}
    status = manifest.get('status', job.get('status', 'queued'))
    tail = log_tail(path)
    stage = progress.get('stage') or 'Checking video'
    if not progress and ('Extracting SIFT' in tail or 'feature_extraction' in tail):
        stage = 'Finding visual features'
    if not progress and ('Matching sequential' in tail or 'pairing.cc' in tail):
        stage = 'Matching frames'
    if not progress and ('Recovering cameras' in tail or 'incremental_pipeline' in tail):
        stage = 'Reconstructing 3D'
    if status == 'complete':
        stage = ('Photogrammetric dense map ready' if manifest.get('engine') == 'colmap-mvs'
                 else 'AI dense preview ready' if manifest.get('engine') == 'da3-small' else 'Sparse model ready')
    elif status == 'failed':
        stage = 'Processing failed'
    elif manifest.get('engine') == 'da3-small':
        stage = 'Predicting AI geometry'
    if manifest.get('fusion') and status in {'running', 'complete'}:
        stage = 'Full-video surface ready' if status == 'complete' else f"Fusing {metrics.get('frames_integrated', 0)} / {metrics.get('frames_decoded', '…')} frames"
        if status=='running' and metrics.get('stage')=='Preparing surface for viewing':
            stage='Preparing surface for viewing'
    with lock:
        proc = processes.get(path.name)
    if proc is not None and proc.poll() is not None and status not in {'complete', 'failed'}:
        status, stage = 'failed', 'Worker stopped before completion'
    if status == 'running' and manifest.get('pid'):
        import psutil
        try:
            alive = psutil.Process(manifest['pid']).create_time() <= manifest.get('started_at', 0)
        except psutil.Error:
            alive = False
        if not alive:
            status, stage = 'failed', 'Processing was interrupted'
    name = manifest.get('display_name') or job.get('name') or Path(manifest.get('input', path.name)).name
    warnings = list(manifest.get('warnings', []))
    if metrics.get('registered_ratio', 1) < .8 and not any('Partial reconstruction' in w for w in warnings):
        warnings.append('Partial reconstruction: only part of the video is represented. This is not a complete map of the area.')
    if metrics.get('model_count', 0) > 1 and not any('disconnected' in w for w in warnings):
        warnings.append('Disconnected models were found. Showing the largest by registered frame count.')
    return dict(id=path.name, name=name, status=status, stage=stage, progress=progress,
                metrics=metrics, elapsed_s=manifest.get('elapsed_s'),
                error=manifest.get('error'), warnings=warnings,
                synthetic=path.name == 'synthetic', created=path.stat().st_ctime)


@app.get('/api/runs')
def runs():
    return sorted([describe(p) for p in OUTPUTS.iterdir() if p.is_dir() and
                   ((p/'run_manifest.json').exists() or (p/'job.json').exists())],
                  key=lambda x: x['created'], reverse=True)


@app.get('/api/runs/{identifier}')
def get_run(identifier: str):
    return describe(run_dir(identifier))


@app.get('/api/runs/{identifier}/logs')
def logs(identifier: str):
    return {'text': log_tail(run_dir(identifier))}


@app.get('/api/runs/{identifier}/frames')
def frames(identifier: str):
    path = run_dir(identifier)
    images = sorted((path/'keyframes').glob('*.jpg'))
    return [dict(name=p.name, url=f'/api/runs/{identifier}/frames/{p.name}') for p in images]


@app.get('/api/runs/{identifier}/frames/{name}')
def frame(identifier: str, name: str):
    if not re.fullmatch(r'frame_\d+\.jpg', name):
        raise HTTPException(404)
    path = run_dir(identifier)/'keyframes'/name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@app.get('/api/runs/{identifier}/video')
def video(identifier: str):
    path = run_dir(identifier)
    manifest = read_json(path/'run_manifest.json')
    job = read_json(path/'job.json')
    source = Path(manifest.get('input') or job.get('input') or '')
    if not source.is_file():
        raise HTTPException(404, 'Source video is unavailable')
    return FileResponse(source, media_type='video/mp4')


@app.get('/api/runs/{identifier}/download/{name}')
def download(identifier: str, name: str):
    allowed = {'sparse.ply', 'dense.ply', 'dense_raw.ply', 'mesh_raw.ply', 'surface.glb', 'surface_full.ply', 'texture.png', 'depth_evidence.npz', 'camera_centres.csv', 'frames.csv', 'metrics.json', 'REPORT.md', 'run_manifest.json', 'video_analysis.json', 'keyframe_selection.json', 'keyframe_contact_sheet.jpg', 'camera_configuration.json'}
    if name not in allowed:
        raise HTTPException(404)
    path = run_dir(identifier)/name
    if not path.is_file():
        raise HTTPException(404, 'Artifact is not ready')
    return FileResponse(path, filename=name)


@app.get('/api/runs/{identifier}/surface')
def surface(identifier: str):
    path = run_dir(identifier)/'surface_viewer.json'
    if not path.is_file():
        raise HTTPException(404, 'No surface is available')
    return FileResponse(path, media_type='application/json')


@app.get('/api/runs/{identifier}/cloud')
def cloud(identifier: str):
    path = run_dir(identifier)
    info = describe(path)
    if info['status'] != 'complete':
        raise HTTPException(409, 'Model is still processing')
    cache = path/'viewer.json'
    if cache.exists():
        return FileResponse(cache, media_type='application/json')
    import pycolmap
    import numpy as np
    model = pycolmap.Reconstruction(path/'sparse'/str(info['metrics']['best_model_id']))
    points = list(model.points3D.values())
    stride = max(1, int(np.ceil(len(points)/100000)))
    points = points[::stride]
    positions = np.array([p.xyz for p in points])
    # Preserve raw coordinates; viewer rebases and scales only the display.
    cameras = [dict(name=im.name, centre=im.projection_center().tolist())
               for im in sorted(model.images.values(), key=lambda im: im.name)]
    data = dict(positions=positions.ravel().tolist(), colors=np.array([p.color for p in points]).ravel().tolist(),
                cameras=cameras, total_points=model.num_points3D(), displayed_points=len(points), units='arbitrary')
    from pipeline.cli import save_json
    save_json(cache, data)
    return data


def start_worker(identifier, source, name):
    # CLI creates a fresh directory; worker log lives outside until startup.
    target = OUTPUTS/identifier
    with (ROOT/f'{identifier}.log').open('wb') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'pipeline.cli', 'run', str(source),
                                 '--out', str(target), '--fps', '2', '--max-frames', '100', '--name', name],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    processes[identifier] = proc


@app.post('/api/runs', status_code=202)
def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in {'.mp4', '.mov', '.m4v', '.avi', '.mkv'}:
        raise HTTPException(400, 'Choose an MP4, MOV, M4V, AVI or MKV video')
    identifier = 'run-' + uuid.uuid4().hex[:12]
    folder = ROOT/'data'/'uploads'
    folder.mkdir(parents=True, exist_ok=True)
    source = folder/(identifier+suffix)
    try:
        with lock:
            if any(p.poll() is None for p in processes.values()) or any(
                describe_unlocked_running(p)
                for p in OUTPUTS.iterdir() if p.is_dir()
            ):
                raise HTTPException(409, 'A reconstruction is already running. Wait for it to finish.')
            size = 0
            digest = hashlib.sha256()
            with source.open('xb') as stream:
                while chunk := file.file.read(1024*1024):
                    size += len(chunk)
                    if size > 1024**3:
                        raise HTTPException(413, 'Video limit is 1 GB')
                    stream.write(chunk)
                    digest.update(chunk)
            if not size:
                raise HTTPException(400, 'Video is empty')
            matches=[]
            for existing in OUTPUTS.iterdir():
                manifest=read_json(existing/'run_manifest.json')
                if manifest.get('status')=='complete' and manifest.get('input_sha256')==digest.hexdigest() and (existing/'surface.glb').is_file():
                    matches.append(existing)
            if matches:
                existing=max(matches,key=lambda p:p.stat().st_mtime)
                source.unlink()
                return {'id':existing.name,'status':'complete','reused':True,'name':read_json(existing/'run_manifest.json').get('display_name',existing.name)}
            import av
            try:
                with av.open(str(source)) as container:
                    next(container.decode(video=0))
            except Exception:
                raise HTTPException(400, 'The video cannot be decoded')
            start_worker(identifier, source, file.filename or source.name)
    except Exception:
        source.unlink(missing_ok=True)
        raise
    return {'id': identifier, 'name': file.filename, 'status': 'queued'}


@app.post('/api/runs/{identifier}/ai', status_code=202)
def build_ai(identifier: str):
    source = run_dir(identifier)
    if len(list((source/'keyframes').glob('*.jpg'))) < 2:
        raise HTTPException(409, 'Prepare at least two keyframes first')
    calibrated = camera_source(source)
    if calibrated is None:
        raise HTTPException(409, 'Camera calibration needs attention before dense reconstruction. Supply camera intrinsics or use a video with stronger overlapping views. No reliable camera run exists for this video.')
    source = calibrated
    executable = ROOT/'.venv-ai/Scripts/python.exe'
    if not executable.exists() or not (ROOT/'models/MoGe-2-Base/model.pt').exists():
        raise HTTPException(503, 'AI model setup is not complete')
    new_id = 'ai-' + uuid.uuid4().hex[:12]
    with lock:
        if any(p.poll() is None for p in processes.values()) or any(
            describe_unlocked_running(p) for p in OUTPUTS.iterdir() if p.is_dir()
        ):
            raise HTTPException(409, 'A reconstruction is already running. Wait for it to finish.')
        with (ROOT/f'{new_id}.log').open('wb') as log:
            processes[new_id] = subprocess.Popen(
                [str(executable), '-m', 'pipeline.semantic_fusion', str(source), '--out', str(OUTPUTS/new_id),
                 '--resolution', '336', '--flat-field', '--solid-objects'],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return {'id': new_id, 'status': 'queued'}


def describe_unlocked_running(path):
    manifest = read_json(path/'run_manifest.json')
    if manifest.get('status') != 'running':
        return False
    if not manifest.get('pid'):
        return True
    import psutil
    try:
        return psutil.Process(manifest['pid']).create_time() <= manifest.get('started_at', 0)
    except psutil.Error:
        return False


app.mount('/', StaticFiles(directory=WEB, html=True), name='web')
