"""Shared coordinate and provenance checks between reconstruction stages."""
import hashlib
import json
from pathlib import Path
import numpy as np


def rigid_matrix(rotation, origin):
    rotation, origin = np.asarray(rotation, float), np.asarray(origin, float)
    if rotation.shape != (3,3) or origin.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(origin).all():
        raise ValueError('Alignment needs a finite 3x3 rotation and 3-vector origin')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError('Alignment must be a proper rigid rotation; scale, shear and reflection are forbidden')
    matrix = np.eye(4)
    matrix[:3,:3], matrix[:3,3] = rotation, -rotation @ origin
    return matrix


def reconstruction_digest(dense):
    """Independent of folder location; different reconstructions of one video differ."""
    folder = Path(dense)/'workspace'/'sparse'
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in {'.bin','.txt'})
    if not files:
        raise ValueError('Missing calibrated sparse model for provenance check')
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
    return digest.hexdigest()


def mesh_coordinates(dense, mesh_manifest):
    dense = Path(dense).resolve()
    manifest = json.loads((dense/'run_manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('input_sha256') or manifest['input_sha256'] != mesh_manifest.get('input_sha256'):
        raise ValueError('Mesh and dense cameras must come from the same video')
    config = mesh_manifest.get('configuration', {})
    if config.get('depth_scale', 1) != 1:
        raise ValueError('Depth-only scaling is incompatible with calibrated cameras; rebuild with depth_scale=1')
    digest = reconstruction_digest(dense)
    if mesh_manifest.get('reconstruction_sha256'):
        if mesh_manifest['reconstruction_sha256'] != digest:
            raise ValueError('Mesh and cameras use different reconstructions of this video')
    elif Path(config.get('source_run', '')).resolve() != dense:
        raise ValueError('Legacy mesh provenance cannot be verified; regenerate the mesh from this dense run')
    saved = mesh_manifest.get('raw_to_world')
    if saved is not None:
        matrix = np.asarray(saved, float)
        if matrix.shape != (4,4) or not np.allclose(matrix[3], [0,0,0,1]):
            raise ValueError('Invalid saved coordinate transform')
        checked = rigid_matrix(matrix[:3,:3], -matrix[:3,:3].T @ matrix[:3,3])
        return checked, digest
    path = config.get('ground_transform')
    if path:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return rigid_matrix(data['rotation'], data['origin']), digest
    if mesh_manifest.get('metrics', {}).get('ground_aligned'):
        raise ValueError('Ground-aligned mesh is missing its coordinate transform')
    return np.eye(4), digest
