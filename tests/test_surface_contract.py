"""Exercise real Open3D export and fusion guards without model downloads."""
import json
from types import SimpleNamespace
import numpy as np
import pytest

o3d = pytest.importorskip('open3d')
trimesh = pytest.importorskip('trimesh')
from pipeline.open3d_refine import write_surface, run


def test_surface_export_preserves_dimensions_and_topology(tmp_path):
    mesh=o3d.geometry.TriangleMesh.create_box(width=2,height=3,depth=5)
    mesh.paint_uniform_color([.2,.5,.7])
    vertices,faces=write_surface(mesh,tmp_path)
    payload=json.loads((tmp_path/'surface_viewer.json').read_text())
    points=np.array(payload['positions']).reshape(-1,3)
    np.testing.assert_allclose(np.ptp(points,axis=0),[2,3,5])
    loaded=o3d.io.read_triangle_mesh(str(tmp_path/'surface_open3d.ply'))
    np.testing.assert_allclose(np.asarray(loaded.vertices),np.asarray(mesh.vertices))
    exported=trimesh.load(tmp_path/'surface.glb',force='scene')
    np.testing.assert_allclose(exported.extents,[2,3,5])
    assert vertices==8 and faces==12
    assert len(payload['indices'])==faces*3


def test_depth_scaling_fails_before_creating_output(tmp_path):
    with pytest.raises(ValueError,match='Depth-only scaling'):
        run(SimpleNamespace(depth_scale=2,source_run=str(tmp_path/'missing'),out=str(tmp_path/'out')))
    assert not (tmp_path/'out').exists()
