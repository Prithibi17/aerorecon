import numpy as np
import pytest

pytest.importorskip('open3d')
pytest.importorskip('transformers')
from pipeline.complete_scene import closed_back, fit_ground


def test_sloping_ground_is_rigidly_aligned_without_changing_distances():
    rng=np.random.default_rng(1)
    xz=rng.uniform(-2,2,(800,2))
    points=np.column_stack([xz[:,0],.2*xz[:,0]-.1*xz[:,1]+.3,xz[:,1]])
    rotation,origin,support=fit_ground(points)
    aligned=(points-origin)@rotation.T
    assert np.max(np.abs(aligned[:,1]))<1e-6
    assert np.linalg.det(rotation)==pytest.approx(1)
    assert np.linalg.norm(aligned[0]-aligned[1])==pytest.approx(np.linalg.norm(points[0]-points[1]))
    assert support>.99


def test_open_front_gets_a_closed_grounded_back_with_local_colours():
    points=np.array([[0,.1,0],[0,.6,0],[1,.4,0],[1,.8,1],[0,.5,1]])
    mesh=closed_back(points,np.tile([30,110,60],(5,1)))
    assert mesh.is_watertight
    assert mesh.vertices[:,1].min()==0
    assert mesh.volume>0
    assert np.all(np.asarray(mesh.visual.vertex_colors)[:,:3]==[30,110,60])
