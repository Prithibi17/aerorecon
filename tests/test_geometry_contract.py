import json
import numpy as np
import pytest
from pipeline.geometry_contract import rigid_matrix, reconstruction_digest, mesh_coordinates
from pipeline.georeference import similarity


def scene(tmp_path):
    dense = tmp_path/'dense'
    (dense/'workspace/sparse').mkdir(parents=True)
    (dense/'workspace/sparse/cameras.bin').write_bytes(b'camera-model-a')
    (dense/'run_manifest.json').write_text(json.dumps({'input_sha256':'video-a'}))
    mesh = {'input_sha256':'video-a','configuration':{'source_run':str(dense)},
            'reconstruction_sha256':reconstruction_digest(dense),'raw_to_world':np.eye(4).tolist()}
    return dense, mesh


def test_alignment_keeps_image_projection_and_distances():
    rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    transform=rigid_matrix(rotation,[3,4,1])
    points=np.array([[1,2,9,1],[4,3,7,1]],float)
    view=np.eye(4);view[:3,3]=[.2,.3,1]
    aligned=points@transform.T
    updated=view@np.linalg.inv(transform)
    np.testing.assert_allclose(aligned@updated.T,points@view.T)
    assert np.linalg.norm(aligned[0,:3]-aligned[1,:3])==pytest.approx(np.linalg.norm(points[0,:3]-points[1,:3]))


@pytest.mark.parametrize('rotation',[np.eye(3)*2,np.diag([1,1,-1]),np.full((3,3),np.nan)])
def test_nonrigid_alignment_rejected(rotation):
    with pytest.raises(ValueError): rigid_matrix(rotation,[0,0,0])


def test_same_video_different_reconstruction_rejected(tmp_path):
    dense,mesh=scene(tmp_path)
    (dense/'workspace/sparse/cameras.bin').write_bytes(b'camera-model-b')
    with pytest.raises(ValueError,match='different reconstructions'):mesh_coordinates(dense,mesh)


def test_saved_transform_survives_folder_move(tmp_path):
    dense,mesh=scene(tmp_path)
    destination=tmp_path/'moved'
    dense.rename(destination)
    matrix,digest=mesh_coordinates(destination,mesh)
    np.testing.assert_allclose(matrix,np.eye(4))
    assert digest==mesh['reconstruction_sha256']


def test_depth_only_scale_rejected(tmp_path):
    dense,mesh=scene(tmp_path)
    mesh['configuration']['depth_scale']=2
    with pytest.raises(ValueError,match='Depth-only'):mesh_coordinates(dense,mesh)


def test_missing_legacy_transform_is_not_identity(tmp_path):
    dense,mesh=scene(tmp_path)
    del mesh['raw_to_world']
    mesh['configuration']['ground_transform']=str(tmp_path/'missing.json')
    with pytest.raises(FileNotFoundError):mesh_coordinates(dense,mesh)


def test_different_video_rejected(tmp_path):
    dense,mesh=scene(tmp_path)
    mesh['input_sha256']='video-b'
    with pytest.raises(ValueError,match='same video'):mesh_coordinates(dense,mesh)


@pytest.mark.parametrize('target',[np.zeros((4,3)),np.full((4,3),np.nan),np.ones((3,3))])
def test_bad_metric_references_rejected(target):
    with pytest.raises(ValueError):similarity(np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]),target)


def test_server_selects_matching_camera_model_instead_of_newest(tmp_path, monkeypatch):
    from pipeline import server
    dense,mesh=scene(tmp_path)
    (dense/'run_manifest.json').write_text(json.dumps({'input_sha256':'video-a','status':'complete','engine':'colmap-mvs'}))
    other=tmp_path/'newer'
    (other/'workspace/sparse').mkdir(parents=True)
    (other/'workspace/sparse/cameras.bin').write_bytes(b'another-solution')
    (other/'run_manifest.json').write_text((dense/'run_manifest.json').read_text())
    surface=tmp_path/'mesh';surface.mkdir()
    (surface/'run_manifest.json').write_text(json.dumps(mesh))
    monkeypatch.setattr(server,'OUTPUTS',tmp_path)
    assert server.matching_dense(surface)==dense
