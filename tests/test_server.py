import json
import pytest
from fastapi.testclient import TestClient
from pipeline import server


def test_local_access_and_artifact_allowlist(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'OUTPUTS', tmp_path)
    (tmp_path/'synthetic').mkdir()
    (tmp_path/'synthetic/run_manifest.json').write_text('{"status":"complete"}')
    with TestClient(server.app) as client:
        assert client.get('/').status_code == 200
        assert client.get('/api/runs').status_code == 200
        assert client.get('/api/runs/synthetic/download/run_manifest.json').status_code == 200
        assert client.get('/api/runs/synthetic/download/database.db').status_code == 404
        assert client.get('/api/runs',headers={'host':'evil.example'}).status_code == 403
        assert client.post('/api/runs',headers={'X-AeroRecon':'local','origin':'https://example.org'}).status_code == 403
        assert client.post('/api/runs').status_code == 403


def test_reject_invalid_upload():
    with TestClient(server.app) as client:
        result=client.post('/api/runs',headers={'X-AeroRecon':'local'},files={'file':('bad.txt',b'fake','text/plain')})
        assert result.status_code == 400


def test_cloud_matches_exported_geometry():
    if not (server.OUTPUTS/'synthetic/metrics.json').exists():
        pytest.skip('Generate the documented synthetic reconstruction before this integration check')
    import numpy as np
    import pycolmap
    with TestClient(server.app) as client:
        response=client.get('/api/runs/synthetic/cloud')
        assert response.status_code == 200
        cloud=response.json()
        metrics=json.loads((server.OUTPUTS/'synthetic/metrics.json').read_text())
        model=pycolmap.Reconstruction(server.OUTPUTS/'synthetic/sparse'/str(metrics['best_model_id']))
        assert cloud['total_points'] == model.num_points3D()
        assert len(cloud['positions']) == len(cloud['colors'])
        centres={im.name: im.projection_center() for im in model.images.values()}
        for camera in cloud['cameras']:
            np.testing.assert_allclose(camera['centre'],centres[camera['name']])
        assert cloud['units'] == 'arbitrary'


def test_upload_decodes_and_starts_worker(tmp_path, monkeypatch):
    from test_pipeline import video
    source = tmp_path/'source.mp4'
    video(source)
    output = tmp_path/'outputs'
    output.mkdir()
    monkeypatch.setattr(server,'ROOT',tmp_path)
    monkeypatch.setattr(server,'OUTPUTS',output)
    monkeypatch.setattr(server,'processes',{})
    launched=[]
    monkeypatch.setattr(server,'start_worker',lambda *args: launched.append(args))
    with TestClient(server.app) as client:
        response=client.post('/api/runs',headers={'X-AeroRecon':'local'},files={'file':('my-video.mp4',source.read_bytes(),'video/mp4')})
        assert response.status_code == 202
        assert len(launched)==1
        assert launched[0][1].read_bytes()==source.read_bytes()
        assert launched[0][2]=='my-video.mp4'
        bad=client.post('/api/runs',headers={'X-AeroRecon':'local'},files={'file':('corrupt.mp4',b'not a video','video/mp4')})
        assert bad.status_code==400
        assert len(list((tmp_path/'data/uploads').iterdir()))==1


def test_interrupted_worker_does_not_block_new_runs(tmp_path):
    (tmp_path/'run_manifest.json').write_text(json.dumps({'status':'running','pid':99999999,'started_at':0}))
    assert not server.describe_unlocked_running(tmp_path)


def test_same_video_camera_selection_rejects_bad_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(server,'OUTPUTS',tmp_path)
    for name,digest,bad in [('upload','same',True),('good','same',False),('other','different',False)]:
        p=tmp_path/name;p.mkdir();(p/'sparse/0').mkdir(parents=True)
        (p/'run_manifest.json').write_text(json.dumps({'status':'complete','input_sha256':digest}))
        (p/'metrics.json').write_text(json.dumps({'registered_ratio':1,'calibration_suspect':bad,'best_model_id':0}))
    assert server.camera_source(tmp_path/'upload')==tmp_path/'good'


def test_repeat_upload_reuses_exact_completed_video(tmp_path, monkeypatch):
    import hashlib
    from test_pipeline import video
    source=tmp_path/'source.mp4';video(source)
    output=tmp_path/'outputs';existing=output/'existing';existing.mkdir(parents=True)
    (existing/'surface.glb').write_bytes(b'test artifact')
    (existing/'run_manifest.json').write_text(json.dumps({'status':'complete','input_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}))
    monkeypatch.setattr(server,'ROOT',tmp_path);monkeypatch.setattr(server,'OUTPUTS',output);monkeypatch.setattr(server,'processes',{})
    monkeypatch.setattr(server,'start_worker',lambda *args:pytest.fail('Duplicate should not launch another worker'))
    with TestClient(server.app) as client:
        r=client.post('/api/runs',headers={'X-AeroRecon':'local'},files={'file':('same.mp4',source.read_bytes(),'video/mp4')})
    assert r.status_code==202 and r.json()['reused'] and r.json()['id']=='existing'
    assert not list((tmp_path/'data/uploads').iterdir())
