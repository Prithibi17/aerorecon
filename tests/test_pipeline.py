import csv
import json
from pathlib import Path
from types import SimpleNamespace
import av
import numpy as np
import pytest
from pipeline.cli import extract, run
from pipeline.progress import ProgressWriter
from pipeline.sfm import CameraConfig
from pipeline.video_analysis import SelectionConfig, analyze_and_select


def video(path):
    rng = np.random.default_rng(3)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mpeg4', rate=10)
        stream.width, stream.height = 320, 240
        stream.pix_fmt = 'yuv420p'
        for _ in range(20):
            frame = av.VideoFrame.from_ndarray(rng.integers(30,220,(240,320,3),dtype=np.uint8), format='bgr24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_extraction_timestamps_and_cap(tmp_path):
    source = tmp_path/'input.mp4'
    video(source)
    out = tmp_path/'run'
    out.mkdir()
    rows = extract(source,out,fps=2,max_frames=3)
    assert [r['source_frame'] for r in rows] == [0,7,14]
    assert [r['time_s'] for r in rows] == pytest.approx([0,.7,1.4])
    assert len(list((out/'keyframes').glob('*.jpg'))) == 3
    assert all(r['selected'] for r in rows)


@pytest.mark.parametrize('fps',[0,-1,float('nan')])
def test_invalid_fps(tmp_path, fps):
    with pytest.raises(ValueError):
        extract(tmp_path/'missing',tmp_path,fps=fps)


def test_failed_run_retains_evidence(tmp_path):
    import json
    source = tmp_path/'bad.mp4'
    source.write_bytes(b'not a video')
    args = SimpleNamespace(video=str(source),out=str(tmp_path/'run'),fps=2,max_frames=20,min_sharpness=25,prepare_only=False,camera_params='')
    with pytest.raises(Exception):
        run(args)
    manifest = json.loads((tmp_path/'run/run_manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert (tmp_path/'run/error.log').exists()
    with pytest.raises(FileExistsError):
        run(args)


def test_content_aware_selection_rejects_duplicate_frames(tmp_path):
    source = tmp_path/'duplicate.mp4'
    checker = ((np.indices((240,320)).sum(0)//12)%2*180+35).astype(np.uint8)
    image = np.repeat(checker[...,None],3,axis=2)
    with av.open(str(source), 'w') as container:
        stream = container.add_stream('mpeg4', rate=10)
        stream.width, stream.height, stream.pix_fmt = 320, 240, 'yuv420p'
        for _ in range(35):
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format='bgr24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    out=tmp_path/'selection';out.mkdir()
    rows=analyze_and_select(source,out,SelectionConfig(candidate_fps=5,max_frames=10,
        min_features=20,max_keyframe_gap_s=.5))
    assert any(row['reason']=='redundant' for row in rows)
    assert (out/'keyframe_contact_sheet.jpg').is_file()
    assert (out/'keyframe_selection.json').is_file()


def test_progress_is_machine_readable(tmp_path):
    writer=ProgressWriter(tmp_path)
    writer.update('Frame matching',.5,'Matching',current=2,total=4)
    state=json.loads((tmp_path/'progress.json').read_text())
    assert state['stage']=='Frame matching'
    assert state['stage_index']==3
    assert state['current']==2 and state['total']==4


@pytest.mark.parametrize(('model','parameters'),[
    ('SIMPLE_RADIAL','1000,640,360'),
    ('UNKNOWN','1000,640,360,0'),
    ('PINHOLE','-1,1000,640,360'),
])
def test_camera_configuration_validation(model,parameters):
    with pytest.raises(ValueError):
        CameraConfig(model=model,parameters=parameters).validate()
