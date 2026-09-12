import csv
from pathlib import Path
from types import SimpleNamespace
import av
import numpy as np
import pytest
from pipeline.cli import extract, run


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
    assert [r['source_frame'] for r in rows] == [0,5,10]
    assert [r['time_s'] for r in rows] == pytest.approx([0,.5,1])
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
