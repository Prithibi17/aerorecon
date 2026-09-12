"""Render textured planes at multiple depths with exact pinhole projection."""
from pathlib import Path
import json
import cv2
import numpy as np


def generate(out):
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(17)
    width, height, focal = 800, 600, 700.
    planes = [(-9, -6, 9, 6, 18), (-5, -3, -1, 3, 11), (1, -2, 5, 2, 9)]
    textures = []
    for _ in planes:
        tex = rng.integers(35, 220, (512, 512, 3), dtype=np.uint8)
        tex = cv2.GaussianBlur(tex, (5, 5), 0)
        for i in range(300):
            x, y = rng.integers(0, 490, 2)
            color = tuple(int(c) for c in rng.integers(20, 235, 3))
            cv2.rectangle(tex, (x, y), (x+int(rng.integers(4, 22)), y+int(rng.integers(4, 22))), color, -1)
        textures.append(tex)
    writer = cv2.VideoWriter(str(out/'scene.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 4, (width, height))
    if not writer.isOpened():
        raise RuntimeError('MP4 encoder unavailable')
    poses = []
    try:
        for index, xcam in enumerate(np.linspace(-3, 3, 32)):
            ycam = .3*np.sin(index/31*np.pi)
            image = np.full((height, width, 3), 100, np.uint8)
            for plane, tex in zip(planes, textures):
                x0, y0, x1, y1, z = plane
                corners = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], np.float32)
                corners = (corners - np.array([xcam,ycam]))*focal/z + np.array([width/2,height/2])
                transform = cv2.getPerspectiveTransform(np.array([[0,0],[511,0],[511,511],[0,511]],np.float32), corners.astype(np.float32))
                warped = cv2.warpPerspective(tex, transform, (width,height))
                mask = cv2.warpPerspective(np.full((512,512),255,np.uint8), transform, (width,height))
                image[mask>250] = warped[mask>250]
            writer.write(image)
            poses.append(dict(frame=index, time_s=index/4, centre=[float(xcam),float(ycam),0]))
    finally:
        writer.release()
    (out/'ground_truth.json').write_text(json.dumps(dict(synthetic=True, camera_params=[focal,width/2,height/2,0], poses=poses, planes=planes),indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    generate(parser.parse_args().out)
