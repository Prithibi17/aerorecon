"""Evaluate held-out synthetic camera centres after alignment on alternating frames."""
import csv
import json
from pathlib import Path
import numpy as np


def evaluate(truth_path, run):
    truth = json.loads(truth_path.read_text())
    rows = list(csv.DictReader((run/'camera_centres.csv').open()))
    src = np.array([[float(row[k]) for k in ('x','y','z')] for row in rows])
    dst = np.array([truth['poses'][int(Path(row['image']).stem.split('_')[1])]['centre'] for row in rows])
    train = np.arange(len(rows)) % 2 == 0
    a, b = src[train], dst[train]
    x, y = a-a.mean(0), b-b.mean(0)
    u, singular, vt = np.linalg.svd(y.T@x/len(x))
    sign = np.eye(3)
    sign[-1,-1] = np.linalg.det(u@vt)
    rotation = u@sign@vt
    scale = np.sum(singular*np.diag(sign))/np.mean(np.sum(x*x,axis=1))
    translation = b.mean(0)-scale*rotation@a.mean(0)
    errors = np.linalg.norm(scale*(src@rotation.T)+translation-dst,axis=1)
    held = errors[~train]
    result = dict(synthetic_only=True, alignment_frames=int(train.sum()), held_out_frames=int((~train).sum()),
                  held_out_camera_rmse_scene_units=float(np.sqrt(np.mean(held**2))),
                  held_out_camera_max_error_scene_units=float(held.max()),
                  warning='Ideal synthetic camera trajectory test; not real drone or surface accuracy validation.')
    (run/'synthetic_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if len(rows) < 28 or result['held_out_camera_rmse_scene_units'] > .1:
        raise SystemExit('Synthetic regression gate failed')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('truth',type=Path)
    p.add_argument('run',type=Path)
    a = p.parse_args()
    evaluate(a.truth,a.run)
