"""Backfill calibration warnings on older artifacts without changing geometry."""
import json
from pathlib import Path
import pycolmap

for path in Path('outputs').iterdir():
    metrics_path=path/'metrics.json'
    if not metrics_path.exists():
        continue
    metrics=json.loads(metrics_path.read_text())
    if 'best_model_id' not in metrics:
        continue
    model=pycolmap.Reconstruction(path/'sparse'/str(metrics['best_model_id']))
    metrics['cameras']=[dict(model=c.model_name,parameters=c.params.tolist(),width=c.width,height=c.height,
                             focal_ratio=float(c.focal_length/max(c.width,c.height))) for c in model.cameras.values()]
    metrics['calibration_suspect']=any(c['focal_ratio']<.2 or c['focal_ratio']>5 for c in metrics['cameras'])
    metrics['geometry_validated']=False
    metrics_path.write_text(json.dumps(metrics,indent=2))
    manifest_path=path/'run_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    warning='UNRELIABLE GEOMETRY: implausible focal length estimated. This point cloud is a diagnostic result, not a usable map.'
    if metrics['calibration_suspect']:
        warnings=manifest.setdefault('warnings',[])
        if warning not in warnings:
            warnings.insert(0,warning)
            with (path/'REPORT.md').open('a') as report:
                report.write('\n\n'+warning+'\n')
    manifest['metrics']=metrics
    manifest_path.write_text(json.dumps(manifest,indent=2))
    print(path.name, 'suspect=',metrics['calibration_suspect'])
