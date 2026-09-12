"""Optional DA3 Small dense preview. Every generated surface is AI-inferred."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def save(path, data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def project_depth(depth, intrinsic, extrinsic):
    """Backproject camera z-depth with OpenCV world-to-camera conventions."""
    h,w=depth.shape
    y,x=np.indices((h,w))
    rays=np.stack([x,y,np.ones_like(x)],-1)@np.linalg.inv(intrinsic).T
    camera=rays*depth[...,None]
    return (camera-extrinsic[:3,3])@extrinsic[:3,:3]


def run(args):
    source_run=Path(args.source_run).resolve(strict=True)
    output=Path(args.out).resolve()
    if args.frames<2 or args.frames>24 or args.resolution<140 or args.resolution>504:
        raise ValueError('Use 2–24 frames and resolution 140–504 for this laptop profile')
    if not 0<=args.exclude_top<.8:
        raise ValueError('Exclude-top must be in [0, 0.8)')
    originals=sorted((source_run/'keyframes').glob('*.jpg'))
    if len(originals)<2:
        raise ValueError('Prepare video keyframes first')
    indices=np.unique(np.linspace(0,len(originals)-1,min(args.frames,len(originals))).astype(int))
    frames=[originals[i] for i in indices]
    source=json.loads((source_run/'run_manifest.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    (output/'keyframes').mkdir()
    start=time.time()
    manifest=dict(schema='aerorecon.run/v1',status='running',pid=os.getpid(),started_at=start,
                  input=source['input'],input_sha256=source['input_sha256'],display_name=f'AI terrain · {len(frames)} views',
                  engine='da3-small',units='arbitrary',georeferenced=False,geometry_provenance='inferred',
                  configuration=vars(args),warnings=[
                      'AI-INFERRED GEOMETRY: depth and camera poses are predicted by a model. Not validated for measurement.',
                      f'Top {args.exclude_top:.0%} of image height excluded from geometry, as configured.'])
    save(output/'run_manifest.json',manifest)
    try:
        for frame in frames:
            shutil.copy2(frame,output/'keyframes'/frame.name)
        with (source_run/'frames.csv').open() as f:
            rows=list(csv.DictReader(f))
        names={frame.name for frame in frames}
        with (output/'frames.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader()
            writer.writerows(row for row in rows if row['image'] in names)
        print('Loading Depth Anything 3 Small on GPU...',flush=True)
        import torch
        from depth_anything_3.api import DepthAnything3
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in the AI environment')
        torch.set_num_threads(4)
        checkpoint=ROOT/'models/DA3-SMALL'
        model=DepthAnything3.from_pretrained(str(checkpoint)).to('cuda').eval()
        torch.cuda.reset_peak_memory_stats()
        print(f'Predicting multi-view depth for {len(frames)} images...',flush=True)
        with torch.inference_mode():
            prediction=model.inference([str(p) for p in frames],process_res=args.resolution,
                                       use_ray_pose=True,ref_view_strategy='middle')
        depth=prediction.depth.astype(np.float32)
        confidence=prediction.conf.astype(np.float32)
        intrinsics=prediction.intrinsics.astype(np.float32)
        extrinsics=prediction.extrinsics.astype(np.float32)
        images=prediction.processed_images
        np.savez_compressed(output/'depth_evidence.npz',depth=depth,confidence=confidence,
                            intrinsics=intrinsics,extrinsics=extrinsics,images=images)
        print('Backprojecting dense AI geometry...',flush=True)
        points=[];colors=[];cameras=[];surfaces=[]
        for i,(d,k,e,rgb) in enumerate(zip(depth,intrinsics,extrinsics,images)):
            xyz=project_depth(d,k,e)
            h,w=d.shape
            valid=np.isfinite(d)&(d>0)&np.isfinite(confidence[i])&np.isfinite(xyz).all(-1)
            valid[:int(h*args.exclude_top)]=False
            if valid.sum()<10:
                raise ValueError('No usable predicted depths')
            cutoff=np.percentile(confidence[i][valid],35)
            far=np.percentile(d[valid],97)
            valid &= (confidence[i]>=cutoff)&(d<=far)
            points.append(xyz[valid]);colors.append(rgb[valid])
            centre=(-e[:3,:3].T@e[:3,3]).tolist()
            cameras.append(dict(name=frames[i].name,centre=centre))
            surfaces.append((xyz,rgb,valid,d))
        points=np.concatenate(points).astype(np.float32);colors=np.concatenate(colors).astype(np.uint8)
        if len(points)<100:
            raise ValueError('Too little usable AI geometry')
        # A depth surface from the middle view has actual image color at each vertex.
        # No unseen surfaces or closing caps are added.
        xyz,rgb,valid,d=surfaces[len(surfaces)//2]
        h,w=valid.shape
        a=np.arange((h-1)*w).reshape(h-1,w)[:,:-1].ravel()
        b=a+1;c=a+w;dd=c+1
        faces=np.concatenate([np.stack([a,c,b],1),np.stack([b,c,dd],1)])
        flat_valid=valid.ravel();flat_depth=d.ravel()
        good=flat_valid[faces].all(1)
        z=flat_depth[faces]
        good &= (z.max(1)-z.min(1)) < .08*np.maximum(z.mean(1),1e-6)
        faces=faces[good]
        import trimesh
        mesh=trimesh.Trimesh(vertices=xyz.reshape(-1,3),faces=faces,vertex_colors=rgb.reshape(-1,3),process=False)
        mesh.export(output/'surface.glb')
        save(output/'surface_viewer.json',dict(positions=xyz.reshape(-1).tolist(),colors=rgb.reshape(-1).tolist(),indices=faces.ravel().tolist(),geometry_provenance='inferred'))
        trimesh.points.PointCloud(points,colors=colors).export(output/'dense.ply')
        # Viewer consumes common JSON instead of pretending AI output is a COLMAP model.
        take=np.linspace(0,len(points)-1,min(len(points),150000)).astype(int)
        viewer=dict(positions=points[take].ravel().tolist(),colors=colors[take].ravel().tolist(),
                    cameras=cameras,total_points=len(points),displayed_points=len(take),units='arbitrary',
                    geometry_provenance='inferred',surface_available=True)
        save(output/'viewer.json',viewer)
        with (output/'camera_centres.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['image','x','y','z'])
            writer.writerows([camera['name'],*camera['centre']] for camera in cameras)
        weights=checkpoint/'model.safetensors'
        with weights.open('rb') as stream:
            checksum=hashlib.file_digest(stream,'sha256').hexdigest()
        metrics=dict(engine='da3-small',predicted_views=len(frames),selected_images=len(frames),
                     points=len(points),surface_triangles=len(faces),geometry_provenance='inferred',
                     geometry_validated=False,units='arbitrary',georeferenced=False,
                     gpu_peak_memory_mb=torch.cuda.max_memory_allocated()/1024**2,
                     model='depth-anything/DA3-SMALL',license='Apache-2.0',
                     checkpoint_sha256=checksum,model_revision=(checkpoint/'revision.txt').read_text().strip(),
                     source_commit=subprocess.check_output(['git','-C',str(ROOT/'third_party/Depth-Anything-3'),'rev-parse','HEAD'],text=True).strip())
        save(output/'metrics.json',metrics)
        (output/'REPORT.md').write_text(
            '# AI dense reconstruction\n\n'
            f'Depth Anything 3 Small predicted {len(frames)} views and {len(points):,} retained points.\n\n'
            'All geometry is AI-inferred, relative-scale and unvalidated. No metres, GPS position, or accuracy claim.\n\n'
            'The surface is a triangulated predicted depth map from the middle input view. '
            'The point cloud combines all selected views. They are different products; neither is a validated survey mesh.\n\n'
            f'Top-image exclusion: {args.exclude_top:.0%}. Confidence bottom 35% and furthest depth 3% excluded per view. '
            'Depth discontinuities are excluded from surface triangles. Raw predictions remain in depth_evidence.npz.\n',encoding='utf-8')
        manifest['status']='complete';manifest['metrics']=metrics
        print(f'AI preview complete: {len(points):,} points, {len(faces):,} surface triangles.',flush=True)
    except BaseException as error:
        manifest['status']='failed';manifest['error']=str(error)
        (output/'error.log').write_text(traceback.format_exc())
        raise
    finally:
        manifest['elapsed_s']=time.time()-start
        save(output/'run_manifest.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source_run')
    parser.add_argument('--out',required=True)
    parser.add_argument('--frames',type=int,default=12)
    parser.add_argument('--resolution',type=int,default=336)
    parser.add_argument('--exclude-top',type=float,default=0)
    run(parser.parse_args())
