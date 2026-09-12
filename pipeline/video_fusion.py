"""Full-video DA3 depth with shared anchors and multi-view TSDF fusion."""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from pipeline.ai_reconstruct import ROOT, save, project_depth


def similarity(source, target):
    """Robust proper Sim(3); maps row points as scale * points @ rotation.T + t."""
    keep = np.isfinite(source).all(1) & np.isfinite(target).all(1)
    for _ in range(4):
        x, y = source[keep], target[keep]
        xc, yc = x.mean(0), y.mean(0)
        u, singular, vt = np.linalg.svd((y-yc).T @ (x-xc) / len(x))
        sign = np.ones(3); sign[-1] = np.linalg.det(u @ vt)
        rotation = (u * sign) @ vt
        scale = (singular * sign).sum() / np.mean(np.sum((x-xc)**2, axis=1))
        translation = yc-scale*rotation@xc
        residual = np.linalg.norm(scale*source@rotation.T+translation-target, axis=1)
        keep = np.isfinite(residual) & (residual <= np.nanpercentile(residual, 75))
    return scale, rotation, translation, float(np.median(residual[keep]))


def run(args):
    import av
    import cv2
    import open3d as o3d
    import torch
    import trimesh
    from depth_anything_3.api import DepthAnything3
    source = json.loads((Path(args.source_run)/'run_manifest.json').read_text())
    output = Path(args.out).resolve(); output.mkdir(parents=True, exist_ok=False)
    frames_dir = output/'keyframes'; frames_dir.mkdir()
    start = time.time()
    manifest = dict(schema='aerorecon.run/v1', status='running', pid=os.getpid(), started_at=start,
        input=source['input'], input_sha256=source['input_sha256'], display_name='Full video · fused terrain',
        engine='da3-small', fusion='tsdf-all-frames', units='arbitrary', georeferenced=False,
        configuration=vars(args), warnings=['AI-inferred depth and poses; relative scale, not validated for measurement.',
        f'Top {args.exclude_top:.0%} of each image excluded from geometry.'])
    save(output/'run_manifest.json', manifest)
    try:
        times=[]; paths=[]
        with av.open(source['input']) as container:
            for i, frame in enumerate(container.decode(video=0)):
                rgb=frame.to_ndarray(format='rgb24')
                h,w=rgb.shape[:2]
                rgb=cv2.resize(rgb,(args.resolution,round(h*args.resolution/w)))
                path=frames_dir/f'frame_{i:06d}.jpg'
                cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                paths.append(str(path)); times.append(float(frame.time) if frame.time is not None else i/25)
        if len(paths)<2: raise ValueError('Video needs at least two decodable frames')
        print(f'Decoded all {len(paths)} frames, {times[0]:.2f}–{times[-1]:.2f} seconds.',flush=True)
        torch.set_num_threads(4)
        model=DepthAnything3.from_pretrained(str(ROOT/'models/DA3-SMALL')).to('cuda').eval()
        torch.cuda.reset_peak_memory_stats()
        anchors=np.unique(np.linspace(0,len(paths)-1,min(8,len(paths))).astype(int)).tolist()
        # Reuse a complete feature-based trajectory when available in the ancestry.
        pose_source=Path(args.source_run).resolve(); visited=set(); conditioning={}
        while pose_source not in visited:
            visited.add(pose_source)
            old=json.loads((pose_source/'run_manifest.json').read_text())
            old_metrics=json.loads((pose_source/'metrics.json').read_text()) if (pose_source/'metrics.json').exists() else {}
            if old_metrics.get('registered_ratio',0)>.95 and not old_metrics.get('calibration_suspect',True):
                import pycolmap
                from scipy.spatial.transform import Rotation, Slerp
                rec=pycolmap.Reconstruction(pose_source/'sparse'/str(old_metrics['best_model_id']))
                with (pose_source/'frames.csv').open() as f: timestamps={row['image']:float(row['time_s']) for row in csv.DictReader(f)}
                registered=sorted(rec.images.values(),key=lambda im:timestamps[im.name])
                knot=np.array([timestamps[im.name] for im in registered])
                matrices=np.array([im.cam_from_world().matrix() for im in registered])
                centres=np.array([im.projection_center() for im in registered])
                query=np.clip(times,knot[0],knot[-1])
                rotations=Slerp(knot,Rotation.from_matrix(matrices[:,:3,:3]))(query).as_matrix()
                locations=np.stack([np.interp(query,knot,centres[:,axis]) for axis in range(3)],1)
                all_e=np.tile(np.eye(4),(len(paths),1,1));all_e[:,:3,:3]=rotations
                all_e[:,:3,3]=-np.einsum('nij,nj->ni',rotations,locations)
                cam=rec.cameras[registered[0].camera_id];k=cam.calibration_matrix()
                k[0]*=args.resolution/cam.width;k[1]*=round(cam.height*args.resolution/cam.width)/cam.height
                all_k=np.tile(k,(len(paths),1,1))
                conditioning=dict(extrinsics=all_e,intrinsics=all_k)
                manifest['pose_method']='Feature-based keyframe poses; intermediate poses interpolated with rotation SLERP; endpoint poses held outside registered interval'
                manifest['warnings'].append('Camera intrinsics are assumed. Intermediate frame poses are interpolated from feature-based keyframes.')
                print(f'Using {len(registered)} registered camera poses to constrain every batch.',flush=True)
                break
            parent=old.get('configuration',{}).get('source_run')
            if not parent: break
            pose_source=Path(parent).resolve()
        def predict(indices):
            kwargs={key:value[indices].copy() for key,value in conditioning.items()}
            with torch.inference_mode():
                return model.inference([paths[i] for i in indices],process_res=args.resolution,use_ray_pose=True,ref_view_strategy='middle',**kwargs)
        with torch.inference_mode():
            reference=predict(anchors)
        h,w=reference.depth.shape[1:]
        mask=np.zeros((h,w),bool); mask[int(h*args.exclude_top):]=True
        samples=[]; targets=[]
        for j in range(len(anchors)):
            valid=mask & np.isfinite(reference.depth[j]) & (reference.depth[j]>0)
            valid &= reference.conf[j]>=np.percentile(reference.conf[j][valid],50)
            indices=np.flatnonzero(valid)[::20];samples.append(indices)
            targets.append(project_depth(reference.depth[j],reference.intrinsics[j],reference.extrinsics[j]).reshape(-1,3)[indices])
        target=np.concatenate(targets)
        typical=float(np.median(reference.depth[:,mask]))
        voxel=typical/220
        volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel,sdf_trunc=voxel*4,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        remaining=[i for i in range(len(paths)) if i not in anchors]
        rows=[]; cameras=[]; errors=[]; total_pixels=0
        for batch_index, offset in enumerate(range(0,max(1,len(remaining)),16)):
            batch=anchors+remaining[offset:offset+16]
            with torch.inference_mode():
                pred=predict(batch)
            points=np.concatenate([project_depth(pred.depth[j],pred.intrinsics[j],pred.extrinsics[j]).reshape(-1,3)[samples[j]] for j in range(len(anchors))])
            scale,rotation,translation,error=similarity(points,target)
            if conditioning:
                # Poses already share a fixed world; correct only batch depth scale
                # against the identical anchor pixels, preserving the camera baseline.
                ratios=np.concatenate([reference.depth[j].ravel()[samples[j]]/np.maximum(pred.depth[j].ravel()[samples[j]],1e-6) for j in range(len(anchors))])
                depth_scale=float(np.median(ratios))
                pred.depth*=depth_scale
                error=float(np.median(np.concatenate([np.abs(pred.depth[j].ravel()[samples[j]]-reference.depth[j].ravel()[samples[j]]) for j in range(len(anchors))])))
                scale=1.;rotation=np.eye(3);translation=np.zeros(3)
            if not .01<scale<100 or not np.isfinite(error): raise ValueError('Unstable batch alignment')
            errors.append(error/typical)
            for j,i in enumerate(batch):
                if batch_index and i in anchors: continue
                d=pred.depth[j].astype(np.float32)*scale
                confidence=pred.conf[j]
                valid=mask & np.isfinite(d) & (d>0) & np.isfinite(confidence)
                cutoff=np.percentile(confidence[valid],35)
                valid &= (confidence>=cutoff) & (d<=np.percentile(d[valid],97))
                d[~valid]=0
                e=np.eye(4);e[:3,:3]=pred.extrinsics[j,:3,:3]@rotation.T
                e[:3,3]=scale*pred.extrinsics[j,:3,3]-e[:3,:3]@translation
                k=pred.intrinsics[j]
                rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(pred.processed_images[j])),o3d.geometry.Image(np.ascontiguousarray(d)),
                    depth_scale=1,depth_trunc=float(max(d.max()+1,1)),convert_rgb_to_intensity=False)
                volume.integrate(rgbd,o3d.camera.PinholeCameraIntrinsic(w,h,float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2])),e)
                pixels=int(valid.sum());total_pixels+=pixels
                rows.append(dict(frame=i,image=Path(paths[i]).name,timestamp_s=times[i],integrated_pixels=pixels,batch=batch_index,alignment_error_relative=error/typical))
                cameras.append(dict(name=Path(paths[i]).name,centre=(-e[:3,:3].T@e[:3,3]).tolist()))
            save(output/'progress.json',dict(frames_decoded=len(paths),frames_integrated=len(rows),stage='Fusing video frames'))
            print(f'Fused {len(rows)}/{len(paths)} frames; anchor residual {error/typical:.2%} of median depth.',flush=True)
            del pred
        print('Extracting the shared multi-view surface...',flush=True)
        mesh=volume.extract_triangle_mesh()
        mesh.remove_duplicated_triangles();mesh.remove_degenerate_triangles();mesh.remove_unreferenced_vertices()
        if len(mesh.triangles)==0: raise ValueError('Fusion produced no surface')
        # Preserve full mesh on disk; use a bounded mesh for browser interaction.
        o3d.io.write_triangle_mesh(str(output/'surface_full.ply'),mesh)
        if len(mesh.triangles)>250000: mesh=mesh.simplify_quadric_decimation(250000)
        vertices=np.asarray(mesh.vertices); colors=np.clip(np.asarray(mesh.vertex_colors)*255,0,255).astype(np.uint8)
        faces=np.asarray(mesh.triangles)
        trimesh.Trimesh(vertices=vertices,faces=faces,vertex_colors=colors,process=False).export(output/'surface.glb')
        trimesh.points.PointCloud(vertices,colors=colors).export(output/'dense.ply')
        save(output/'surface_viewer.json',dict(positions=vertices.ravel().tolist(),colors=colors.ravel().tolist(),indices=faces.ravel().tolist(),geometry_provenance='inferred',fusion='tsdf-all-frames'))
        cameras.sort(key=lambda c:c['name'])
        take=np.linspace(0,len(vertices)-1,min(150000,len(vertices))).astype(int)
        save(output/'viewer.json',dict(positions=vertices[take].ravel().tolist(),colors=colors[take].ravel().tolist(),cameras=cameras,
            total_points=len(vertices),displayed_points=len(take),units='arbitrary',geometry_provenance='inferred',surface_available=True))
        rows.sort(key=lambda row:row['frame'])
        with (output/'frames.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with (output/'camera_centres.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['image','x','y','z']);writer.writerows([c['name'],*c['centre']] for c in cameras)
        metrics=dict(engine='da3-small',fusion='tsdf-all-frames',predicted_views=len(rows),selected_images=len(paths),frames_decoded=len(paths),
            frames_integrated=len(rows),coverage_percent=100*len(rows)/len(paths),first_timestamp_s=times[0],last_timestamp_s=times[-1],
            points=len(vertices),surface_triangles=len(faces),integrated_depth_samples=total_pixels,anchor_views=len(anchors),
            alignment_median_relative=float(np.median(errors)),alignment_max_relative=float(max(errors)),voxel_length=voxel,
            geometry_validated=False,geometry_provenance='inferred',units='arbitrary',georeferenced=False,
            model='depth-anything/DA3-SMALL',license='Apache-2.0',gpu_peak_memory_mb=torch.cuda.max_memory_allocated()/1024**2)
        save(output/'metrics.json',metrics)
        (output/'REPORT.md').write_text(f'# Full-video fused reconstruction\n\nAll {len(rows)} / {len(paths)} decoded frames ({times[0]:.2f}–{times[-1]:.2f} seconds) contributed depth to one TSDF volume. Surface triangles are extracted from this combined volume, not a reference image.\n\nEight shared anchor images connect every inference batch using robust 3D similarity alignment. Camera poses for all frames are predicted, not interpolated. Frames.csv records each frame and its contribution. Median anchor alignment residual: {np.median(errors):.2%} of median depth. This is internal consistency, not ground-truth accuracy.\n\nAI-inferred, relative-scale geometry. No GPS or metric calibration. Unseen surfaces cannot be recovered from this video. Bottom 35% confidence and farthest 3% depth excluded per frame; top-image exclusion {args.exclude_top:.0%}. TSDF voxel size is median anchor depth / 220. Full resolution mesh: surface_full.ply; browser mesh capped at 250,000 triangles.\n',encoding='utf-8')
        manifest['status']='complete';manifest['metrics']=metrics
        if conditioning:
            report=(output/'REPORT.md').read_text(encoding='utf-8').replace('Camera poses for all frames are predicted, not interpolated.', manifest['pose_method']+'.').replace('using robust 3D similarity alignment','using a fixed feature-based trajectory and median anchor depth scale correction')
            (output/'REPORT.md').write_text(report,encoding='utf-8')
        print(f'Complete: {len(rows)} frames fused, {len(faces):,} triangles.',flush=True)
    except BaseException as error:
        manifest['status']='failed';manifest['error']=str(error)
        (output/'error.log').write_text(traceback.format_exc());raise
    finally:
        manifest['elapsed_s']=time.time()-start;save(output/'run_manifest.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source_run');parser.add_argument('--out',required=True)
    parser.add_argument('--resolution',type=int,default=336)
    parser.add_argument('--exclude-top',type=float,default=0)
    run(parser.parse_args())
