"""MoGe-2 geometry, semantic sky rejection, sparse scale alignment and ground leveling."""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from pipeline.ai_reconstruct import ROOT, save, project_depth


def ground_rotation(normal):
    """Rotate an estimated ground normal to +Y without changing dimensions."""
    normal=np.asarray(normal,dtype=float);normal/=np.linalg.norm(normal)
    target=np.array([0.,1.,0.]);v=np.cross(normal,target);c=float(normal@target)
    if c < -.999999: return np.diag([1.,-1.,-1.])
    skew=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+skew+skew@skew/(1+c)


def vector_rotation(source,target):
    """Return a proper rotation that maps one unit direction to another."""
    source=np.asarray(source,dtype=float);source/=np.linalg.norm(source)
    target=np.asarray(target,dtype=float);target/=np.linalg.norm(target)
    v=np.cross(source,target);c=float(source@target)
    if c<-.999999:
        axis=np.array([1.,0.,0.]) if abs(source[0])<.9 else np.array([0.,1.,0.])
        axis-=source*(axis@source);axis/=np.linalg.norm(axis)
        return 2*np.outer(axis,axis)-np.eye(3)
    skew=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+skew+skew@skew/(1+c)


def fit_field_plane(points,seed=17):
    """Fit a dominant local ground plane with deterministic vectorized RANSAC."""
    points=np.asarray(points,dtype=np.float64)
    points=points[np.isfinite(points).all(1)]
    if len(points)<120:return None
    points=points[::max(1,len(points)//1200)]
    rng=np.random.default_rng(seed)
    triples=rng.integers(0,len(points),size=(96,3))
    a,b,c=points[triples[:,0]],points[triples[:,1]],points[triples[:,2]]
    normals=np.cross(b-a,c-a);length=np.linalg.norm(normals,axis=1)
    usable=length>1e-6
    normals=normals[usable]/length[usable,None];offsets=-np.sum(normals*a[usable],axis=1)
    if not len(normals):return None
    threshold=max(.025,float(np.median(np.linalg.norm(points,axis=1)))*.012)
    distances=np.abs(points@normals.T+offsets)
    winner=int(np.argmax(np.sum(distances<threshold,axis=0)))
    inliers=distances[:,winner]<threshold
    if inliers.sum()<80 or inliers.mean()<.3:return None
    selected=points[inliers];centre=selected.mean(0)
    _,_,vh=np.linalg.svd(selected-centre,full_matrices=False);normal=vh[-1]
    offset=-float(normal@centre)
    if offset<0:normal=-normal;offset=-offset
    if not .1<offset<100:return None
    return normal,offset,float(inliers.mean())


def level_field_pose(depth,valid,ground_mask,intrinsic,extrinsic,state,frame_index):
    """Remove path bending by mapping every observed field plane to one plane."""
    h,w=depth.shape;y,x=np.indices((h,w));mask=valid&ground_mask
    centre=-extrinsic[:3,:3].T@extrinsic[:3,3]
    fitted=None
    if mask.sum()>=120:
        rays=np.stack([x[mask],y[mask],np.ones(mask.sum())],1)@np.linalg.inv(intrinsic).T
        fitted=fit_field_plane(rays*depth[mask,None],17+frame_index)
    measured=fitted is not None
    if fitted is None:
        if state.get('normal') is None:return extrinsic,centre,False
        normal_camera=extrinsic[:3,:3]@state['normal']
        height=float(np.median(state['heights'][-30:]));support=0.
    else:
        normal_camera,height,support=fitted
    camera_world=extrinsic[:3,:3].T;observed_normal=camera_world@normal_camera
    if state.get('normal') is None:
        state['normal']=observed_normal/np.linalg.norm(observed_normal)
        state['base_centre']=centre.copy();state['ground_origin']=centre-state['normal']*height
    normal=state['normal'];camera_world=vector_rotation(observed_normal,normal)@camera_world
    delta=centre-state['base_centre'];horizontal=delta-normal*(delta@normal)
    centre=state['ground_origin']+horizontal+normal*height
    result=np.eye(4);result[:3,:3]=camera_world.T;result[:3,3]=-result[:3,:3]@centre
    state.setdefault('heights',[]).append(height)
    if measured:state.setdefault('supports',[]).append(support)
    state['pose_frames']=state.get('pose_frames',0)+1
    return result,centre,measured


def refine_depth(depth, pixels, target):
    """Fit a smooth local depth correction; retain it only if held-out tracks improve."""
    import cv2
    from scipy.spatial import cKDTree
    pixels=np.asarray(pixels);target=np.asarray(target)
    predicted=depth[pixels[:,1],pixels[:,0]]
    valid=np.isfinite(predicted)&(predicted>0)&np.isfinite(target)&(target>0)
    pixels=pixels[valid];target=target[valid];predicted=predicted[valid]
    if len(target)<50:return depth,False,None,None
    held=np.arange(len(target))%5==0
    train=~held
    ratios=np.clip(np.log(target[train]/predicted[train]),-np.log(2),np.log(2))
    tree=cKDTree(pixels[train])
    def field(query):
        distance,index=tree.query(query,k=min(12,train.sum()))
        weights=1/(distance+4)**2
        gain=(weights*ratios[index]).sum(1)/weights.sum(1)
        # Fade toward the global scale outside observed image regions.
        return np.exp(gain*np.exp(-(distance[:,0]/50)**2))
    before=float(np.median(np.abs(predicted[held]-target[held])/target[held]))
    h,w=depth.shape;y,x=np.mgrid[0:h:16,0:w:16]
    coarse=field(np.stack([x.ravel(),y.ravel()],1)).reshape(x.shape).astype(np.float32)
    gain=cv2.resize(coarse,(w,h),interpolation=cv2.INTER_LINEAR)
    corrected=depth*gain
    after=float(np.median(np.abs(corrected[pixels[held,1],pixels[held,0]]-target[held])/target[held]))
    if after<before*.85:return corrected,True,before,after
    return depth,False,before,before


def apply_corrector(model,rgb,depth,valid,device):
    """Apply the trained scale-invariant residual while preserving global scale."""
    import cv2,torch
    log=np.log(np.maximum(depth,1e-4));median=float(np.median(log[valid]))
    normalized=np.clip(log-median,-4,4)
    small_rgb=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA)
    small_depth=cv2.resize(normalized,(256,256),interpolation=cv2.INTER_LINEAR)
    features=np.concatenate([(small_rgb.transpose(2,0,1)/255).astype(np.float32),small_depth[None].astype(np.float32)])
    with torch.inference_mode(): residual=model(torch.from_numpy(features[None]).to(device))[0,0].cpu().numpy()
    residual=cv2.resize(residual,(depth.shape[1],depth.shape[0]),interpolation=cv2.INTER_LINEAR)
    return depth*np.exp(residual)


def choose_corrected_depth(base,candidate,pixels,target):
    """Gate domain transfer using sparse tracks unseen by the scale fit."""
    pixels=np.asarray(pixels);target=np.asarray(target)
    held=np.arange(len(target))%5==0;train=~held
    def score(depth):
        values=depth[pixels[:,1],pixels[:,0]]
        scale=float(np.median(target[train]/np.maximum(values[train],1e-6)))
        error=float(np.median(np.abs(values[held]*scale-target[held])/target[held]))
        return depth*scale,error
    baseline,before=score(base);corrected,after=score(candidate)
    return (corrected,True,before,after) if after<before*.97 else (baseline,False,before,before)


def run(args):
    import cv2,torch,pycolmap,trimesh,open3d as o3d
    from scipy.spatial.transform import Rotation,Slerp
    from scipy.spatial import cKDTree
    from moge.model.v2 import MoGeModel
    from transformers import SegformerImageProcessor,SegformerForSemanticSegmentation
    source_dir=Path(args.source_run).resolve()
    source=json.loads((source_dir/'run_manifest.json').read_text())
    output=Path(args.out).resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'keyframes').mkdir()
    start=time.time()
    manifest=dict(schema='aerorecon.run/v1',status='running',pid=os.getpid(),started_at=start,
        input=source['input'],input_sha256=source['input_sha256'],engine='moge-2',fusion='semantic-tsdf',
        display_name=f"{source.get('display_name', 'Reconstruction')} · full-video map",configuration=vars(args),units='arbitrary',georeferenced=False,
        warnings=['Experimental reconstruction: camera calibration is assumed and metric dimensions are unvalidated.',
        'Semantic classes and ground orientation are AI estimates. A flat-field prior is applied to semantic ground only.'])
    save(output/'run_manifest.json',manifest)
    try:
        pose_dir=source_dir;visited=set()
        while pose_dir not in visited:
            visited.add(pose_dir)
            metrics=json.loads((pose_dir/'metrics.json').read_text())
            if metrics.get('registered_ratio',0)>.95 and not metrics.get('calibration_suspect',True):break
            parent=json.loads((pose_dir/'run_manifest.json').read_text()).get('configuration',{}).get('source_run')
            if not parent: raise ValueError('This method needs a complete feature-based camera reconstruction first')
            pose_dir=Path(parent).resolve()
        rec=pycolmap.Reconstruction(pose_dir/'sparse'/str(metrics['best_model_id']))
        with (pose_dir/'frames.csv').open() as f: timestamps={r['image']:float(r['time_s']) for r in csv.DictReader(f)}
        registered=sorted(rec.images.values(),key=lambda im:timestamps[im.name])
        knot=np.array([timestamps[im.name] for im in registered])
        centres=np.array([im.projection_center() for im in registered])
        matrices=np.array([im.cam_from_world().matrix() for im in registered])
        slerp=Slerp(knot,Rotation.from_matrix(matrices[:,:3,:3]))
        cam=rec.cameras[registered[0].camera_id]
        width=args.resolution;height=round(cam.height*width/cam.width)
        k=cam.calibration_matrix();k[0]*=width/cam.width;k[1]*=height/cam.height
        fov=float(np.degrees(2*np.arctan(width/(2*k[0,0]))))
        from pipeline.pose_tracking import track_frame_poses
        frame_poses,tracking=track_frame_poses(source['input'],rec,registered,k,width,height)
        manifest['pose_method']='Per-frame optical feature tracking with PnP, anchored to registered keyframes'
        manifest['warnings'].append(f'{tracking["tracked_frames"]} frames use visual pose tracking; {tracking["fallback_frames"]} frames use interpolation because tracking evidence was insufficient.')
        save(output/'run_manifest.json',manifest)
        print(f'Per-frame poses: {tracking["registered_frames"]} registered, {tracking["tracked_frames"]} tracked, {tracking["fallback_frames"]} fallback.',flush=True)
        sparse_by_view=[]
        for im in registered:
            sparse_by_view.append(np.array([rec.points3D[p.point3D_id].xyz for p in im.points2D if p.has_point3D()]))
        torch.set_num_threads(4)
        model=MoGeModel.from_pretrained(str(ROOT/'models/MoGe-2-Base/model.pt')).cuda().eval()
        corrector=None;corrector_path=ROOT/'models/Aerial-Depth-Corrector'
        if (corrector_path/'model.pt').exists() and not (corrector_path/'DO_NOT_USE').exists():
            from pipeline.train_aerial_corrector import Corrector
            corrector=Corrector().cuda().eval();corrector.load_state_dict(torch.load(corrector_path/'model.pt',map_location='cpu',weights_only=True)['state_dict'])
            manifest['display_name']=f"{source.get('display_name', 'Reconstruction')} · trained full-video map"
            manifest['warnings'].append('A synthetic-aerial residual model is enabled only on frames where held-out sparse tracks improve.')
            save(output/'run_manifest.json',manifest)
        processor=SegformerImageProcessor.from_pretrained(str(ROOT/'models/SegFormer-B0'))
        seg=SegformerForSemanticSegmentation.from_pretrained(str(ROOT/'models/SegFormer-B0')).cuda().eval()
        names=seg.config.id2label
        ground_ids=[i for i,name in names.items() if name in {'road','earth','grass','field','path','land','sand'}]
        sky_id=next(i for i,name in names.items() if name=='sky')
        building_ids=[i for i,name in names.items() if name in {'building','house','wall','skyscraper'}]
        tree_ids=[i for i,name in names.items() if name in {'tree','plant','palm'}]
        palette=np.array([[155,155,155],[115,195,105],[244,165,70],[40,125,80],[100,185,245]],np.uint8)
        volume=None;ground=[];semantic_points=[];semantic_groups=[];texture_points=[];texture_colors=[];texture_groups=[];texture_centres=[];rows=[];cameras=[];scale_history=[]
        field_state={};field_levelled_frames=0
        import av
        with av.open(source['input']) as container:
            total=container.streams.video[0].frames
            for i,frame in enumerate(container.decode(video=0)):
                timestamp=float(frame.time);query=float(np.clip(timestamp,knot[0],knot[-1]))
                e=frame_poses[i]
                centre=-e[:3,:3].T@e[:3,3]
                rgb=cv2.resize(frame.to_ndarray(format='rgb24'),(width,height))
                with torch.inference_mode():
                    pred=model.infer(torch.tensor(rgb/255,dtype=torch.float32,device='cuda').permute(2,0,1),resolution_level=2,fov_x=fov)
                    logits=seg(**processor(images=rgb,return_tensors='pt').to('cuda')).logits
                    labels=torch.nn.functional.interpolate(logits,size=(height,width),mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy()
                depth=pred['depth'].cpu().numpy().astype(np.float32)
                valid=pred['mask'].cpu().numpy().astype(bool)&np.isfinite(depth)&(depth>0)&(labels!=sky_id)
                # The laptop GPU is shared by two inference models. Release each
                # frame's large outputs before the next pass to prevent allocator
                # growth during long videos.
                del pred,logits
                if i%10==0: torch.cuda.empty_cache()
                nearest=int(np.argmin(np.abs(knot-query)))
                pc=sparse_by_view[nearest]@e[:3,:3].T+e[:3,3]
                uv=pc@k.T;uv=uv[:,:2]/np.maximum(uv[:,2:],1e-6)
                px=np.rint(uv).astype(int)
                keep=(pc[:,2]>0)&(px[:,0]>=0)&(px[:,0]<width)&(px[:,1]>=0)&(px[:,1]<height)
                px=px[keep];z=pc[keep,2]
                good=valid[px[:,1],px[:,0]]
                correction_used=False;correction_before=None;correction_after=None
                if corrector is not None and good.sum()>=50:
                    candidate=apply_corrector(corrector,rgb,depth,valid,'cuda')
                    depth,correction_used,correction_before,correction_after=choose_corrected_depth(depth,candidate,px[good],z[good])
                ratio=z[good]/depth[px[good,1],px[good,0]]
                if len(ratio)<20: raise ValueError(f'Insufficient sparse depth support at frame {i}')
                scale=float(np.median(ratio));scale_history.append(scale)
                depth*=scale
                depth,refined,held_before,held_after=refine_depth(depth,px[good],z[good])
                residual=float(np.median(np.abs(depth[px[good,1],px[good,0]]-z[good])/z[good]))
                far=float(np.percentile(depth[valid],95));valid &= depth<=far
                # Reject pixels at strong depth edges to avoid stretched vertical curtains.
                gradient=np.maximum(np.abs(cv2.Sobel(depth,cv2.CV_32F,1,0,ksize=3)),np.abs(cv2.Sobel(depth,cv2.CV_32F,0,1,ksize=3)))
                valid &= np.isfinite(gradient)&(gradient<.5*depth)
                depth[~valid]=0
                groups=np.zeros_like(labels,dtype=np.uint8)
                groups[np.isin(labels,ground_ids)]=1;groups[np.isin(labels,building_ids)]=2;groups[np.isin(labels,tree_ids)]=3;groups[labels==sky_id]=4
                if args.flat_field:
                    e,centre,levelled=level_field_pose(depth,valid,groups==1,k,e,field_state,i)
                    field_levelled_frames+=int(levelled)
                if i<3:
                    print(f'Frame {i} depth min/median/max/far: {depth[valid].min():.4f}/{np.median(depth[valid]):.4f}/{depth[valid].max():.4f}/{far:.4f}; aerial correction={correction_used}',flush=True)
                if volume is None:
                    # Bound spatial resolution by the recovered flight baseline.
                    # A bad transferred depth must not allocate a near-zero TSDF grid.
                    flight_extent=float(np.ptp(centres,axis=0).max())
                    voxel=max(float(np.median(depth[valid]))/250,flight_extent/500)
                    volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel,sdf_trunc=voxel*4,color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
                    print(f'TSDF voxel length: {voxel:.6f}',flush=True)
                rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.ascontiguousarray(rgb)),o3d.geometry.Image(depth),depth_scale=1,depth_trunc=far+1,convert_rgb_to_intensity=False)
                volume.integrate(rgbd,o3d.camera.PinholeCameraIntrinsic(width,height,k[0,0],k[1,1],k[0,2],k[1,2]),e)
                xyz=project_depth(depth,k,e)
                ground.append(xyz[valid&(groups==1)][::80])
                semantic_points.append(xyz[valid][::60]);semantic_groups.append(groups[valid][::60])
                # Texture only from bundle-adjusted anchor frames. Tracked poses
                # are adequate for fusion but their small errors smear imagery.
                if tracking['methods'][i]=='registered':
                    texture_points.append(xyz[valid][::4].astype(np.float32));texture_colors.append(rgb[valid][::4])
                    texture_groups.append(groups[valid][::4]);texture_centres.append(centre.copy())
                name=f'frame_{i:06d}.jpg'
                if i%25==0 or i==total-1:
                    overlay=(rgb*.55+palette[groups]*.45).astype(np.uint8)
                    cv2.imwrite(str(output/'keyframes'/name),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
                cameras.append(dict(name=name,centre=centre.tolist()))
                rows.append(dict(frame=i,image=name,timestamp_s=timestamp,integrated_pixels=int(valid.sum()),sky_pixels=int((labels==sky_id).sum()),ground_pixels=int((groups==1).sum()),building_pixels=int((groups==2).sum()),depth_scale=scale,sparse_depth_residual=residual,aerial_corrector_used=correction_used,aerial_before=correction_before,aerial_after=correction_after,locally_refined=refined,heldout_before=held_before,heldout_after=held_after))
                if i%25==0:
                    save(output/'progress.json',dict(frames_decoded=total,frames_integrated=len(rows),stage='MoGe geometry and semantic fusion'))
                    print(f'MoGe + semantics fused {len(rows)}/{total}; sparse depth discrepancy {residual:.1%}',flush=True)
        save(output/'progress.json',dict(frames_decoded=len(rows),frames_integrated=len(rows),stage='Preparing surface for viewing'))
        mesh=volume.extract_triangle_mesh();mesh.remove_duplicated_triangles();mesh.remove_degenerate_triangles();mesh.remove_unreferenced_vertices()
        land=np.concatenate(ground)
        if len(land)<100: raise ValueError('Insufficient semantic ground points to estimate orientation')
        land_cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(land))
        plane,inliers=land_cloud.segment_plane(distance_threshold=voxel*3,ransac_n=3,num_iterations=1500)
        normal=np.array(plane[:3]);origin=np.median(land[inliers],axis=0)
        # Camera must sit above the estimated ground; stored +Y points downward.
        corrected_centres=np.array([camera['centre'] for camera in cameras])
        if np.dot(np.median(corrected_centres,axis=0)-origin,normal)<0:normal=-normal
        rotation=ground_rotation(normal)
        vertices=np.asarray(mesh.vertices).copy()
        mesh.vertices=o3d.utility.Vector3dVector((vertices-origin)@rotation.T)
        o3d.io.write_triangle_mesh(str(output/'surface_full.ply'),mesh)
        if len(mesh.triangles)>1000000:
            mesh=mesh.simplify_vertex_clustering(voxel*2)
        if len(mesh.triangles)>250000:mesh=mesh.simplify_quadric_decimation(250000)
        vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles)
        colors=np.clip(np.asarray(mesh.vertex_colors)*255,0,255).astype(np.uint8)
        samples=(np.concatenate(semantic_points)-origin)@rotation.T
        _,nearest=cKDTree(samples).query(vertices)
        semantic_ids=np.concatenate(semantic_groups)[nearest]
        semantic_colors=palette[semantic_ids]
        object_height_cap=None
        if args.flat_field:
            aligned_cameras=(corrected_centres-origin)@rotation.T
            positive=aligned_cameras[:,1][aligned_cameras[:,1]>0]
            object_height_cap=max(voxel*8,float(np.percentile(positive,90))*.9) if len(positive) else voxel*20
            vertices[semantic_ids==1,1]=0
            objects=np.isin(semantic_ids,[2,3])
            vertices[objects,1]=np.clip(vertices[objects,1],0,object_height_cap)
            vertices[semantic_ids==0,1]=np.clip(vertices[semantic_ids==0,1],-voxel*2,object_height_cap)
            mesh.vertices=o3d.utility.Vector3dVector(vertices)
        solid_houses=solid_trees=0
        if args.solid_objects:
            from pipeline.object_models import solidify
            solid,semantic_ids,solid_houses,solid_trees=solidify(vertices,faces,colors,semantic_ids,object_height_cap or max(voxel*20,.5))
            vertices=np.asarray(solid.vertices);faces=np.asarray(solid.faces);colors=np.asarray(solid.visual.vertex_colors)[:,:3]
            semantic_colors=palette[semantic_ids];mesh=o3d.geometry.TriangleMesh()
            mesh.vertices=o3d.utility.Vector3dVector(vertices);mesh.triangles=o3d.utility.Vector3iVector(faces)
            # Keep every downloadable surface consistent with the recognisable
            # object geometry shown in the viewer.
            mesh.vertex_colors=o3d.utility.Vector3dVector(colors.astype(np.float64)/255)
            o3d.io.write_triangle_mesh(str(output/'surface_full.ply'),mesh)
        from pipeline.texture_atlas import bake_orthophoto
        aligned_batches=[]
        for observed,observed_groups,camera_centre in zip(texture_points,texture_groups,texture_centres):
            aligned=(observed-origin)@rotation.T
            camera_aligned=(camera_centre-origin)@rotation.T
            is_ground=observed_groups==1
            direction=aligned[is_ground]-camera_aligned
            usable=np.abs(direction[:,1])>1e-6
            intersection=aligned[is_ground].copy()
            factor=-camera_aligned[1]/direction[usable,1]
            intersection[usable]=camera_aligned+direction[usable]*factor[:,None]
            intersection[usable,1]=0
            aligned[is_ground]=intersection
            aligned_batches.append(aligned[is_ground])
        aligned_texture_points=np.concatenate(aligned_batches)
        aligned_texture_colors=np.concatenate([color[group==1] for color,group in zip(texture_colors,texture_groups)])
        texture_uv,texture_info=bake_orthophoto(aligned_texture_points,aligned_texture_colors,vertices,output/'texture.png',
            faces=faces,vertex_colors=colors,semantic_ids=semantic_ids)
        from PIL import Image
        textured=trimesh.Trimesh(vertices=vertices,faces=faces,process=False,
            visual=trimesh.visual.texture.TextureVisuals(uv=texture_uv,image=Image.open(output/'texture.png')))
        textured.export(output/'surface.glb')
        trimesh.points.PointCloud(vertices,colors=colors).export(output/'dense.ply')
        save(output/'surface_viewer.json',dict(positions=vertices.ravel().tolist(),colors=colors.ravel().tolist(),semantic_colors=semantic_colors.ravel().tolist(),indices=faces.ravel().tolist(),uv=texture_uv.ravel().tolist(),texture='texture.png',geometry_provenance='inferred'))
        for camera in cameras:camera['centre']=((np.array(camera['centre'])-origin)@rotation.T).tolist()
        take=np.linspace(0,len(vertices)-1,min(150000,len(vertices))).astype(int)
        save(output/'viewer.json',dict(positions=vertices[take].ravel().tolist(),colors=colors[take].ravel().tolist(),cameras=cameras,total_points=len(vertices),displayed_points=len(take),units='arbitrary',geometry_provenance='inferred',surface_available=True,ground_aligned=True))
        with (output/'frames.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with (output/'camera_centres.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['image','x','y','z']);writer.writerows([c['name'],*c['centre']] for c in cameras)
        metrics=dict(engine='moge-2',fusion='semantic-tsdf',frames_decoded=len(rows),frames_integrated=len(rows),predicted_views=len(rows),coverage_percent=100,
            first_timestamp_s=rows[0]['timestamp_s'],last_timestamp_s=rows[-1]['timestamp_s'],points=len(vertices),surface_triangles=len(faces),
            ground_aligned=True,ground_plane_inlier_ratio=len(inliers)/len(land),sparse_depth_residual_median=float(np.median([r['sparse_depth_residual'] for r in rows])),
            geometry_validated=False,units='arbitrary',model='Ruicheng/moge-2-vitb-normal',semantic_model='nvidia/segformer-b0-finetuned-ade-512-512')
        metrics['refined_frames']=sum(r['locally_refined'] for r in rows)
        metrics['aerial_corrector_frames']=sum(r['aerial_corrector_used'] for r in rows)
        metrics['field_levelled_frames']=field_levelled_frames
        metrics['field_pose_frames']=field_state.get('pose_frames',0)
        metrics['field_plane_support_median']=float(np.median(field_state.get('supports',[0])))
        metrics['relative_camera_height_median']=float(np.median(field_state.get('heights',[0])))
        metrics['relative_object_height_cap']=object_height_cap
        metrics['solid_house_models']=solid_houses
        metrics['solid_tree_models']=solid_trees
        metrics['surface_appearance']='Video-projected vertex colours on visible object and terrain surfaces'
        metrics['texture_atlas']=texture_info
        metrics.update({key:value for key,value in tracking.items() if key not in {'methods','inliers'}})
        metrics['heldout_depth_error_before']=float(np.median([r['heldout_before'] for r in rows if r['heldout_before'] is not None]))
        metrics['heldout_depth_error_after']=float(np.median([r['heldout_after'] for r in rows if r['heldout_after'] is not None]))
        if metrics['sparse_depth_residual_median']>.1:
            manifest['warnings'].append('Substantial disagreement between inferred depth and tracked features remains; building dimensions are unreliable.')
        save(output/'metrics.json',metrics)
        save(output/'ground_transform.json',dict(rotation=rotation.tolist(),origin=origin.tolist(),plane=np.asarray(plane).tolist(),convention='Raw +Y down; viewer +Y up, X/Z horizontal; heading arbitrary'))
        (output/'REPORT.md').write_text(f'# MoGe-2 semantic video reconstruction\n\n{len(rows)} frames fused. MoGe-2 Base predicts depth and validity; SegFormer labels sky, ground, buildings and vegetation. Sky is excluded using semantic labels, with no fixed top crop. Each frame depth scale is fitted to visible sparse feature depths.\n\nA flat-field prior maps the local semantic ground plane in each supported frame to one shared plane; building and vegetation relief is retained. Final ground-plane support {len(inliers)/len(land):.1%}. Horizontal heading remains arbitrary.\n\nCamera intrinsics are assumed. {tracking["registered_frames"]} frames use registered keyframe poses, {tracking["tracked_frames"]} use optical-flow/PnP camera poses, and {tracking["fallback_frames"]} use interpolation where tracking evidence was insufficient. Median tracking support was {tracking["median_tracking_inliers"]:.0f} inlier 3D features. Median sparse depth disagreement {metrics["sparse_depth_residual_median"]:.1%}; not a ground-truth accuracy measure. No metric or GPS accuracy is established. Semantic labels may be wrong. MoGe-2: MIT; SegFormer: NVIDIA research/non-commercial license.\n\nKeyframes show semantic overlays: green ground, orange buildings, dark green vegetation, blue sky, gray other.\n',encoding='utf-8')
        with (output/'REPORT.md').open('a',encoding='utf-8') as report:
            report.write(f'\nFlat-field leveling used measured local planes in {field_levelled_frames} of {len(rows)} frames and applied the shared field pose to {metrics["field_pose_frames"]} frames. Median measured plane support was {metrics["field_plane_support_median"]:.1%}. Semantic ground vertices are projected to the field plane; building and vegetation relief is capped below relative height {object_height_cap:.3f} to reject impossible depth sheets above the drone. This scene prior is appropriate for the supplied field footage and should be disabled for mountainous scenes.\n')
            report.write(f'\nStretched semantic object sheets were removed. {solid_houses} compact building clusters were converted to closed wall-and-pitched-roof models, and {solid_trees} compact vegetation clusters were converted to trunk-and-canopy models. These procedural solids show recognizable object classes and approximate placement; they are not measured architectural or botanical geometry.\n')
            report.write(f'\nA {texture_info["resolution"]} × {texture_info["resolution"]} photographic texture atlas was baked from {texture_info["samples"]:,} aligned video samples and embedded in the GLB. It provides real image detail on upward-facing visible surfaces. Vertical and unseen faces remain inferred because the source flight does not observe them completely.\n')
            report.write(f'\nLocal depth refinement was accepted in {metrics["refined_frames"]} frames only when withheld feature depths improved by at least 15%. Median withheld relative depth discrepancy: {metrics["heldout_depth_error_before"]:.1%} before, {metrics["heldout_depth_error_after"]:.1%} after. These features come from the same assumed-calibration reconstruction; this is not independent ground truth.\n')
            report.write(f'\nThe lightweight aerial corrector was trained on TartanAir V2 synthetic RGB/depth pairs. Its best whole-trajectory validation AbsRel improved from 11.67% to 10.20%. On this video it was accepted in {metrics["aerial_corrector_frames"]} of {len(rows)} frames, only when a separate held-out subset of tracked feature depths improved by at least 3%. Synthetic validation and this internal gate do not establish real-world or metric accuracy.\n')
        manifest['status']='complete';manifest['metrics']=metrics
        print(f'Complete: {len(rows)} frames, semantic sky rejection, rigid ground alignment.',flush=True)
    except BaseException as error:
        manifest['status']='failed';manifest['error']=str(error);(output/'error.log').write_text(traceback.format_exc());raise
    finally:
        manifest['elapsed_s']=time.time()-start;save(output/'run_manifest.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source_run');parser.add_argument('--out',required=True);parser.add_argument('--resolution',type=int,default=504)
    parser.add_argument('--flat-field',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--solid-objects',action=argparse.BooleanOptionalAction,default=True)
    run(parser.parse_args())
