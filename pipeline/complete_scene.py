"""Depth-checked semantics, ground alignment and explicitly inferred closed backs."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree
import torch
import trimesh
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from pipeline.cli_support import save_json
from pipeline.open3d_refine import read_colmap_array
from pipeline.semantic_fusion import ground_rotation


def fit_ground(points):
    if len(points) < 100:
        raise ValueError('Insufficient stereo-supported semantic ground')
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d.utility.random.seed(42)
    plane, inliers = cloud.segment_plane(.045, 3, 1500)
    normal = np.asarray(plane[:3]); normal /= np.linalg.norm(normal)
    if normal[1] < 0:
        normal *= -1
    return ground_rotation(normal), np.mean(points[inliers], axis=0), len(inliers)/len(points)


def closed_back(points, colors):
    """Close a cluster down to its ground footprint, without inventing a class template."""
    footprint = points[::max(1, len(points)//400)].copy()
    footprint[:, 1] = 0
    evidence = points.copy(); evidence[:, 1] = np.maximum(evidence[:, 1], 0)
    hull = trimesh.convex.convex_hull(np.concatenate([evidence, footprint]))
    _, nearest = cKDTree(points).query(hull.vertices)
    hull.visual.vertex_colors = colors[nearest]
    return hull


def run(args):
    source, dense, out = Path(args.source), Path(args.dense), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    source_manifest = json.loads((source/'run_manifest.json').read_text())
    dense_manifest=json.loads((dense/'run_manifest.json').read_text())
    camera_manifest=json.loads((Path(args.cameras)/'run_manifest.json').read_text())
    digest=source_manifest.get('input_sha256')
    if not digest or any(m.get('input_sha256')!=digest for m in [dense_manifest,camera_manifest]):
        raise ValueError('Mesh, dense depths, and camera dataset must belong to the same video')
    if Path(camera_manifest.get('configuration',{}).get('mesh_run','')).resolve()!=source.resolve():
        raise ValueError('Camera coordinates must have been exported for this source mesh')
    manifest = {**source_manifest, 'engine':'semantic-completion', 'status':'running',
                'pid':os.getpid(), 'started_at':started,
                'display_name':'Ground-aligned scene · optional completed backs',
                'configuration':{'source':str(source), 'dense':str(dense)},
                'metrics':{}, 'elapsed_s':0,
                'warnings':['Closed backs and ground gap filling are inferred. Relative scale; no GPS or metric accuracy.']}
    save_json(out/'run_manifest.json', manifest)
    try:
        mesh = o3d.io.read_triangle_mesh(str(source/'surface_open3d.ply'))
        vertices = np.asarray(mesh.vertices).copy()
        faces = np.asarray(mesh.triangles).copy()
        colors = np.clip(np.asarray(mesh.vertex_colors)*255,0,255).astype(np.uint8)
        dataset = json.loads((Path(args.cameras)/'training/dataset.json').read_text())
        cameras = dataset['cameras']
        processor = SegformerImageProcessor.from_pretrained('models/SegFormer-B0',local_files_only=True)
        model = SegformerForSemanticSegmentation.from_pretrained('models/SegFormer-B0',local_files_only=True).cuda().eval()
        lookup = np.zeros(150, np.uint8)
        for key, name in model.config.id2label.items():
            lookup[int(key)] = (4 if name=='sky' else 1 if name in {'road','earth','grass','field','path','land','sand'}
                                else 2 if name in {'building','house','wall','skyscraper'} else 3 if name in {'tree','plant','palm'} else 0)
        votes = np.zeros((len(vertices),5), np.uint16)
        sky_votes = np.zeros(len(vertices), np.uint16)
        in_view = np.zeros(len(vertices), np.uint16)
        views=[]
        for index, camera in enumerate(cameras):
            rgb = np.asarray(Image.open(dense/'workspace/images'/camera['name']).convert('RGB'))
            rgb = cv2.resize(rgb,(camera['width'],camera['height']),interpolation=cv2.INTER_AREA)
            with torch.inference_mode():
                inputs=processor(images=Image.fromarray(rgb),return_tensors='pt').to('cuda')
                logits=model(**inputs).logits
                label=torch.nn.functional.interpolate(logits,size=rgb.shape[:2],mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy()
            groups=lookup[label]
            # Horizon silhouettes have unstable stereo depth and create distant walls.
            # Exclude a conservative band around sky, not only labelled sky pixels.
            sky=cv2.dilate((groups==4).astype(np.uint8),np.ones((17,17),np.uint8))>0
            groups[sky]=4
            depth=read_colmap_array(dense/'workspace/stereo/depth_maps'/f"{camera['name']}.geometric.bin")
            depth=cv2.resize(depth,(camera['width'],camera['height']),interpolation=cv2.INTER_NEAREST)
            transform=np.asarray(camera['world_to_camera']); K=np.asarray(camera['K'])
            camera_points=vertices@transform[:3,:3].T+transform[:3,3]
            xy=camera_points@K.T
            pixel=np.rint(np.clip(xy[:,:2]/np.maximum(xy[:,2:],1e-8),-1e6,1e6)).astype(int)
            valid=(camera_points[:,2]>0)&(pixel[:,0]>=0)&(pixel[:,0]<rgb.shape[1])&(pixel[:,1]>=0)&(pixel[:,1]<rgb.shape[0])
            ids=np.flatnonzero(valid); x,y=pixel[ids].T
            in_view[ids]+=1; sky_votes[ids]+=sky[y,x]
            z=depth[y,x]
            supported=(z>0)&(np.abs(z-camera_points[ids,2])<np.maximum(.04,z*.025))
            ids=ids[supported]; x,y=pixel[ids].T
            votes[ids,groups[y,x]]+=1
            views.append((rgb,groups,depth,transform,K))
            if index%10==0: print(f'Semantic/depth visibility {index+1}/{len(cameras)}',flush=True)
        del model; torch.cuda.empty_cache()
        semantic=votes.argmax(1)
        supported=votes[:,:4].sum(1)>=5
        keep=supported&(sky_votes/np.maximum(in_view,1)<.03)&(semantic!=4)
        ground=keep&(semantic==1)
        rotation, origin, support=fit_ground(vertices[ground])
        aligned=(vertices-origin)@rotation.T
        # Reject geometry substantially beneath the measured ground. Small stereo noise
        # is flattened only in the explicitly inferred completion layer.
        keep &= aligned[:,1]>-.09
        observed_faces=faces[np.all(keep[faces],axis=1)]
        triangle_points=aligned[observed_faces]
        normals=np.cross(triangle_points[:,1]-triangle_points[:,0],triangle_points[:,2]-triangle_points[:,0])
        verticality=np.abs(normals[:,1])/np.maximum(np.linalg.norm(normals,axis=1),1e-10)
        ground_face=(semantic[observed_faces]==1).sum(1)>=2
        ground_wall=ground_face&(verticality<.55)
        rejected_ground_walls=int(ground_wall.sum())
        observed_faces=observed_faces[~ground_wall]
        span=np.ptp(aligned[keep][:,[0,2]],axis=0).max()
        triangles=aligned[observed_faces]
        longest=np.max(np.linalg.norm(triangles-np.roll(triangles,1,axis=1),axis=2),axis=1)
        observed_faces=observed_faces[longest<max(span*.025,.15)]
        observed=trimesh.Trimesh(aligned,observed_faces,vertex_colors=colors,process=False)
        observed.remove_unreferenced_vertices()
        if len(observed.faces)>60000:
            reduced=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(observed.vertices),o3d.utility.Vector3iVector(observed.faces))
            reduced.vertex_colors=o3d.utility.Vector3dVector(np.asarray(observed.visual.vertex_colors)[:,:3]/255)
            reduced=reduced.simplify_quadric_decimation(60000)
            observed=trimesh.Trimesh(np.asarray(reduced.vertices),np.asarray(reduced.triangles),vertex_colors=np.asarray(reduced.vertex_colors)*255,process=False)
        additions=[]; counts={}
        span=np.ptp(aligned[keep][:,[0,2]],axis=0).max()
        for group,name in [(2,'building'),(3,'vegetation')]:
            ids=np.flatnonzero(keep&(semantic==group)&(aligned[:,1]>.07))
            cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned[ids]))
            labels=np.asarray(cloud.cluster_dbscan(eps=max(span*.012,.075),min_points=20))
            count=0
            for label in np.unique(labels[labels>=0]):
                cluster=ids[labels==label]
                if len(cluster)<40: continue
                pts=aligned[cluster]
                if np.ptp(pts[:,[0,2]],axis=0).max()>span*(.12 if group==2 else .18): continue
                try: item=closed_back(pts,colors[cluster])
                except Exception: continue
                if item.is_watertight:
                    additions.append(item); count+=1
            counts[name]=count
        # Fill only near observed ground; this avoids an invented giant rectangular field.
        # Preserve the footprint of all supported ground, including ground below
        # the uncertain fitted plane. Its planar replacement is explicitly inferred.
        ground_points=aligned[ground]
        ground_colors=colors[ground]
        cell=max(span/160,.04)
        low=ground_points[:,[0,2]].min(0); high=ground_points[:,[0,2]].max(0)
        gx,gz=np.meshgrid(np.arange(low[0],high[0]+cell,cell),np.arange(low[1],high[1]+cell,cell))
        query=np.column_stack([gx.ravel(),gz.ravel()])
        distance,nearest=cKDTree(ground_points[:,[0,2]]).query(query)
        grid=np.column_stack([query[:,0],np.zeros(len(query)),query[:,1]])
        grid_raw=grid@rotation+origin
        # Appearance on the uncertain plane is inferred from local 3D samples.
        # Camera projection here would stretch oblique pixels across large gaps.
        distances,neighbors=cKDTree(ground_points[:,[0,2]]).query(query,k=min(8,len(ground_points)))
        weights=1/np.maximum(distances,cell*.3)**2
        grid_colors=np.clip((ground_colors[neighbors]*weights[...,None]).sum(1)/weights.sum(1)[:,None],0,255).astype(np.uint8)
        grid_votes=np.zeros(len(grid),int)
        for rgb,groups,depth,T,K in views:
            cp=grid_raw@T[:3,:3].T+T[:3,3]; projected=cp@K.T
            pixel=np.rint(np.clip(projected[:,:2]/np.maximum(projected[:,2:],1e-8),-1e6,1e6)).astype(int)
            valid=(cp[:,2]>0)&(pixel[:,0]>=1)&(pixel[:,0]<rgb.shape[1]-1)&(pixel[:,1]>=1)&(pixel[:,1]<rgb.shape[0]-1)
            ids=np.flatnonzero(valid); x,y=pixel[ids].T
            ids=ids[groups[y,x]==1]
            grid_votes[ids]+=1
        height,width=gx.shape
        a=(np.arange(height-1)[:,None]*width+np.arange(width-1)).ravel()
        triangles=np.concatenate([np.column_stack([a,a+width,a+1]),np.column_stack([a+1,a+width,a+width+1])])
        fill=(grid_votes>=3)|(distance<cell*3)
        triangles=triangles[np.all(fill[triangles],axis=1)]
        ground_mesh=trimesh.Trimesh(grid,triangles,vertex_colors=grid_colors,process=False)
        ground_mesh.remove_unreferenced_vertices(); additions.insert(0,ground_mesh)
        combined=trimesh.util.concatenate([observed,*additions])
        # Bake a per-triangle atlas. Observed triangles use a calibrated, depth-checked
        # source image. Generated triangles use local transferred appearance.
        tile=16; columns=256; total=len(combined.faces)
        atlas=np.zeros((int(np.ceil(total/columns))*tile,columns*tile,3),np.uint8)
        expanded=np.asarray(combined.vertices)[combined.faces].reshape(-1,3)
        vertex_colors=np.asarray(combined.visual.vertex_colors)[:,:3][combined.faces]
        centers=expanded.reshape(-1,3,3).mean(1)
        raw_centers=centers@rotation+origin
        choice=np.full(total,-1,int); best=np.full(total,np.inf)
        for index,(rgb,groups,depth,T,K) in enumerate(views):
            cp=raw_centers@T[:3,:3].T+T[:3,3]; uv=cp@K.T
            px=np.rint(np.clip(uv[:,:2]/np.maximum(uv[:,2:],1e-8),-1e6,1e6)).astype(int)
            valid=(cp[:,2]>0)&(px[:,0]>=2)&(px[:,0]<rgb.shape[1]-2)&(px[:,1]>=2)&(px[:,1]<rgb.shape[0]-2)
            ids=np.flatnonzero(valid); x,y=px[ids].T; z=depth[y,x]
            good=(groups[y,x]!=4)&(z>0)&(np.abs(z-cp[ids,2])<np.maximum(.05,z*.03))&(cp[ids,2]<best[ids])
            ids=ids[good]; best[ids]=cp[ids,2]; choice[ids]=index
        uv_out=np.zeros((total,3,2),np.float32)
        dst=np.array([[2,2],[tile-3,2],[2,tile-3]],np.float32)
        yy,xx=np.mgrid[:tile,:tile]; b=np.clip((xx-2)/(tile-5),0,1); c=np.clip((yy-2)/(tile-5),0,1); a=np.maximum(1-b-c,0); norm=a+b+c
        photo_faces=0
        for i,face in enumerate(expanded.reshape(-1,3,3)):
            patch=(a[...,None]*vertex_colors[i,0]+b[...,None]*vertex_colors[i,1]+c[...,None]*vertex_colors[i,2])/norm[...,None]
            if choice[i]>=0 and i<len(observed.faces):
                rgb,groups,depth,T,K=views[choice[i]]
                raw_face=face@rotation+origin; cp=raw_face@T[:3,:3].T+T[:3,3]; p=cp@K.T; src=(p[:,:2]/p[:,2:]).astype(np.float32)
                sx=np.rint(src[:,0]).astype(int); sy=np.rint(src[:,1]).astype(int)
                if np.all(cp[:,2]>0) and np.all((sx>=0)&(sx<rgb.shape[1])&(sy>=0)&(sy<rgb.shape[0])) and np.all(groups[sy,sx]!=4):
                    affine=cv2.getAffineTransform(src,dst)
                    warped_sky=cv2.warpAffine((groups==4).astype(np.uint8),affine,(tile,tile),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=1)
                    interior=(xx>=2)&(yy>=2)&(xx+yy<=tile-1)
                    if not warped_sky[interior].any():
                        patch=cv2.warpAffine(rgb,affine,(tile,tile),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                        photo_faces+=1
            x=(i%columns)*tile; y=(i//columns)*tile
            atlas[y:y+tile,x:x+tile]=np.clip(patch,0,255).astype(np.uint8)
            uv_out[i,:,0]=(dst[:,0]+x+.5)/atlas.shape[1]
            uv_out[i,:,1]=1-(dst[:,1]+y+.5)/atlas.shape[0]
        image=Image.fromarray(atlas); image.save(out/'texture.png')
        exported=trimesh.Trimesh(expanded,np.arange(len(expanded)).reshape(-1,3),process=False)
        exported.visual=trimesh.visual.texture.TextureVisuals(uv=uv_out.reshape(-1,2),image=image)
        exported.export(out/'surface.glb')
        save_json(out/'surface_viewer.json',{'positions':expanded.ravel().tolist(),'indices':np.arange(len(expanded)).tolist(),
                  'colors':np.full(expanded.shape,255,np.uint8).ravel().tolist(),'uv':uv_out.ravel().tolist(),'texture':'texture.png',
                  'inferred_face_start':len(observed.faces),'geometry_provenance':'Stereo-supported observed mesh plus optional inferred closure'})
        aligned_cameras=[{'name':c['name'],'centre':((np.asarray(c['centre'])-origin)@rotation.T).tolist()} for c in cameras]
        with (out/'camera_centres.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(['image','x','y','z'])
            writer.writerows([c['name'],*c['centre']] for c in aligned_cameras)
        save_json(out/'viewer.json',{'positions':aligned[keep].ravel().tolist(),'colors':colors[keep].ravel().tolist(),
                  'cameras':aligned_cameras,'ground_aligned':True,'surface_available':True,'completion_available':True,
                  'total_points':int(keep.sum()),'displayed_points':int(keep.sum()),'units':'arbitrary'})
        trimesh.points.PointCloud(aligned[keep],colors[keep]).export(out/'dense.ply')
        metrics={'engine':'semantic-completion','points':int(keep.sum()),'registered_images':len(cameras),'selected_images':len(cameras),
                 'registered_ratio':1.,'rejected_vertices':int((~keep).sum()),'ground_plane_support':support,
                 'observed_faces':len(observed.faces),'inferred_faces':total-len(observed.faces),'photo_textured_faces':photo_faces,
                 'closed_building_clusters':counts['building'],'closed_vegetation_clusters':counts['vegetation'],
                 'ground_faces':len(ground_mesh.faces),'ground_grid_camera_supported':int((grid_votes>=3).sum()),
                 'horizon_mask_dilation_pixels':17,'minimum_supporting_views':5,
                 'rejected_steep_ground_faces':rejected_ground_walls,
                 'geometry_validated':False,'units':'arbitrary'}
        save_json(out/'metrics.json',metrics)
        save_json(out/'alignment.json',{'rotation':rotation.tolist(),'origin':origin.tolist(),'ground_support':support})
        if support<.5:
            manifest['warnings'].append(f'Ground fit has {support:.0%} support; plane alignment is uncertain. Completion is illustrative.')
        (out/'REPORT.md').write_text('# Ground and object completion\n\n'+json.dumps(metrics,indent=2)+'\n\nSky is excluded using semantic masks and depth-checked multi-view votes. Ground is refitted from stereo-supported ground points. Closed convex backs and ground patches are inferred. Their colours are interpolated from the same local object; unseen photographic details are not recovered. Toggle inferred completion to inspect only observed mesh. No GPS or metric validation.\n',encoding='utf-8')
        manifest.update(status='complete',metrics=metrics,elapsed_s=time.time()-started)
        save_json(out/'progress.json',{'status':'complete','stage':'Completed mesh ready','stage_index':4,'percent':100,
                  'stages':['Depth-checked semantics','Ground plane fit','Sky / artifact rejection','Inferred closure','Texture atlas']})
        print(json.dumps(metrics,indent=2),flush=True)
    except Exception as error:
        manifest.update(status='failed',error=str(error)); raise
    finally: save_json(out/'run_manifest.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='outputs/drone-open3d-tsdf-v2')
    parser.add_argument('--dense',default='outputs/drone-photogrammetry-dense')
    parser.add_argument('--cameras',default='outputs/drone-gsplat-v2')
    parser.add_argument('--out',required=True)
    run(parser.parse_args())
