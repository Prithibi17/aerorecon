"""Per-frame camera pose tracking anchored to a feature-based 3D reconstruction."""
from pathlib import Path
import re

import cv2
import numpy as np


def frame_number(name):
    match=re.fullmatch(r'frame_(\d+)\.[A-Za-z]+',Path(name).name)
    if not match: raise ValueError(f'Unsupported keyframe name: {name}')
    return int(match.group(1))


def interpolate_poses(frame_count, anchors):
    """Linear camera-centre interpolation with rotation SLERP fallback poses."""
    from scipy.spatial.transform import Rotation,Slerp
    indices=np.array([a['frame'] for a in anchors])
    matrices=np.array([a['extrinsic'] for a in anchors])
    query=np.clip(np.arange(frame_count),indices[0],indices[-1])
    rotations=Slerp(indices,Rotation.from_matrix(matrices[:,:3,:3]))(query).as_matrix()
    centres=np.array([-e[:3,:3].T@e[:3,3] for e in matrices])
    locations=np.stack([np.interp(query,indices,centres[:,axis]) for axis in range(3)],1)
    poses=np.tile(np.eye(4),(frame_count,1,1));poses[:,:3,:3]=rotations
    poses[:,:3,3]=-np.einsum('nij,nj->ni',rotations,locations)
    return poses


def solve_pose(object_points,image_points,intrinsic,initial=None):
    """Robustly solve a world-to-camera pose and return its inlier indices."""
    if len(object_points)<30:return None,None
    kwargs={}
    if initial is not None:
        rvec,_=cv2.Rodrigues(initial[:3,:3]);tvec=initial[:3,3,None].copy()
        kwargs=dict(rvec=rvec,tvec=tvec,useExtrinsicGuess=True)
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(
        object_points.astype(np.float64),image_points.astype(np.float64),intrinsic.astype(np.float64),None,
        iterationsCount=150,reprojectionError=4.0,confidence=.999,flags=cv2.SOLVEPNP_ITERATIVE,**kwargs)
    if not ok or inliers is None or len(inliers)<24:return None,None
    selected=inliers.ravel()
    rvec,tvec=cv2.solvePnPRefineLM(object_points[selected].astype(np.float64),image_points[selected].astype(np.float64),
        intrinsic.astype(np.float64),None,rvec,tvec)
    pose=np.eye(4);pose[:3,:3]=cv2.Rodrigues(rvec)[0];pose[:3,3]=tvec.ravel()
    return pose,selected


def track_frame_poses(video_path,reconstruction,registered,intrinsic,width,height):
    """Track reconstructed 3D features through every decoded video frame."""
    frames=[]
    import av
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            image=cv2.resize(frame.to_ndarray(format='gray'),(width,height),interpolation=cv2.INTER_AREA)
            frames.append(image)
    if not frames:raise ValueError('Video has no decodable frames')
    sx=width/reconstruction.cameras[registered[0].camera_id].width
    sy=height/reconstruction.cameras[registered[0].camera_id].height
    anchors=[]
    for image in registered:
        objects=[];pixels=[]
        for point in image.points2D:
            if point.has_point3D():
                objects.append(reconstruction.points3D[point.point3D_id].xyz)
                pixels.append(point.xy)
        pixels=(np.asarray(pixels,dtype=np.float32)*np.asarray([sx,sy],dtype=np.float32)).astype(np.float32)
        objects=np.asarray(objects,dtype=np.float32)
        inside=(pixels[:,0]>=0)&(pixels[:,0]<width)&(pixels[:,1]>=0)&(pixels[:,1]<height)
        extrinsic=np.eye(4);extrinsic[:3]=image.cam_from_world().matrix()
        anchors.append(dict(frame=frame_number(image.name),extrinsic=extrinsic,
            objects=objects[inside],pixels=pixels[inside]))
    anchors=sorted((a for a in anchors if a['frame']<len(frames)),key=lambda a:a['frame'])
    if len(anchors)<2:raise ValueError('At least two registered anchor frames are required')
    fallback=interpolate_poses(len(frames),anchors);poses=fallback.copy()
    methods=np.full(len(frames),'interpolated',dtype=object);inlier_counts=np.zeros(len(frames),dtype=int)
    for anchor_index,anchor in enumerate(anchors):
        start=anchor['frame'];stop=anchors[anchor_index+1]['frame'] if anchor_index+1<len(anchors) else len(frames)
        poses[start]=anchor['extrinsic'];methods[start]='registered';inlier_counts[start]=len(anchor['objects'])
        objects=anchor['objects'].copy();pixels=anchor['pixels'].copy();previous=frames[start]
        for index in range(start+1,stop):
            following,status,error=cv2.calcOpticalFlowPyrLK(previous,frames[index],pixels.reshape(-1,1,2),None,
                winSize=(31,31),maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
            following=following.reshape(-1,2);status=status.ravel().astype(bool);error=error.ravel()
            good=status&np.isfinite(following).all(1)&(error<30)
            good&=(following[:,0]>=0)&(following[:,0]<width)&(following[:,1]>=0)&(following[:,1]<height)
            objects=objects[good];pixels=following[good].astype(np.float32)
            pose,inliers=solve_pose(objects,pixels,intrinsic,fallback[index])
            if pose is not None:
                poses[index]=pose;methods[index]='tracked';inlier_counts[index]=len(inliers)
                objects=objects[inliers];pixels=pixels[inliers]
            previous=frames[index]
            if len(objects)<30:break
    return poses,dict(
        decoded_frames=len(frames),registered_frames=int(np.sum(methods=='registered')),
        tracked_frames=int(np.sum(methods=='tracked')),fallback_frames=int(np.sum(methods=='interpolated')),
        median_tracking_inliers=float(np.median(inlier_counts[methods=='tracked'])) if np.any(methods=='tracked') else 0,
        methods=methods.tolist(),inliers=inlier_counts.tolist())
