"""Bake aligned video observations into a top-down photographic texture."""
from pathlib import Path

import cv2
import numpy as np


def bake_orthophoto(points, colors, vertices, output, resolution=1024, faces=None, vertex_colors=None, semantic_ids=None):
    """Write a dense X/Z orthophoto and return UVs for the supplied mesh."""
    from scipy.ndimage import distance_transform_edt

    points=np.asarray(points,float);colors=np.asarray(colors,np.uint8);vertices=np.asarray(vertices,float)
    low=np.percentile(vertices[:,[0,2]],.5,axis=0)
    high=np.percentile(vertices[:,[0,2]],99.5,axis=0)
    span=np.maximum(high-low,1e-6)
    pixel=np.rint((points[:,[0,2]]-low)/span*(resolution-1)).astype(int)
    inside=np.all((pixel>=0)&(pixel<resolution),axis=1)
    pixel=pixel[inside];heights=points[inside,1];sample_colors=colors[inside]
    flat=pixel[:,1]*resolution+pixel[:,0]
    order=np.lexsort((heights,flat))
    ordered=flat[order]
    # Keep the highest observation where several surfaces project to one cell.
    last=np.r_[ordered[1:]!=ordered[:-1],True]
    chosen=order[last];pixel=pixel[chosen];sample_colors=sample_colors[chosen]
    image=np.zeros((resolution,resolution,3),np.uint8)
    known=np.zeros((resolution,resolution),bool)
    image[pixel[:,1],pixel[:,0]]=sample_colors;known[pixel[:,1],pixel[:,0]]=True
    if not known.any():raise ValueError('No aligned video samples overlap the final surface')
    nearest=distance_transform_edt(~known,return_distances=False,return_indices=True)
    image=image[nearest[0],nearest[1]]
    # Small blur removes single-pixel seams while preserving field boundaries.
    image=cv2.GaussianBlur(image,(3,3),0)
    uv=(vertices[:,[0,2]]-low)/span
    uv[:,1]=1-uv[:,1]
    # Paint accepted object footprints from their video-coloured solid meshes.
    # This prevents oblique background/building rays from streaking the field.
    if faces is not None and vertex_colors is not None and semantic_ids is not None:
        faces=np.asarray(faces,int);vertex_colors=np.asarray(vertex_colors,np.uint8);semantic_ids=np.asarray(semantic_ids)
        object_faces=faces[np.all(np.isin(semantic_ids[faces],[2,3]),axis=1)]
        image_uv=np.column_stack([uv[:,0]*(resolution-1),(1-uv[:,1])*(resolution-1)])
        for face in object_faces:
            polygon=np.rint(image_uv[face]).astype(np.int32)
            color=tuple(int(x) for x in np.mean(vertex_colors[face],axis=0))
            cv2.fillConvexPoly(image,polygon,color,lineType=cv2.LINE_AA)
    cv2.imwrite(str(Path(output)),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    return np.clip(uv,0,1).astype(np.float32),dict(resolution=resolution,bounds_xz=[low.tolist(),high.tolist()],samples=int(len(points)))
