"""Convert reliable semantic clusters into simple closed house and tree meshes."""
import math
import cv2
import numpy as np
import trimesh


def components(points,cell=.06,min_area=15,max_area=800):
    """Return compact X/Z clusters while rejecting isolated noise and giant sheets."""
    if len(points)==0:return []
    xz=points[:,[0,2]];origin=xz.min(0)-cell
    # Nearest-cell quantisation avoids splitting a continuous surface when a
    # coordinate such as 0.18 is represented just below its decimal value.
    ij=np.rint((xz-origin)/cell).astype(int);shape=ij.max(0)+2
    mask=np.zeros((shape[1],shape[0]),np.uint8);mask[ij[:,1],ij[:,0]]=1
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    result=[]
    for label in range(1,count):
        area=int(stats[label,cv2.CC_STAT_AREA])
        if min_area<=area<=max_area:
            selected=points[labels[ij[:,1],ij[:,0]]==label]
            if len(selected)>=20:result.append(selected)
    return result


def house(cluster,height_cap):
    rect=cv2.minAreaRect(cluster[:,[0,2]].astype(np.float32));(cx,cz),(width,length),angle=rect
    width=float(np.clip(width+.08,.18,1.15));length=float(np.clip(length+.08,.18,1.4))
    height=float(np.clip(np.percentile(cluster[:,1],85),.18,height_cap*.9));wall=height*.68
    transform=trimesh.transformations.rotation_matrix(math.radians(-angle),[0,1,0])
    transform[:3,3]=[cx,wall/2,cz]
    walls=trimesh.creation.box([width,wall,length],transform=transform)
    local=np.array([[-width/2,wall,-length/2],[width/2,wall,-length/2],[0,height,-length/2],
                    [-width/2,wall,length/2],[width/2,wall,length/2],[0,height,length/2]],float)
    local=trimesh.transform_points(local,transform@trimesh.transformations.translation_matrix([0,-wall/2,0]))
    faces=np.array([[0,1,2],[3,5,4],[0,3,4],[0,4,1],[1,4,5],[1,5,2],[2,5,3],[2,3,0]])
    roof=trimesh.Trimesh(local,faces,process=False)
    result=trimesh.util.concatenate([walls,roof]);result.visual.vertex_colors=np.tile([196,120,65,255],(len(result.vertices),1))
    return result


def tree(cluster,height_cap):
    centre=np.median(cluster[:,[0,2]],axis=0)
    spread=np.percentile(np.linalg.norm(cluster[:,[0,2]]-centre,axis=1),75)
    radius=float(np.clip(spread,.12,.42));height=float(np.clip(np.percentile(cluster[:,1],85),.24,height_cap*.95))
    trunk_height=height*.48
    rotation=trimesh.transformations.rotation_matrix(-np.pi/2,[1,0,0])
    transform=trimesh.transformations.translation_matrix([centre[0],trunk_height/2,centre[1]])@rotation
    trunk=trimesh.creation.cylinder(radius=radius*.18,height=trunk_height,sections=10,transform=transform)
    trunk.visual.vertex_colors=np.tile([105,72,40,255],(len(trunk.vertices),1))
    canopy=trimesh.creation.icosphere(subdivisions=2,radius=1)
    canopy.vertices*=np.array([radius,(height-trunk_height)/2,radius])
    canopy.vertices+=np.array([centre[0],trunk_height+(height-trunk_height)/2,centre[1]])
    canopy.visual.vertex_colors=np.tile([43,120,67,255],(len(canopy.vertices),1))
    result=trimesh.util.concatenate([trunk,canopy])
    result.metadata['trunk_vertices']=len(trunk.vertices)
    return result


def transfer_video_colors(mesh,sample_points,sample_colors,start=0):
    """Project nearby observed video colours onto generated object vertices."""
    from scipy.spatial import cKDTree
    if len(sample_points)==0 or start>=len(mesh.vertices):return mesh
    query=np.asarray(mesh.vertices)[start:][:,[0,2]]
    reference=np.asarray(sample_points)[:,[0,2]]
    k=min(8,len(reference));distance,index=cKDTree(reference).query(query,k=k)
    if k==1:distance=distance[:,None];index=index[:,None]
    weights=1/np.maximum(distance,.02)**2
    painted=(np.asarray(sample_colors)[index]*weights[...,None]).sum(1)/weights.sum(1)[:,None]
    rgba=np.asarray(mesh.visual.vertex_colors).copy()
    rgba[start:,:3]=np.clip(painted,0,255).astype(np.uint8);rgba[start:,3]=255
    mesh.visual.vertex_colors=rgba
    return mesh


def solidify(vertices,faces,colors,semantic_ids,height_cap):
    """Remove object depth sheets and replace compact clusters with closed solids."""
    vertices=np.asarray(vertices);faces=np.asarray(faces);colors=np.asarray(colors)
    keep=~np.any(np.isin(semantic_ids[faces],[2,3]),axis=1)
    kept_faces=faces[keep];used=np.unique(kept_faces)
    remap=np.full(len(vertices),-1,dtype=int);remap[used]=np.arange(len(used))
    base_ids=semantic_ids[used]
    base=trimesh.Trimesh(vertices[used],remap[kept_faces],vertex_colors=colors[used],process=False)
    building_clusters=components(vertices[(semantic_ids==2)&(vertices[:,1]>.08)])
    tree_clusters=components(vertices[(semantic_ids==3)&(vertices[:,1]>.08)])
    building_samples=(semantic_ids==2)&(vertices[:,1]>.08)
    tree_samples=(semantic_ids==3)&(vertices[:,1]>.08)
    houses=[transfer_video_colors(house(cluster,height_cap),vertices[building_samples],colors[building_samples]) for cluster in building_clusters]
    trees=[]
    for cluster in tree_clusters:
        item=tree(cluster,height_cap)
        trees.append(transfer_video_colors(item,vertices[tree_samples],colors[tree_samples],item.metadata.get('trunk_vertices',0)))
    combined=trimesh.util.concatenate([base,*houses,*trees])
    ids=np.zeros(len(combined.vertices),np.uint8);ids[:len(base.vertices)]=base_ids;offset=len(base.vertices)
    for item in houses:ids[offset:offset+len(item.vertices)]=2;offset+=len(item.vertices)
    for item in trees:ids[offset:offset+len(item.vertices)]=3;offset+=len(item.vertices)
    return combined,ids,len(houses),len(trees)
