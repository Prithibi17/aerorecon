import numpy as np
from pipeline.ai_reconstruct import project_depth
from pipeline.video_fusion import similarity
from pipeline.semantic_fusion import ground_rotation


def test_ground_alignment_is_rigid_and_preserves_height():
    for normal in ([.2,.8,.3],[0,-1,0],[0,1,0]):
        n=np.array(normal,dtype=float);n/=np.linalg.norm(n)
        r=ground_rotation(n)
        np.testing.assert_allclose(r@n,[0,1,0],atol=1e-10)
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(r),1,atol=1e-10)
        base=np.array([4.,5.,2.]);roof=base-3*n
        transformed=np.array([base,roof])@r.T
        np.testing.assert_allclose(transformed[1,1]-transformed[0,1],-3,atol=1e-10)


def test_batch_alignment_recovers_world_transform_with_outliers():
    rng=np.random.default_rng(3)
    source=rng.normal(size=(1000,3))
    rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    target=2.5*source@rotation.T+[4,-2,7]
    target[:80]+=rng.normal(size=(80,3))*10
    scale,r,t,error=similarity(source,target)
    np.testing.assert_allclose(scale,2.5,atol=1e-6)
    np.testing.assert_allclose(r,rotation,atol=1e-6)
    np.testing.assert_allclose(t,[4,-2,7],atol=1e-6)
    assert error<1e-6


def test_depth_backprojection_and_world_camera_roundtrip():
    k=np.array([[100.,0,1],[0,100,1],[0,0,1]])
    rotation=np.array([[0.,0,1],[0,1,0],[-1,0,0]])
    translation=np.array([2.,-1,3])
    extrinsic=np.column_stack([rotation,translation])
    depth=np.full((3,3),5.)
    world=project_depth(depth,k,extrinsic)
    camera=world@rotation.T+translation
    np.testing.assert_allclose(camera[...,2],5)
    projected=camera@k.T
    projected=projected[...,:2]/projected[...,2:]
    y,x=np.indices((3,3))
    np.testing.assert_allclose(projected,np.stack([x,y],-1),atol=1e-12)
