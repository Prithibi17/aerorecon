import importlib.util
import unittest
import numpy as np
from pipeline.semantic_fusion import refine_depth, choose_corrected_depth, fit_field_plane, level_field_pose


@unittest.skipUnless(importlib.util.find_spec('scipy'), 'Run in the AI environment for SciPy')
class DepthRefinementTest(unittest.TestCase):
    def test_smooth_depth_bias_improves_withheld_tracks(self):
        rng=np.random.default_rng(5)
        y,x=np.indices((240,400));truth=5+.01*x+.003*y
        predicted=truth*(1+.3*np.sin(x/70))
        pixels=np.column_stack([rng.integers(0,400,600),rng.integers(0,240,600)])
        corrected,used,before,after=refine_depth(predicted,pixels,truth[pixels[:,1],pixels[:,0]])
        self.assertTrue(used)
        self.assertLess(after,before*.85)
        self.assertTrue(np.isfinite(corrected).all() and (corrected>0).all())

    def test_correct_geometry_is_not_adjusted(self):
        y,x=np.indices((100,160));truth=5+.01*x
        pixels=np.column_stack([x.ravel()[::17],y.ravel()[::17]])
        corrected,used,_,_=refine_depth(truth,pixels,truth[pixels[:,1],pixels[:,0]])
        self.assertFalse(used)
        np.testing.assert_array_equal(corrected,truth)

    def test_domain_correction_is_gated_by_withheld_tracks(self):
        y,x=np.indices((80,120));truth=4+.02*x+.01*y
        bad=truth*(1+.2*np.sin(x/20));pixels=np.column_stack([x.ravel()[::7],y.ravel()[::7]])
        selected,used,before,after=choose_corrected_depth(bad,truth,pixels,truth[pixels[:,1],pixels[:,0]])
        self.assertTrue(used);self.assertLess(after,before*.97)
        np.testing.assert_allclose(selected,truth)
        selected,used,before,after=choose_corrected_depth(truth,bad,pixels,truth[pixels[:,1],pixels[:,0]])
        self.assertFalse(used);np.testing.assert_allclose(selected,truth)

    def test_flat_field_pose_maps_ground_to_shared_plane(self):
        h,w=90,160;k=np.array([[120.,0,80],[0,120,45],[0,0,1.]])
        y,x=np.indices((h,w));rays=np.stack([x,y,np.ones_like(x)],-1)@np.linalg.inv(k).T
        normal=np.array([0.,-.6,-.8]);normal/=np.linalg.norm(normal)
        depth=-3/(rays@normal);valid=np.isfinite(depth)&(depth>0)
        state={};pose,centre,used=level_field_pose(depth.astype(np.float32),valid,valid,k,np.eye(4),state,0)
        self.assertTrue(used)
        points=(rays*depth[...,None])[valid]
        world=(points-pose[:3,3])@pose[:3,:3]
        shared=state['normal'];origin=state['ground_origin']
        self.assertLess(np.percentile(np.abs((world-origin)@shared),99),1e-4)
