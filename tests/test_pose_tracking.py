import unittest

import cv2
import numpy as np

from pipeline.pose_tracking import frame_number,solve_pose


class PoseTrackingTests(unittest.TestCase):
    def test_frame_number(self):
        self.assertEqual(frame_number('frame_00000585.jpg'),585)

    def test_recovers_world_to_camera_pose(self):
        rng=np.random.default_rng(4)
        world=rng.uniform([-2,-1,4],[2,1,10],size=(200,3)).astype(np.float32)
        rotation=cv2.Rodrigues(np.array([.02,-.08,.03]))[0]
        translation=np.array([.3,-.1,.7])
        camera=world@rotation.T+translation
        intrinsic=np.array([[700,0,320],[0,700,180],[0,0,1]],dtype=float)
        pixels=camera@intrinsic.T;pixels=(pixels[:,:2]/pixels[:,2:]).astype(np.float32)
        pose,inliers=solve_pose(world,pixels,intrinsic)
        self.assertGreater(len(inliers),190)
        self.assertLess(np.max(np.abs(pose[:3,:3]-rotation)),1e-4)
        self.assertLess(np.max(np.abs(pose[:3,3]-translation)),1e-4)


if __name__=='__main__':unittest.main()
