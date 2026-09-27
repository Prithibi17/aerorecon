import unittest
import numpy as np
from pipeline.detail_semantics import (label_lookup, tiled_probabilities,
    consensus, scaled_intrinsics, tile_starts, COARSE)


class DetailTests(unittest.TestCase):
    def test_classes_do_not_merge_cars_with_buildings(self):
        labels = label_lookup({0:'car', 1:'house', 2:'plant', 3:'tree', 4:'field', 5:'sky'})
        np.testing.assert_array_equal(labels, [5,2,6,3,7,4])
        np.testing.assert_array_equal(COARSE[labels], [0,2,3,3,1,4])

    def test_overlap_reconstructs_probabilities_at_edges(self):
        image = np.zeros((79,121,3), np.uint8)
        image[15:20,52:57,0] = 255  # Small object spans crop boundaries.
        def predict(crop):
            result = np.zeros((*crop.shape[:2],8),np.float32)
            result[...,5] = crop[...,0]/255
            result[...,1] = 1-result[...,5]
            return result
        result = tiled_probabilities(image,predict,size=32,overlap=12)
        np.testing.assert_allclose(result,predict(image),atol=1e-6)
        np.testing.assert_allclose(result.sum(2),1,atol=1e-6)

    def test_single_view_and_disagreement_are_unknown(self):
        votes = np.zeros((4,8),int)
        votes[0,5]=1
        votes[1,5]=3; votes[1,2]=1
        votes[2,5]=2; votes[2,2]=2
        np.testing.assert_array_equal(consensus(votes),[0,5,0,0])

    def test_projection_scales_with_image_not_depth(self):
        matrix=np.array([[500,0,320],[0,500,180],[0,0,1]])
        point=np.array([.4,.2,2])
        before=matrix@point
        after=scaled_intrinsics(matrix,640,360,1280,720)@point
        np.testing.assert_allclose(after[:2]/after[2],2*before[:2]/before[2])
        self.assertEqual(matrix[0,0],500)

    def test_bad_overlap_rejected(self):
        with self.assertRaises(ValueError):
            tile_starts(100,32,32)


if __name__ == '__main__':
    unittest.main()
