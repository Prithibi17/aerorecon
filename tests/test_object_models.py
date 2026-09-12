import unittest
import importlib.util
import tempfile
from pathlib import Path

import numpy as np

if importlib.util.find_spec('trimesh') is None:
    import pytest
    pytest.skip('object-model dependencies are installed in the AI environment', allow_module_level=True)

from pipeline.object_models import components, house, solidify, tree
from pipeline.texture_atlas import bake_orthophoto


class ObjectModelTests(unittest.TestCase):
    def test_semantic_clusters_become_closed_house_and_tree_models(self):
        ground = np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], float)
        grid = np.array([[x * .06, .35, z * .06] for x in range(5) for z in range(5)])
        buildings = grid + np.array([-.65, 0, 0])
        vegetation = grid + np.array([.45, .05, 0])
        vertices = np.vstack([ground, buildings, vegetation])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        ids = np.r_[np.ones(4, np.uint8), np.full(25, 2, np.uint8), np.full(25, 3, np.uint8)]
        colors = np.tile([120, 150, 110], (len(vertices), 1)).astype(np.uint8)

        model, model_ids, houses, trees = solidify(vertices, faces, colors, ids, .8)

        self.assertEqual((houses, trees), (1, 1))
        self.assertTrue(np.any(model_ids == 2))
        self.assertTrue(np.any(model_ids == 3))
        self.assertEqual(len(model.vertices), len(np.unique(model.faces)))
        self.assertGreaterEqual(model.bounds[0, 1], 0)
        self.assertLessEqual(model.bounds[1, 1], .8)
        building_colours=np.asarray(model.visual.vertex_colors)[model_ids==2,:3]
        self.assertLess(np.abs(building_colours-[120,150,110]).mean(),1)

    def test_giant_connected_sheet_is_rejected(self):
        sheet = np.array([[x * .06, .2, z * .06] for x in range(30) for z in range(30)])
        self.assertEqual(components(sheet), [])

    def test_each_procedural_object_is_closed(self):
        cluster = np.array([[x * .06, .4, z * .06] for x in range(5) for z in range(5)])
        self.assertTrue(house(cluster, .8).is_watertight)
        tree_mesh = tree(cluster, .8)
        self.assertTrue(all(part.is_watertight for part in tree_mesh.split(only_watertight=False)))
        self.assertGreaterEqual(tree_mesh.bounds[0, 1], -1e-8)
        self.assertLessEqual(tree_mesh.bounds[1, 1], .8)

    @unittest.skipIf(importlib.util.find_spec('scipy') is None, 'texture baker needs scipy')
    def test_orthophoto_has_uvs_and_real_pixels(self):
        points=np.array([[0,0,0],[1,0,0],[0,0,1],[1,0,1]],float)
        colors=np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,255]],np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'texture.png'
            uv,info=bake_orthophoto(points,colors,points,target,resolution=32)
            self.assertTrue(target.is_file())
            self.assertEqual(uv.shape,(4,2))
            self.assertTrue(np.all((uv>=0)&(uv<=1)))
            self.assertEqual(info['samples'],4)


if __name__ == '__main__':
    unittest.main()
