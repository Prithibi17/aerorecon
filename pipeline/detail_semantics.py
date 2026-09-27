"""Overlap-blended semantic crops and conservative multi-view detail labels.

Classes describe observed surfaces, not object instances or generated geometry.
"""
import numpy as np

NAMES = ('unknown', 'ground', 'building', 'tree', 'sky', 'vehicle',
         'low vegetation', 'field')
COLORS = np.array([[130,130,130], [157,125,84], [255,166,56], [29,110,58],
                   [95,180,255], [235,65,82], [123,201,65], [208,194,63]], dtype=np.uint8)
COARSE = np.array([0,1,2,3,4,0,3,1], dtype=np.uint8)


def label_lookup(id2label):
    groups = ({'road','earth','grass','path','land','sand','dirt track'},
              {'building','house','wall','skyscraper'}, {'tree','palm'}, {'sky'},
              {'car','truck','bus','van'}, {'plant','bush','shrub','flower'}, {'field'})
    lookup = np.zeros(max(map(int, id2label)) + 1, dtype=np.uint8)
    for key, name in id2label.items():
        for index, names in enumerate(groups, 1):
            if name.lower().strip() in names:
                lookup[int(key)] = index
                break
    return lookup


def tile_starts(length, size, overlap):
    if size < 32 or not 0 <= overlap < size:
        raise ValueError('Tile size must be >=32 with 0 <= overlap < size')
    end = max(length-size, 0)
    return sorted(set([*range(0, end+1, size-overlap), end]))


def tiled_probabilities(rgb, predict, size=512, overlap=128):
    """predict returns HxWx8 probabilities. Blend crops with full-view context."""
    height, width = rgb.shape[:2]
    context = predict(rgb)
    if max(height, width) <= size:
        return context
    sums = np.zeros_like(context, dtype=np.float32)
    weights = np.zeros((height, width, 1), np.float32)
    for y in tile_starts(height, size, overlap):
        for x in tile_starts(width, size, overlap):
            crop = rgb[y:y+size, x:x+size]
            h, w = crop.shape[:2]
            weight = np.maximum(np.outer(np.hanning(h), np.hanning(w)), .05)[...,None]
            sums[y:y+h, x:x+w] += predict(crop)*weight
            weights[y:y+h, x:x+w] += weight
    return .25*context + .75*sums/weights


def consensus(votes, minimum=2, agreement=.6):
    """Unknown votes count against agreement; isolated detections stay unknown."""
    labels = votes.argmax(1)
    support = votes[np.arange(len(votes)), labels]
    reliable = (support >= minimum) & (support/np.maximum(votes.sum(1), 1) >= agreement)
    return np.where(reliable, labels, 0).astype(np.uint8)


def scaled_intrinsics(matrix, old_width, old_height, new_width, new_height):
    result = np.asarray(matrix, dtype=float).copy()
    result[0] *= new_width/old_width
    result[1] *= new_height/old_height
    return result
