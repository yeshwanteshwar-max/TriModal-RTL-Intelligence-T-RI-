import numpy as np


def fuse_features(*feature_maps: dict[str, int]) -> np.ndarray:
    keys = sorted(set().union(*(feature.keys() for feature in feature_maps)))
    return np.array([[features.get(key, 0) for key in keys] for features in feature_maps],
                    dtype=np.float32).reshape(-1)
