import pickle
import numpy as np
from math import factorial, ceil
from random import sample
from itertools import combinations
from torch.utils.data import Dataset


def load_features(type, label, file_path):
    infile = open(file_path, 'rb')
    return pickle.load(infile)


def get_sample_features(features, total_n, k, verbose=False):
    total_patches = sum([len(patches) for _, patches in features.items()])
    sample_ratio = total_n / total_patches
    sample_features = []
    for wsi, patches in features.items():
        patches = list(patches.values())
        n = max(ceil(len(patches) * sample_ratio), 1)
        ret = get_random_n_sets_of_k_patches(patches, n, k, True)
        if verbose:
            print("{}: {} : {} : {}".format(wsi, len(patches), n, len(ret)))
        sample_features.extend(ret)
    return sample_features


def get_random_n_sets_of_k_patches(patches, n, k, rep=True):
    """
    patches: list of 2D-array
    n: # of sets
    k: # of patches per set
    """
    num_patches = len(patches)
    if num_patches < k and not rep:
        return [np.vstack(patches)]
    elif num_patches < k:
        return [np.vstack(patches)] * n
    num_combs = factorial(num_patches) / \
        factorial(num_patches - k) / factorial(k)
    indices = list(range(num_patches))
    if (num_combs < n) and not rep:
        combs = list(combinations(indices, k))
    else:
        combs = [sample(indices, k) for _ in range(n)]
    ret = []
    for c in combs:
        ret_patches = []
        for i in c:
            ret_patches.append(patches[i])
        ret.append(np.vstack(ret_patches))
    return ret


def pad_features(features, k):
    dim = features[0].shape[1]
    padded_features = np.zeros((len(features), k, dim))
    for i, feat in enumerate(features):
        padded_features[i, :len(feat), :] = feat
    return padded_features
