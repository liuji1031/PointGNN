# use ransac to remove the ground points
import copy
import random

import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud


def estimate(xyz):
    """Get the plane coefficients from xyz."""
    axyz = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    return np.linalg.svd(axyz)[-1][[-1], :]


# https://github.com/falcondai/py-ransac/blob/master/ransac.py
def run_ransac(
    data, inlier_threshold, sample_size, max_iterations, random_seed=None, verbose=False
):
    """_summary_

    Args:
        data (_type_): _description_
        estimate (_type_): _description_
        is_inlier (bool): _description_
        sample_size (_type_): _description_
        goal_inliers (_type_): _description_
        max_iterations (_type_): _description_
        stop_at_goal (bool, optional): _description_. Defaults to True.
        random_seed (_type_, optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to False.
    Returns:
        _type_: _description_
    """
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    for i in range(max_iterations):
        # sample sample_size points from data
        s = np.random.choice(data.shape[0], sample_size, replace=False)
        m = estimate(data[s])  # 1 x 4
        inlier_count = 0
        aug_data = np.hstack([data, np.ones((len(data), 1))])
        projection = np.dot(aug_data, m.T)  # n x 1

        inlier_count = np.sum(np.abs(projection) < inlier_threshold)
        if inlier_count > best_ic:
            best_ic = inlier_count
            best_model = m
    if verbose:
        print(
            "took iterations:", i + 1, "best model:", best_model, "explains:", best_ic
        )
    return best_model, best_ic


def ground_removal(
    pc: LidarPointCloud,
    inlier_threshold=0.2,
    sample_size=10,
    max_iterations=100,
    random_seed=0,
    restrict_range=20,
    inplace=False,
):
    """Remove ground from point cloud using RANSAC.

    Args:
        pc (LidarPointCloud): _description_
        inlier_threshold (float, optional): _description_. Defaults to 0.2.
        sample_size (int, optional): _description_. Defaults to 3.
        max_iterations (int, optional): _description_. Defaults to 1000.
        random_seed (int, optional): _description_. Defaults to 0.
        restrict_range (int, optional): use points within the restrict_range for
        RANSAC calculation. Defaults to 20.
        inplace (bool, optional): _description_. Defaults to False.
    Returns:
        _type_: _description_
    """
    data = pc.points[:3, :].T
    # restrict range
    mask1 = np.logical_and(
        np.abs(data[:, 0]) < restrict_range, np.abs(data[:, 1]) < restrict_range
    )
    model, _ = run_ransac(
        data[mask1], inlier_threshold, sample_size, max_iterations, random_seed
    )

    # remove ground points
    aug_data = np.hstack([data, np.ones((data.shape[0], 1))])
    projection = np.dot(aug_data, model.T)  # n x 1
    mask2 = np.ravel(np.abs(projection) < inlier_threshold)
    mask = (1 - (mask1 & mask2)).astype(bool)

    if not inplace:
        pc_ = copy.deepcopy(pc)
        pc_.points = pc_.points[:, mask]
        return pc_
    else:
        pc.points = pc.points[:, mask]
        return pc
