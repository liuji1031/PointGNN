# Description: This file contains the abstract class for preprocessing point cloud data.
from abc import ABC, abstractmethod

import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud

from dataset.ground_removal import ground_removal


class PreprocessRegistry:
    REGISTRY = {}

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            if name in cls.REGISTRY:
                raise ValueError(f"Cannot register duplicate class ({name})")
            cls.REGISTRY[name] = subclass
            return subclass

        return decorator

    @classmethod
    def build(cls, name: str, config: dict):
        if name not in cls.REGISTRY:
            raise ValueError(f"Unknown class {name}")
        return cls.REGISTRY[name](**config)


class PreprocessPointCloud(ABC):
    @abstractmethod
    def preprocess(self, pc: LidarPointCloud, **kwargs):
        pass


@PreprocessRegistry.register("ground_removal_ransac")
class GroundRemovalRANSAC(PreprocessPointCloud):
    def __init__(
        self,
        inlier_threshold=0.2,
        sample_size=10,
        max_iterations=100,
        random_seed=0,
        restrict_range=20,
        inplace=True,
    ):
        self.inlier_threshold = inlier_threshold
        self.sample_size = sample_size
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.restrict_range = restrict_range
        self.inplace = inplace

    def preprocess(self, pc: LidarPointCloud, **kwargs):
        return ground_removal(
            pc,
            self.inlier_threshold,
            self.sample_size,
            self.max_iterations,
            self.random_seed,
            self.restrict_range,
            self.inplace,
        )


@PreprocessRegistry.register("voxel_downsample")
class VoxelDownsample(PreprocessPointCloud):
    """Reduce the number of points by voxelization, i.e., returns the average of
    points in each voxel.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        x_range,
        y_range,
        z_range,
        voxel_size,
    ):
        dx, dy, dz = voxel_size["x"], voxel_size["y"], voxel_size["z"]

        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range

        nx = int(np.round((x_max - x_min) / dx))
        ny = int(np.round((y_max - y_min) / dy))
        nz = int(np.round((z_max - z_min) / dz))

        # make the range integer copy of the interval
        x_max = x_min + nx * dx
        y_max = y_min + ny * dy
        z_max = z_min + nz * dz

        self.x_range = [x_min, x_max]
        self.y_range = [y_min, y_max]
        self.z_range = [z_min, z_max]
        self.voxel_size_xyz = np.array([dx, dy, dz])[np.newaxis, :]
        self.lower_limit = np.array([x_min, y_min, z_min])[np.newaxis, :]

    def preprocess(self, pc: LidarPointCloud, **kwargs):
        points = pc.points.T
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        z_min, z_max = self.z_range

        # exclude out of range points
        mask = (
            (x >= x_min)
            & (x < x_max)
            & (y >= y_min)
            & (y < y_max)
            & (z >= z_min)
            & (z < z_max)
        )
        points = points[mask, :]

        # compute voxel index
        coord = ((points[:, :3] - self.lower_limit) / self.voxel_size_xyz).astype(
            np.int32
        )

        unique_coord, inverse_ind, counts = np.unique(
            coord, return_inverse=True, return_counts=True, axis=0
        )

        # sort by the inverse index, so it matches with the unique_corrd
        isort = np.argsort(inverse_ind)
        pc_sort = points[isort, :]

        # compute the mean of points in each voxel
        # if counts is [0,10,20,30], then ind_slice is [0,10,30,50]
        ind_slice = np.concatenate([np.array([0]), np.cumsum(counts)])[:-1]

        # reduceat is a numpy function that computes the result of each slice
        # , given the indices that define the boundary of each slice
        pc_mean = np.add.reduceat(pc_sort, ind_slice, axis=0) / counts[:, np.newaxis]

        # update the point cloud
        pc.points = pc_mean.T


if __name__ == "__main__":
    print(PreprocessRegistry.REGISTRY)
