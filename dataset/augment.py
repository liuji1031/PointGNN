from typing import List, Union

import numpy as np
import torch
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from dataset.bbox import BoundingBox3D


class AugmentRegistry:
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


@AugmentRegistry.register("random_jitter")
class RandomJitter(BaseTransform):
    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def forward(self, data: Data):
        assert "sensor_loc" in data
        sensor_loc = data.sensor_loc
        data.pos = self._random_jitter(data.pos, sensor_loc)
        return data

    def _random_jitter(self, points: torch.Tensor, sensor_loc: torch.Tensor):
        """Randomly jitter the points in the point cloud.

        Args:
            points (torch.Tensor): N x 3 array of point cloud expressd in the ego (car) frame
            sensor_loc (torch.Tensor): sensor location in the ego frame
        """
        assert sensor_loc.ndim == 2
        assert points.shape[1] == 3
        # jitter along the reflection direction
        tmp = points - sensor_loc
        d = torch.linalg.norm(tmp, dim=1, keepdim=True)  # n x 1
        u = tmp / d  # unit vector, n x 3
        jitter = torch.randn_like(d) * self.sigma  # n x 1
        points += u * jitter  # add jitter along the reflection direction
        return points


@AugmentRegistry.register("random_rotation")
class RandomRotation(BaseTransform):
    def __init__(self, sigma: float = np.pi / 8):
        self.sigma = sigma

    def forward(self, data: Data):
        assert "sensor_loc" in data
        assert "gt_boxes" in data
        data.pos = self._random_rotation(data.pos, data.gt_boxes)
        return data

    def _random_rotation(self, points: torch.Tensor, gt_boxes: List[BoundingBox3D]):
        """Randomly rotate the points in the point cloud.

        Args:
            points (torch.Tensor): point cloud expressd in the ego (car) frame
        """
        assert points.shape[1] == 3
        # rotation matrix
        r = np.random.randn() * self.sigma
        R = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
        R = torch.from_numpy(R).float()
        points = torch.matmul(R, points.T).t()

        # rotate gt boxes
        q = Quaternion(axis=[0, 0, 1], angle=r)
        for box in gt_boxes:
            box.rotate(q)
            box.update()
        return points


@AugmentRegistry.register("random_flip_y")
class RandomFlipY(BaseTransform):
    """Randomly flip the points in the point cloud along the y-axis."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def forward(self, data: Data):
        assert "sensor_loc" in data
        assert "gt_boxes" in data
        data.pos = self._random_flip_y(data.pos, data.gt_boxes)
        return data

    def _random_flip_y(self, points: torch.Tensor, gt_boxes: List[BoundingBox3D]):
        """Randomly flip the points in the point cloud along the y-axis.

        Args:
            points (torch.Tensor): N x 3 array of point cloud expressd in the ego (car) frame
        """
        if np.random.rand() < self.prob:
            points[:, 1] *= -1  # flip along y-axis

            # flip gt boxes
            for box in gt_boxes:
                box.flip_y()
                box.update()

        return points


if __name__ == "__main__":
    print(AugmentRegistry.REGISTRY)
    pc_test = LidarPointCloud(np.random.randn(4, 100))
    sensor_loc_test = np.random.randn(3)[np.newaxis, :]

    for aug_name, aug_cls in AugmentRegistry.REGISTRY.items():
        aug = aug_cls()
        pc_aug = aug.augment(pc_test, sensor_loc=sensor_loc_test)
