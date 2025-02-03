import itertools
import typing

import numpy as np
import torch
from einops import rearrange
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
from typing_extensions import Self


class BoundingBox3D(Box):
    """
    class describing a bounding box
    """

    def __init__(
        self, x: float, y: float, z: float, w: float, l: float, h: float, r: float
    ):
        """initialize new bounding box

        Args:
            x (float): x coordinate of the center of the bounding box.
            y (float): y coordinate of the center of the bounding box.
            z (float): z coordinate of the center of the bounding box.
            l (float): length of the bounding box.
            w (float): width of the bounding box.
            h (float): height of the bounding box.
            r (float): rotation angle of the bounding box around the z-axis in radian.
        """
        super().__init__(
            center=[x, y, z],
            size=[w, l, h],
            orientation=Quaternion(axis=[0, 0, 1], angle=r),
        )
        # create shapely polygon, for iou calculation
        self.polygon = self._get_2d_polgon()
        self.numpy = self.to_numpy()
        self.tensor = self.to_tensor()

    @classmethod
    def from_nuscene_box(cls, box: Box):
        """initialize a bounding box from a nuscenes Box object

        Args:
            box (Box): nuscenes Box object

        Returns:
            BoundingBox3D: a BoundingBox3D object
        """
        return cls(
            box.center[0],
            box.center[1],
            box.center[2],
            box.wlh[0],
            box.wlh[1],
            box.wlh[2],
            box.orientation.yaw_pitch_roll[0],
        )

    def _get_2d_polgon(self):
        """get 2d polygon of the bounding box

        Returns:
            GeoT@translate: the translated and rotated 2d polygon of the bounding box
        """
        l, w = self.l, self.w
        tmp = Polygon(
            [[-l / 2, -w / 2], [-l / 2, w / 2], [l / 2, w / 2], [l / 2, -w / 2]]
        )
        tmp = rotate(tmp, self.r, origin=(0, 0), use_radians=True)
        tmp = translate(tmp, xoff=self.x, yoff=self.y)
        return tmp

    @property
    def x(self):
        return self.center[0]

    @property
    def y(self):
        return self.center[1]

    @property
    def z(self):
        return self.center[2]

    @property
    def w(self):
        return self.wlh[0]

    @w.setter
    def w(self, value: float):
        self.wlh[0] = value

    @property
    def l(self):
        return self.wlh[1]

    @l.setter
    def l(self, value: float):
        self.wlh[1] = value

    @property
    def h(self):
        return self.wlh[2]

    @h.setter
    def h(self, value: float):
        self.wlh[2] = value

    @property
    def r(self):
        """Yaw angle of the bounding box."""
        return self.orientation.yaw_pitch_roll[0]

    @property
    def xyz_np(self):
        return self.numpy[[0], :3]

    @property
    def lwh_np(self):
        return self.numpy[[0], 3:6]

    @property
    def r_np(self):
        return self.numpy[[0], -1:]

    @property
    def xyz_tensor(self):
        return self.tensor[[0], :3]

    @property
    def lwh_tensor(self):
        return self.tensor[[0], 3:6]

    @property
    def r_tensor(self):
        return self.tensor[[0], -1:]

    def to_tensor(self):
        """convert the bounding box to a tensor

        Returns:
            torch.Tensor: a 1x7 tensor representing the bounding box
        """
        return torch.tensor([self.x, self.y, self.z, self.l, self.w, self.h, self.r])[
            torch.newaxis, :
        ]

    def to_numpy(self) -> np.ndarray:
        """convert the bounding box to a numpy array

        Returns:
            np.ndarray: a 1x7 numpy array representing the bounding box
        """
        return np.array([self.x, self.y, self.z, self.l, self.w, self.h, self.r])[
            np.newaxis, :
        ]
    
    def update(self):
        """Update the bounding box after changing the attributes."""
        self.polygon = self._get_2d_polgon()
        self.numpy = self.to_numpy()
        self.tensor = self.to_tensor()

    def find_gt_box_normals(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        given a bounding box description (x, y, z, l, w, h, r), return the
        normals of the box faces, as well as the lower and upper bound of
        projected values of the box corners onto the faces.

        Returns:
            np.ndarray: a 3x3 array with each column representing the normals of
            the box faces
            np.ndarray: a 3x2 array representing the lower and upper bound of
                        projected values of the box corners onto. each row is
                        the upper and lower bound of 1 dimension, not sorted
        """
        # compute the normals of the box faces. the columns of the rotation matrix
        # are the normals of the box faces
        R = np.array(
            [
                [np.cos(self.r), -np.sin(self.r), 0],
                [np.sin(self.r), np.cos(self.r), 0],
                [0, 0, 1],
            ]
        )

        # get 2 extreme corners
        corners = np.array(
            [
                [self.l / 2, self.w / 2, self.h / 2],
                [-self.l / 2, -self.w / 2, -self.h / 2],
            ]
        ).T

        corners_ = np.dot(R, corners)
        corners_ += np.array([[self.x, self.y, self.z]]).T

        # compute the projections of the corners
        proj = np.dot(R.T, corners_)  # 3x2

        return R, proj

    def within_gt_box(self, points: np.ndarray) -> np.ndarray:
        """check if a point is within the bounding box

        Args:
            points (np.ndarray): a Nx3 array representing the points to be
            checked

        Returns:
            np.ndarray: a 1D boolean array with each element representing
            whether the corresponding point is within the bounding box
        """
        R, proj_bounds = self.find_gt_box_normals()

        # project the points onto box normal
        proj = np.dot(R.T, points.T)  # 3xN

        # compute lower and upper bound of the projected points
        min_val = np.min(proj_bounds, axis=1, keepdims=True)  # 3x1
        max_val = np.max(proj_bounds, axis=1, keepdims=True)  # 3x1

        # check if the point is within the box, i.e., all 3 boolean values along
        # a column are True
        return np.all((proj >= min_val) & (proj <= max_val), axis=0, keepdims=False)

    def iou_2D(self, other: Self) -> float:
        """compute the 2D intersection over union between two bounding boxes

        Args:
            other (Box): another bounding box

        Returns:
            float: 2D iou considering the rotation
        """
        # reference: https://medium.com/mindkosh/calculating-iou-between-oriented-bounding-boxes-c39f72602cac
        intersection_area = self.polygon.intersection(other.polygon).area
        union_area = self.polygon.union(other.polygon).area
        return intersection_area / union_area

    def flip_x(self):
        """flip the bounding box along the x-axis"""
        self.center[0] = -self.center[0]
        self.orientation = Quaternion(axis=[0, 0, 1], angle=np.pi - self.r)

    def flip_y(self):
        """flip the bounding box along the y-axis"""
        self.center[1] = -self.center[1]
        self.orientation = Quaternion(axis=[0, 0, 1], angle=-self.r)
