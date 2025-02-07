from typing import Union

import numpy as np
import torch
from einops import rearrange
from nuscenes.utils.data_classes import LidarPointCloud

from dataset.bbox import BoundingBox3D


def encode_loc_reg_target(
    box_xyz: Union[np.ndarray, torch.Tensor],
    box_lwh: Union[np.ndarray, torch.Tensor],
    box_r: Union[np.ndarray, torch.Tensor],
    ref_box_lwh: Union[np.ndarray, torch.Tensor],
    pos: Union[np.ndarray, torch.Tensor],
):
    """Encode the regression target for localization.

    Given a point cloud, and a mask that indicates the points within the ground
    truth box, encode the regression target and stores them in reg_target

    Args:
        reg_target (np.ndarray): 2D array of regression targets
        box_xyz (np.ndarray): target box xyz
        box_lwh (np.ndarray): target box lhw
        box_r (np.ndarray): target box rotation
        ref_box (BoundingBox3D): reference box
        points (LidarPointCloud): point cloud
        mask (np.ndarray): mask that indicates the points within the ground truth box
    """
    assert pos.shape[-1] == 3, "Position must have 3 dimensions"
    # x, y, z target
    if isinstance(box_xyz, torch.Tensor):
        log, atan2, sin, cos, pi = torch.log, torch.atan2, torch.sin, torch.cos, torch.pi
    else:
        log,atan2, sin, cos,pi = np.log,np.arctan2, np.sin, np.cos, np.pi
    reg_xyz = (box_xyz - pos) / ref_box_lwh
    reg_lwh = log(box_lwh / ref_box_lwh)
    reg_r = atan2(sin(box_r), cos(box_r)) / pi

    return reg_xyz, reg_lwh, reg_r


def decode_bbox(
    loc_output: torch.Tensor, ref_box: BoundingBox3D, points: LidarPointCloud
):
    """Decode the bounding box from the output of the network.

    Args:
        loc_output (torch.Tensor): output of the network. 2D tensor of shape N x 7
        ref_box (BoundingBox3D): reference box (e.g, average anchor box)
        points (LidarPointCloud): point cloud
        mask (np.ndarray): mask that indicates the points within the bounding box

    Returns:
        Tensor : 2D tensor of bounding boxes
    """
    n = loc_output.shape[0]
    bbox = torch.zeros((n, 7), dtype=torch.float32)
    point_xyz = torch.from_numpy(points.points[:3, :].T)

    # xyz
    bbox[:, :3] = loc_output[:, :3] * ref_box.tensor[[0], :3] + point_xyz

    # lwh
    bbox[:, 3:6] = torch.exp(loc_output[:, 3:6]) * ref_box.tensor[[0], 3:6]

    # rotation
    bbox[:, 6] = loc_output[:, 6] * torch.pi

    return bbox
