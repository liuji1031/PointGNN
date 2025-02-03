import numpy as np
from typing import Union
import torch
from einops import rearrange
from bbox import BoundingBox3D
from nuscenes.utils.data_classes import LidarPointCloud

def encode_loc_reg_target(reg_target:Union[np.ndarray,torch.Tensor],
                          box_xyz:Union[np.ndarray,torch.Tensor],
                          box_lhw:Union[np.ndarray,torch.Tensor],
                          box_r:Union[np.ndarray,torch.Tensor],
                          ref_box:BoundingBox3D,
                          points:LidarPointCloud,
                          mask:np.ndarray):
    """Encode the regression target for localization.

    Given a point cloud, and a mask that indicates the points within the ground 
    truth box, encode the regression target and stores them in reg_target

    Args:
        reg_target (np.ndarray): 2D array of regression targets
        box_xyz (np.ndarray): target box xyz
        box_lhw (np.ndarray): target box lhw
        box_r (np.ndarray): target box rotation
        ref_box (BoundingBox3D): reference box
        points (LidarPointCloud): point cloud
        mask (np.ndarray): mask that indicates the points within the ground truth box
    """
    # x, y, z target
    if box_xyz.ndim == 1:
        box_xyz = rearrange(box_xyz,'d -> 1 d')
    if isinstance(box_xyz, torch.Tensor):
        ref_box_lhw = ref_box.lwh_tensor
        log = torch.log
        points_xyz = torch.from_numpy(points.points[:3,:].T)
    else:
        ref_box_lhw = ref_box.lwh_np
        log = np.log
        points_xyz = points.points[:3,:].T
    reg_target[mask,:3] = ((box_xyz - points_xyz)/ref_box_lhw)[mask,:]

    # l, w, h target
    if box_lhw.ndim == 1:
        box_lhw = rearrange(box_lhw,'d -> 1 d')
    reg_target[mask,3:6] = log(box_lhw/ref_box_lhw)

    # rotation target, scaled to [-1,1]
    if box_r.ndim == 1:
        box_r = rearrange(box_r,'d -> 1 d')
    if isinstance(box_r, torch.Tensor):
        atan2, sin, cos, pi = torch.atan2, torch.sin, torch.cos,torch.pi
    else:
        atan2, sin, cos, pi = np.arctan2, np.sin, np.cos, np.pi
    reg_target[mask,[6]] = atan2(sin(box_r), cos(box_r))/pi


def decode_bbox(loc_output:torch.Tensor, ref_box:BoundingBox3D, points:LidarPointCloud):
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
    bbox = torch.zeros((n,7),dtype=torch.float32)
    point_xyz = torch.from_numpy(points.points[:3,:].T)

    # xyz
    bbox[:,:3] = loc_output[:,:3]*ref_box.tensor[[0],:3] + point_xyz

    # lwh
    bbox[:,3:6] = torch.exp(loc_output[:,3:6])*ref_box.tensor[[0],3:6]

    # rotation
    bbox[:,6] = loc_output[:,6]*torch.pi

    return bbox