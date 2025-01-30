import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
import torch
import torch.utils.data as data
from pyquaternion import Quaternion
from bbox import BoundingBox3D
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix


class NuScenesDataset(data.Dataset):
    def __init__(self, data_root, version='v1.0-mini',
                 train_val_test_split=(0.8, 0.1, 0.1),
                 seed=0,
                 anchor_box_config=None,
                 verbose=False):
        self.data_root = data_root
        self.version = version
        self.verbose = verbose
        self.nusc = NuScenes(version=version, dataroot=data_root, verbose=verbose)
        self.scenes = self.nusc.scene
        self.sample_tokens_by_scene = self._get_sample_tokens_by_scene()
        self.train_tokens, self.val_tokens, self.test_tokens = \
            self._split_train_val_test(train_val_test_split, seed)
        
        self.anchor_boxes, self.category_list = self._gen_anchor_boxes(anchor_box_config)
        # build a dictionary for category and sub-category. example output:
        # {"vehicle": ["car", "truck", "bus"]}
        self.category_dict = self._parse_category()

        # store anchor box information into a torch array
        if len(self.anchor_boxes):
            self.anchor_boxes_tensor = []
            for box in self.anchor_boxes:
                box:BoundingBox3D
                self.anchor_boxes_tensor.append(box.to_tensor())
            self.anchor_boxes_tensor = torch.stack(self.anchor_boxes_tensor,dim=0) # n_anchor_box x 7
            self.anchor_boxes_tensor.requires_grad = False
        
        # get an average size of the anchor boxes
        self.avg_anchor_box = self._get_average_anchor_box()
        self.avg_anchor_box_np = self.avg_anchor_box.to_numpy()

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        ...

    def _parse_category(self):
        """build a dictionary for category and sub-category

        Returns:
            dict: dictionary of category and sub-category. e.g., 
            {"vehicle": ["car", "truck", "bus"]}
        """
        category_dict = {}
        for c in self.category_list:
            if c == "background":
                continue
            cat,sub_cat = c.split(".")[:2]
            if cat not in category_dict:
                category_dict[cat] = []
            category_dict[cat].append(sub_cat)
        return category_dict
    
    def _in_category(self, category_name:str):
        """Check if the category is in the category list.

        Args:
            category_name (str): category name

        Returns:
            bool: True if the category is in the category list
            str: category name i.e,, <category>.<sub-category>
        """
        if len(self.category_list) == 0:
            return False, ""
        else:
            cat, sub_cat = category_name.split(".")[:2]
            if cat in self.category_dict and sub_cat in self.category_dict[cat]:
                return True, cat+"."+sub_cat
            else:
                return False, ""

    def _gen_anchor_boxes(self, anchor_box_config : Union[dict,None]):
        """Generate anchor boxes.

        Args:
            anchor_box_config (dict): configuration of anchor boxes

        Returns:
            list: list of BoundingBox3D objects
        """
        if anchor_box_config is None:
            return None
        
        anchor_boxes = []
        category_list = []
        for config in anchor_box_config:
            w,l,h = config["wlh"]
            bbox = BoundingBox3D(0,0,0,w,l,h,0)
            bbox.name = config["class"]
            anchor_boxes.append(bbox)
            category_list.append(config["class"])
        return anchor_boxes, category_list
    
    def _get_average_anchor_box(self):
        """Get the average of anchor boxes."""
        avg_box = BoundingBox3D(0,0,0,0,0,0,0)
        for box in self.anchor_boxes:
            avg_box.l += box.l
            avg_box.w += box.w
            avg_box.h += box.h
        avg_box.l /= len(self.anchor_boxes)
        avg_box.w /= len(self.anchor_boxes)
        avg_box.h /= len(self.anchor_boxes)
        return avg_box

    def _get_sample_tokens_by_scene(self) -> List[List[str]]:
        """Return all sample tokens in the dataset.

        Args:
        Returns:
            list: list of sample tokens by scene, i.e., 
            [[tokens from scene 1], [tokens from scene 2], ...]
        """
        sample_tokens_by_scene = []
        for scene in self.nusc.scene:
            tokens = []
            sample_token = scene["first_sample_token"]
            while sample_token != "":
                tokens.append(sample_token)
                sample_token = self.nusc.get("sample",sample_token)["next"]
            sample_tokens_by_scene.append(tokens)
        return sample_tokens_by_scene
    
    def _split_train_val_test(self,
                    train_val_test_split:Tuple[float] = (0.8,0.1,0.1),
                    seed:int = 0
                    ) -> Tuple[List[str]]:
        """Split the dataset into training and validation set.

        Args:
            val_split (float, optional): proportion of the validation set. Defaults to 0.2.
            seed (int, optional): random seed. Defaults to 0.

        Returns:
            list: list of sample tokens by scene
        """
        assert np.sum(train_val_test_split) == 1.0
        train_tokens = []
        val_tokens = []
        test_tokens = []
        for tokens in self.sample_tokens_by_scene:
            # split tokens from a single scene by train_val_test_split to ensure
            # stratified split
            # shuffle index
            n_samples = len(tokens)
            split_ind = (np.cumsum(train_val_test_split)*n_samples).astype(int)
            np.random.seed(seed)
            ind = np.random.permutation(n_samples)
            ind_train = ind[:split_ind[0]]
            ind_val = ind[split_ind[0]:split_ind[1]]
            ind_test = ind[split_ind[1]:]
            
            train_tokens += [tokens[k] for k in ind_train]
            val_tokens += [tokens[k] for k in ind_val]
            test_tokens += [tokens[k] for k in ind_test]
        return train_tokens, val_tokens, test_tokens
    
    def _get_category_name(self, ann_token):
        """get category name from annotation token

        Args:
            ann_token (str): annotation token

        Returns:
            str: category name
        """
        return self.nusc.get('sample_annotation', ann_token)["category_name"]
    
    def _get_gt_box_label(self, gt_box : Box):
        """Get the label of the ground truth box.

        Args:
            gt_box (Box): ground truth box

        Returns:
            int: label of the ground truth box
        """
        ...

    def _get_gt_boxes_from_sample(self, sample_token:str="",
                                  sample:Union[dict, None]=None,
                                  render=False) -> List[dict]:
        """Get the ground truth bounding boxes in a sample.

        Args:
            sample_token (str): sample token

        Returns:
            list: list of bounding box dictionaries
        """
        sample = sample if sample is not None else self.nusc.get("sample", sample_token)

        annotations = sample["anns"]
        _, boxes, _ = self.nusc.get_sample_data(sample['data']['LIDAR_TOP'],
                                       use_flat_vehicle_coordinates=True)
        
        gt_boxes = []
        for k, ann_token in enumerate(annotations):
            category_name = self._get_category_name(ann_token)
            
            # keep only category and sub-category name in "category_name"
            in_category, category_name = self._in_category(category_name)
            if in_category:
                gt_box = BoundingBox3D.from_nuscene_box(boxes[k])
                gt_box.name = category_name

                # also figure out the label, i.e., index of the category
                # within the category list
                gt_box.label = self.category_list.index(category_name)
                gt_boxes.append(gt_box)

                if render:
                    print(f"======{k}======")
                    self.nusc.render_annotation(ann_token)
                    plt.show()
                    print(gt_boxes[-1])
        return gt_boxes
    
    def _get_raw_lidar_pts(self, sample):
        """retrieves raw lidar points from filename stored in sample data

        Args:
            sample (dict): NuScenes sample dictionary

        Returns:
            LidarPointCloud: raw lidar points
        """
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_info = self.nusc.get('sample_data', lidar_token)
        
        lidar_path = os.path.join(self.nusc.dataroot,lidar_info["filename"])
        lidar_data = LidarPointCloud.from_file(str(lidar_path))
        return lidar_data
    
    def _transform_matrix_from_pose(self, pose:dict, inverse:bool):
        """wrapper function for getting the transformation matrix from pose dict

        Args:
            pose (dict): NuScenes pose dictionary
            inverse (bool): whether to get the inverse transformation matrix

        Returns:
            <np.float32: 4, 4>: 4 by 4 transformation matrix
        """
        return transform_matrix(pose['translation'],
                                Quaternion(pose['rotation']),
                                inverse=inverse)

    def _get_lidar_pts_singlesweep(self,
                                   sample_token:str="",
                                   sample:Union[dict, None]=None,
                                   sensor="LIDAR_TOP",
                                convert_to_ego_frame=True,
                                min_dist=1.0,
                                nkeep="all"):
        """returns the lidar points for a single sweep

        Args:
            sample_token (str, optional): sample token. Defaults to "".
            sample (Union[dict, None], optional): sample dictionary. Defaults to None.
            sensor (str, optional): sensor name. Defaults to "LIDAR_TOP".
            convert_to_ego_frame (bool, optional): whether to convert the points to ego frame. Defaults to True.
            min_dist (float, optional): minimum distance to remove points. Defaults to 1.0.
            nkeep (Union[str, int], optional): number of points to keep. Defaults to "all".

        Returns:
            LidarPointCloud: lidar points
        """
        sample = sample if sample is not None else self.nusc.get("sample", sample_token)
        sample_data = self.nusc.get("sample_data", sample["data"][sensor])

        pc = self._get_raw_lidar_pts(sample)
        pc.remove_close(min_dist)

        if not convert_to_ego_frame:
            return pc
        
        # get sensor ref car transformation
        pose = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        sensor_ref_car = self._transform_matrix_from_pose(pose,inverse=False)
        
        # convert sensor points to car body frame
        pc.transform(sensor_ref_car)

        return pc
    
    def _encode_loc_reg_target(self, reg_target:np.ndarray, gt_box:BoundingBox3D, points:LidarPointCloud, mask:np.ndarray):
        """Encode the regression target for localization.

        Given a point cloud, and a mask that indicates the points within the ground 
        truth box, encode the regression target and stores them in reg_target

        Args:
            reg_target (np.ndarray): 2D array of regression targets
            gt_box (BoundingBox3D): ground truth box
            points (LidarPointCloud): point cloud
            mask (np.ndarray): mask that indicates the points within the ground truth box
        """
        # x, y, z target
        gt_box_xyz = np.array([gt_box.x, gt_box.y, gt_box.z])[np.newaxis,:]
        reg_target[mask,:3] = (gt_box_xyz - points.points[:3,mask].T)/self.avg_anchor_box_np[[0],:3]

        # l, w, h target
        reg_target[mask,3] = np.log(gt_box.l/self.avg_anchor_box.l)
        reg_target[mask,4] = np.log(gt_box.w/self.avg_anchor_box.w)
        reg_target[mask,5] = np.log(gt_box.h/self.avg_anchor_box.h)

        # rotation target, scaled to [-1,1]
        reg_target[mask,6] = np.arctan2(np.sin(gt_box.r), np.cos(gt_box.r))/np.pi

    def _label_points(self, gt_boxes:List[BoundingBox3D], points:LidarPointCloud):
        """Label the points with the ground truth boxes.

        Args:
            gt_boxes (List[BoundingBox3D]): list of ground truth boxes
            points (LidarPointCloud): lidar points
        Returns:
            np.ndarray: 1D array of labels
        """
        n = points.nbr_points()
        labels = -1*np.ones(n)
        loc_regression_target = np.zeros((n,3), dtype=np.float32)
        for gt_box in gt_boxes:
            mask = gt_box.within_gt_box(points.points[:3,:].T)
            # the points within the gt box are labeled with the label of the gt box
            labels[mask] = gt_box.label
            anchor_box : BoundingBox3D = self.anchor_boxes[gt_box.label]
            # compute localization regression target for each point within the gt box
            loc_regression_target[mask,0] = (gt_box.x - points.points[0,mask])/anchor_box.l
            loc_regression_target[mask,1] = (gt_box.y - points.points[1,mask])/anchor_box.w
            loc_regression_target[mask,2] = (gt_box.z - points.points[2,mask])/anchor_box.h

        return labels
    
    def _compute_loc_regression_target(self,gt_boxes:List[BoundingBox3D], points:LidarPointCloud):
        """Compute the location regression target.

        Args:
            gt_boxes (List[BoundingBox3D]): list of ground truth boxes with class labels
            points (LidarPointCloud): lidar points

        Returns:
            np.ndarray: 2D array of location regression targets
        """
        point_labels = self._label_points(gt_boxes, points)
        label_unique = np.unique(point_labels)

        for label in label_unique:
            if label == -1:
                continue
            mask = point_labels == label
            points_in_box = points.points[:3,:].T[mask]
            for gt_box in gt_boxes:
                if gt_box.label == label:
                    loc_reg_target = gt_box.loc_reg_target(points_in_box)
                    break

        n = points.nbr_points()
        loc_reg_targets = np.zeros((n,3), dtype=np.float32)
        for gt_box in gt_boxes:
            mask = gt_box.within_gt_box(points.points[:3,:].T)
            loc_reg_targets[mask] = gt_box.loc_reg_target(points.points[:3,:].T[mask])
        return loc_reg_targets
    

def encode_loc_reg_target(reg_target:np.ndarray,ref_box:BoundingBox3D,
                          gt_box:BoundingBox3D, points:LidarPointCloud, mask:np.ndarray):
    """Encode the regression target for localization.

    Given a point cloud, and a mask that indicates the points within the ground 
    truth box, encode the regression target and stores them in reg_target

    Args:
        reg_target (np.ndarray): 2D array of regression targets
        ref_box (BoundingBox3D): reference box
        gt_box (BoundingBox3D): ground truth box
        points (LidarPointCloud): point cloud
        mask (np.ndarray): mask that indicates the points within the ground truth box
    """
    # x, y, z target
    gt_box_xyz = np.array([gt_box.x, gt_box.y, gt_box.z])[np.newaxis,:]
    reg_target[mask,:3] = (gt_box_xyz - points.points[:3,mask].T)/ref_box.numpy[[0],:3]

    # l, w, h target
    reg_target[mask,3:6] = np.array([np.log(gt_box.l/ref_box.l),
                                     np.log(gt_box.w/ref_box.w),
                                     np.log(gt_box.h/ref_box.h)])[np.newaxis,:]

    # rotation target, scaled to [-1,1]
    reg_target[mask,6] = np.arctan2(np.sin(gt_box.r), np.cos(gt_box.r))/np.pi


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
    point_xyz = torch.from_numpy(points.points[:3,:].T).to(torch.float32)

    # xyz
    bbox[:,:3] = loc_output[:,:3]*ref_box.tensor[[0],:3] + point_xyz

    # lwh
    bbox[:,3:6] = torch.exp(loc_output[:,3:6])*ref_box.tensor[[0],3:6]

    # rotation
    bbox[:,6] = loc_output[:,6]*np.pi

    return bbox
    