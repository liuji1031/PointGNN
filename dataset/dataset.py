import multiprocessing as mp
import os
import pathlib
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import Compose
from tqdm import tqdm

from bbox import BoundingBox3D
from dataset.augment import AugmentRegistry
from dataset.preprocess import PreprocessPointCloud, PreprocessRegistry
from dataset.util import encode_loc_reg_target


class NuScenesDataset(Dataset):
    def __init__(
        self,
        data_root,
        version,
        mode,
        num_class,
        x_range=(-40, 40),
        y_range=(-40, 40),
        z_range=(-1, 8),
        train_val_test_split=(0.8, 0.1, 0.1),
        seed=0,
        preprocess=None,
        augmentation=None,
        anchor_box=None,
        verbose=False,
        **kwargs,
    ):
        # reminder to self, processed_dir = root/processed, raw_dir = root/raw
        self.data_root = data_root
        self.version = version
        self.verbose = verbose
        self.mode = mode
        self.num_class = num_class
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.nusc = NuScenes(version=version, dataroot=data_root, verbose=verbose)
        self.scenes = self.nusc.scene
        self.sample_tokens_by_scene = self._get_sample_tokens_by_scene()
        self.train_tokens, self.val_tokens, self.test_tokens = (
            self._split_train_val_test(train_val_test_split, seed)
        )

        if self.mode == "train":
            self.sample_tokens = self.train_tokens
        elif self.mode == "val":
            self.sample_tokens = self.val_tokens
        else:
            self.sample_tokens = self.test_tokens

        self.anchor_boxes, self.category_list = self._gen_anchor_boxes(anchor_box)
        # build a dictionary for category and sub-category. example output:
        # {"vehicle": ["car", "truck", "bus"]}
        self.category_dict = self._parse_category()

        # store anchor box information into a torch array
        if len(self.anchor_boxes):
            self.anchor_boxes_tensor = []
            for box in self.anchor_boxes:
                box: BoundingBox3D
                self.anchor_boxes_tensor.append(box.to_tensor())
            self.anchor_boxes_tensor = torch.stack(
                self.anchor_boxes_tensor, dim=0
            )  # n_anchor_box x 7
            self.anchor_boxes_tensor.requires_grad = False

        # get an average size of the anchor boxes
        self.avg_anchor_box = self._get_average_anchor_box()

        # gather preprocessings from the registry
        if preprocess is not None:
            self.preprocesses = self._parse_preprocess_config(preprocess)
        else:
            self.preprocesses = []

        # gather augmentations from the registry
        if augmentation is not None:
            self.augmentations = Compose(self._parse_augment_config(augmentation))
        else:
            self.augmentations = None

        # custom dataset class for nuScenes dataset, init super with None
        super().__init__(
            root=data_root, transform=None, pre_transform=None, pre_filter=None
        )

    @property
    def processed_dir(self):
        """Return the processed directory."""
        return str(pathlib.Path(self.data_root) / f"processed_{self.mode}")

    @property
    def num_classes(self):
        """Return the number of classes in the dataset."""
        return self.num_class

    def len(self):
        """Return the number of samples in the dataset under mode train, val or test."""
        return len(self.sample_tokens)

    @property
    def raw_file_names(self):
        """Return the raw file names.

        Implemented to skip download.
        Returns:
            list: empty list
        """
        return []

    @property
    def processed_file_names(self):
        """Return the processed file names.

        These files must be present in the processed directory in order to skip
        the processing of the raw data.

        Returns:
            list: empty list
        """
        # figure out the number of digits in the length of the dataset
        n_digits = len(str(self.len()))
        return [f"data_{i + 1:0{n_digits}d}.pt" for i in range(self.len())]

    def download(self):
        """Download the dataset.

        Not needed for nuScenes dataset.
        """
        pass

    def process(self):
        """Process raw data from NuScenes dataset and save to disk."""
        processed_dir = pathlib.Path(self.processed_dir)

        def _process(sample_token, fn):
            sample = self.nusc.get("sample", sample_token)
            gt_boxes = self._get_gt_boxes_from_sample(sample=sample)
            points, sensor_loc = self._get_lidar_pts_singlesweep(sample=sample)

            # apply preprocessings
            for pre in self.preprocesses:
                pre: PreprocessPointCloud
                pre.preprocess(points)

            # create torch geometric data from points
            data = Data(
                x=torch.from_numpy(points.points[3:,].T).float(),  # features
                pos=torch.from_numpy(points.points[:3, :].T).float(),  # position
                gt_boxes=gt_boxes,  # custom keyword attribute
                sensor_loc=torch.from_numpy(sensor_loc).float(),
            )

            # save to disk
            torch.save(data, processed_dir / fn)

        for i, (sample_token, fn) in enumerate(
            tqdm(zip(self.sample_tokens, self.processed_file_names), total=self.len())
        ):
            _process(sample_token, fn)

    def get(self, idx):
        """Load a torch geometric data object from disk.

        Args:
            idx (int): index of the sample

        Returns:
            Data: torch geometric data object
        """
        data = torch.load(
            pathlib.Path(self.processed_dir) / self.processed_file_names[idx],
        )

        # do augmentations on the data
        if self.augmentations is not None:
            data = self.augmentations(data)

        # compute localization regression target
        reg_xyz,reg_lwh,reg_r,positive_mask,obj_cls_label = self._compute_loc_regression_target(
            data.gt_boxes, data.pos
        )
        data.reg_xyz = reg_xyz
        data.reg_lwh = reg_lwh
        data.reg_r = reg_r
        data.positive_mask = positive_mask
        data.obj_cls_label = obj_cls_label
        return data

    def _parse_augment_config(self, augment_config: list):
        """Parse the augmentation configuration.

        Args:
            augment_config (dict): augmentation configuration

        Returns:
            dict: dictionary of augmentation objects
        """
        augmentations = []
        for aug_cfg in augment_config:
            aug_name = aug_cfg["name"]
            if aug_name not in AugmentRegistry.REGISTRY:
                raise ValueError(f"Augmentation {aug_name} not found in registry")
            aug_method = AugmentRegistry.REGISTRY[aug_name]
            augmentations.append(aug_method(**aug_cfg["kwargs"]))
        return augmentations

    def _parse_preprocess_config(self, preprocess_config: list):
        """Parse the preprocessing configuration.

        Args:
            preprocess_config (dict): preprocessing configuration

        Returns:
            dict: dictionary of preprocessing objects
        """
        preprocesses = []
        for pre_cfg in preprocess_config:
            pre_name = pre_cfg["name"]
            if pre_name not in PreprocessRegistry.REGISTRY:
                raise ValueError(f"Preprocessing {pre_name} not found in registry")
            pre_method = PreprocessRegistry.REGISTRY[pre_name]
            preprocesses.append(pre_method(**pre_cfg["kwargs"]))
        return preprocesses

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
            cat, sub_cat = c.split(".")[:2]
            if cat not in category_dict:
                category_dict[cat] = []
            category_dict[cat].append(sub_cat)
        return category_dict

    def _in_category(self, category_name: str):
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
                return True, cat + "." + sub_cat
            else:
                return False, ""

    def _gen_anchor_boxes(self, anchor_box_config: Union[dict, None]):
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
            w, l, h = config["wlh"]
            bbox = BoundingBox3D(0.0, 0.0, 0.0, w, l, h, 0.0)
            bbox.name = config["class"]
            anchor_boxes.append(bbox)
            category_list.append(config["class"])
        return anchor_boxes, category_list

    def _get_average_anchor_box(self):
        """Get the average of anchor boxes."""
        avg_box = BoundingBox3D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for box in self.anchor_boxes:
            avg_box.l += box.l
            avg_box.w += box.w
            avg_box.h += box.h
        avg_box.l /= len(self.anchor_boxes)
        avg_box.w /= len(self.anchor_boxes)
        avg_box.h /= len(self.anchor_boxes)
        avg_box.update()
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
                sample_token = self.nusc.get("sample", sample_token)["next"]
            sample_tokens_by_scene.append(tokens)
        return sample_tokens_by_scene

    def _split_train_val_test(
        self, train_val_test_split: Tuple[float] = (0.8, 0.1, 0.1), seed: int = 0
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
            split_ind = (np.cumsum(train_val_test_split) * n_samples).astype(int)
            np.random.seed(seed)
            ind = np.random.permutation(n_samples)
            ind_train = ind[: split_ind[0]]
            ind_val = ind[split_ind[0] : split_ind[1]]
            ind_test = ind[split_ind[1] :]

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
        return self.nusc.get("sample_annotation", ann_token)["category_name"]

    def _get_gt_box_label(self, gt_box: Box):
        """Get the label of the ground truth box.

        Args:
            gt_box (Box): ground truth box

        Returns:
            int: label of the ground truth box
        """
        ...

    def _get_gt_boxes_from_sample(
        self, sample_token: str = "", sample: Union[dict, None] = None, render=False
    ) -> List[BoundingBox3D]:
        """Get the ground truth bounding boxes in a sample.

        Args:
            sample_token (str): sample token

        Returns:
            list: list of bounding box dictionaries
        """
        sample = sample if sample is not None else self.nusc.get("sample", sample_token)

        annotations = sample["anns"]
        _, boxes, _ = self.nusc.get_sample_data(
            sample["data"]["LIDAR_TOP"], use_flat_vehicle_coordinates=True
        )

        gt_boxes = []
        for k, ann_token in enumerate(annotations):
            category_name = self._get_category_name(ann_token)

            # keep only category and sub-category name in "category_name"
            in_category, category_name = self._in_category(category_name)
            if in_category:
                gt_box = BoundingBox3D.from_nuscene_box(boxes[k])

                if (
                    gt_box.x >= self.x_range[0]
                    and gt_box.x <= self.x_range[1]
                    and gt_box.y >= self.y_range[0]
                    and gt_box.y <= self.y_range[1]
                ):
                    pass
                else:
                    continue

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

        if len(gt_boxes) > 0:
            # sort the boxes by distance to the origin
            gt_boxes = sorted(gt_boxes, key=lambda b: b.x**2 + b.y**2)

        return gt_boxes

    def _get_raw_lidar_pts(self, sample):
        """retrieves raw lidar points from filename stored in sample data

        Args:
            sample (dict): NuScenes sample dictionary

        Returns:
            LidarPointCloud: raw lidar points
        """
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_info = self.nusc.get("sample_data", lidar_token)

        lidar_path = os.path.join(self.nusc.dataroot, lidar_info["filename"])
        lidar_data = LidarPointCloud.from_file(str(lidar_path))
        return lidar_data

    def _transform_matrix_from_pose(self, pose: dict, inverse: bool):
        """wrapper function for getting the transformation matrix from pose dict

        Args:
            pose (dict): NuScenes pose dictionary
            inverse (bool): whether to get the inverse transformation matrix

        Returns:
            <np.float32: 4, 4>: 4 by 4 transformation matrix
        """
        return transform_matrix(
            pose["translation"], Quaternion(pose["rotation"]), inverse=inverse
        )

    def _get_lidar_pts_singlesweep(
        self,
        sample_token: str = "",
        sample: Union[dict, None] = None,
        sensor="LIDAR_TOP",
        convert_to_ego_frame=True,
        min_dist=1.0,
    ):
        """Returns the lidar points for a single sweep

        Args:
            sample_token (str, optional): sample token. Defaults to "".
            sample (Union[dict, None], optional): sample dictionary. Defaults to None.
            sensor (str, optional): sensor name. Defaults to "LIDAR_TOP".
            convert_to_ego_frame (bool, optional): whether to convert the points to ego frame.
            Defaults to True.
            min_dist (float, optional): minimum distance to remove points. Defaults to 1.0.
        Returns:
            LidarPointCloud: lidar points
            np.ndarray: sensor location relative to the car
        """
        sample = sample if sample is not None else self.nusc.get("sample", sample_token)
        sample_data = self.nusc.get("sample_data", sample["data"][sensor])

        pc = self._get_raw_lidar_pts(sample)
        pc.remove_close(min_dist)

        if not convert_to_ego_frame:
            return pc

        # get sensor ref car transformation
        pose = self.nusc.get(
            "calibrated_sensor", sample_data["calibrated_sensor_token"]
        )
        sensor_ref_car = self._transform_matrix_from_pose(pose, inverse=False)
        sensor_loc = np.array(pose["translation"])[np.newaxis, :]

        # convert sensor points to car body frame
        pc.transform(sensor_ref_car)

        return pc, sensor_loc

    def _compute_loc_regression_target(
        self, gt_boxes: List[BoundingBox3D], pos: torch.Tensor
    ):
        """Compute the regression target for localization head.

        Args:
            gt_boxes (List[BoundingBox3D]): list of ground truth boxes
            points (LidarPointCloud): lidar points
        Returns:
            np.ndarray: 1D array of labels
        """
        n = pos.shape[0]
        positive_mask = torch.from_numpy(np.zeros(n, dtype=bool))
        box_xyz = torch.from_numpy(np.zeros((n, 3), dtype=np.float32))
        box_lwh = torch.from_numpy(np.zeros((n, 3), dtype=np.float32))
        box_r = torch.from_numpy(np.zeros((n, 1), dtype=np.float32))
        positive_mask = torch.from_numpy(np.zeros(n, dtype=bool))
        obj_cls_label = torch.from_numpy(np.zeros(n, dtype=np.int64))
        for gt_box in gt_boxes:
            mask = gt_box.within_gt_box(pos.numpy())
            box_xyz[mask, :] = gt_box.xyz_tensor.to(torch.float32)
            box_lwh[mask, :] = gt_box.lwh_tensor.to(torch.float32)
            box_r[mask, :] = gt_box.r_tensor.to(torch.float32)
            positive_mask[mask] = True
            obj_cls_label[mask] = gt_box.label

        # compute localization regression target for each point within the gt box
        reg_xyz, reg_lwh, reg_r = encode_loc_reg_target(
            box_xyz=box_xyz,
            box_lwh=box_lwh,
            box_r=box_r,
            ref_box_lwh=self.avg_anchor_box.lwh_tensor,
            pos=pos,
        )

        return reg_xyz, reg_lwh, reg_r, positive_mask, obj_cls_label
