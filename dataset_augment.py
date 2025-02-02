from abc import ABC, abstractmethod
from typing import List
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from bbox import BoundingBox3D
from util import camel_to_snake

class AugmentRegistry():
    REGISTRY = {}
    def __init_subclass__(cls, **kwargs):
        AugmentRegistry.REGISTRY[camel_to_snake(cls.__name__)] = cls
        super().__init_subclass__(**kwargs)

class AugmentPointCloud(ABC):
    @abstractmethod
    def augment(self, points:LidarPointCloud,gt_boxes:List[BoundingBox3D],**kwargs):
        pass
    
class RandomJitter(AugmentPointCloud, AugmentRegistry):
    def __init__(self, sigma:float=0.1):
        self.sigma = sigma
    
    def augment(self, points : LidarPointCloud,gt_boxes:List[BoundingBox3D],**kwargs):
        assert 'sensor_loc' in kwargs
        sensor_loc = kwargs['sensor_loc']
        sigma = kwargs.get('sigma', self.sigma)
        self._random_jitter(points, sensor_loc, sigma)
        
    def _random_jitter(self, points:LidarPointCloud, sensor_loc:np.ndarray, sigma:float=0.01):
        """Randomly jitter the points in the point cloud. 

        Args:
            points (LidarPointCloud): point cloud expressd in the ego (car) frame
            sensor_loc (np.ndarray): sensor location in the ego frame
        """
        assert sensor_loc.ndim==2
        # jitter along the reflection direction
        tmp = points.points[:3,:].T-sensor_loc
        d = np.linalg.norm(tmp,axis=1,keepdims=True) # n x 1
        u = tmp/d # unit vector, n x 3
        jitter = np.random.randn(*d.shape)*sigma # n x 1
        points.points[:3,:] += (u*jitter).T # add jitter along the reflection direction

class RandomRotation(AugmentPointCloud, AugmentRegistry):
    def __init__(self, sigma:float=np.pi/8):
        self.sigma = sigma
    
    def augment(self, points : LidarPointCloud,gt_boxes:List[BoundingBox3D], **kwargs):
        self._random_rotation(points,gt_boxes, self.sigma)

    def _random_rotation(self, points:LidarPointCloud,gt_boxes:List[BoundingBox3D], sigma:float=np.pi/8):
        """Randomly rotate the points in the point cloud. 

        Args:
            points (LidarPointCloud): point cloud expressd in the ego (car) frame
        """
        # rotation matrix
        r = np.random.randn()*sigma
        R = np.array([[np.cos(r), -np.sin(r), 0],
                    [np.sin(r), np.cos(r), 0],
                    [0, 0, 1]])
        points.points[:3,:] = np.dot(R, points.points[:3,:])

        # rotate gt boxes
        q = Quaternion(axis=[0,0,1],angle=r)
        for box in gt_boxes:
            box.rotate(q)


class RandomFlipY(AugmentPointCloud, AugmentRegistry):
    """Randomly flip the points in the point cloud along the y-axis."""
    def __init__(self, prob:float=0.5):
        self.prob = prob
    
    def augment(self, points : LidarPointCloud,gt_boxes:List[BoundingBox3D], **kwargs):
        self._random_flip_y(points,gt_boxes, self.prob)

    def _random_flip_y(self, points:LidarPointCloud,gt_boxes:List[BoundingBox3D], prob:float=0.5):
        """Randomly flip the points in the point cloud along the y-axis. 

        Args:
            points (LidarPointCloud): point cloud expressd in the ego (car) frame
        """
        if np.random.rand()<prob:
            points.points[1,:] *= -1 # flip along y-axis
        
            # flip gt boxes
            for box in gt_boxes:
                box.flip_y()

if __name__ == "__main__":
    print(AugmentRegistry.REGISTRY)
    pc_test = LidarPointCloud(np.random.randn(4,100))
    sensor_loc_test = np.random.randn(3)[np.newaxis,:]

    for aug_name, aug_cls in AugmentRegistry.REGISTRY.items():
        aug = aug_cls()
        pc_aug = aug.augment(pc_test, sensor_loc=sensor_loc_test)
