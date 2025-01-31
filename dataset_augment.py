from abc import ABC, abstractmethod
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud

class AugmentPointCloud(ABC):

    @abstractmethod
    def augment(self, pc:LidarPointCloud, *args, **kwargs):
        pass

class RandomJitter(AugmentPointCloud):
    def __init__(self, sigma:float=0.1):
        self.sigma = sigma
    
    def augment(self, pc : LidarPointCloud, *args, **kwargs):
        sensor_loc = kwargs['sensor_loc']
        return self._random_jitter(pc, sensor_loc, self.sigma)
        
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

class RandomRotation(AugmentPointCloud):
    def __init__(self, sigma:float=np.pi/8):
        self.sigma = sigma
    
    def augment(self, pc : LidarPointCloud, *args, **kwargs):
        return self._random_rotation(pc, self.sigma)

    def _random_rotation(self, points:LidarPointCloud, sigma:float=np.pi/8):
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

class RandomFlipX(AugmentPointCloud):
    def __init__(self, prob:float=0.5):
        self.prob = prob
    
    def augment(self, pc : LidarPointCloud, *args, **kwargs):
        return self._random_flip_x(pc, self.prob)

    def _random_flip_x(self, points:LidarPointCloud, prob:float=0.5):
        """Randomly flip the points in the point cloud along the x-axis. 

        Args:
            points (LidarPointCloud): point cloud expressd in the ego (car) frame
        """
        if np.random.rand()<prob:
            points.points[0,:] = -points.points[0,:] # flip along x-axis