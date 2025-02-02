# Description: This file contains the abstract class for preprocessing point cloud data.
from abc import ABC, abstractmethod
from ground_removal import ground_removal
from util import camel_to_snake
from nuscenes.utils.data_classes import LidarPointCloud


class PreprocessRegistry():
    REGISTRY = {}
    
    def __init_subclass__(cls, **kwargs):
        PreprocessRegistry.REGISTRY[camel_to_snake(cls.__name__)] = cls
        super().__init_subclass__(**kwargs)

class PreprocessPointCloud(ABC):
    @abstractmethod
    def preprocess(self, pc:LidarPointCloud, **kwargs):
        pass

class GroundRemovalRANSAC(PreprocessPointCloud, PreprocessRegistry):
    def __init__(self,
                 inlier_threshold=0.2,
                 sample_size=10,
                 max_iterations=100,
                 random_seed=0,
                 restrict_range=20,
                 inplace=True):
        self.inlier_threshold = inlier_threshold
        self.sample_size = sample_size
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.restrict_range = restrict_range
        self.inplace = inplace

    def preprocess(self, pc:LidarPointCloud, **kwargs):
        return ground_removal(
            pc,
            self.inlier_threshold,
            self.sample_size,
            self.max_iterations,
            self.random_seed,
            self.restrict_range,
            self.inplace
        )

if __name__ == "__main__":
    print(PreprocessRegistry.REGISTRY)