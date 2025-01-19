import torch.utils.data as data
from nuscenes.nuscenes import NuScenes

class NuScenesDataset(data.Dataset):
    def __init__(self, data_root, version='v1.0-trainval', verbose=False):
        self.data_root = data_root
        self.version = version
        self.verbose = verbose
        self.nusc = NuScenes(version=version, dataroot=data_root, verbose=verbose)
        self.scenes = self.nusc.scene

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        ...