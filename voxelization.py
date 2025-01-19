import numpy as np
import numba
import torch
from einops import rearrange, pack, reduce

class VoxelDownsample():
    """Reduce the number of points by voxelization, i.e., returns the average of
    points in each voxel.

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
                 x_range,
                 y_range,
                 z_range,
                 voxel_size,
                 max_voxel_num=3000,
                 init_decoration=True,
                 ):
        super().__init__()

        self.name = "VoxelDownsample"

        # if True, append diff to vox center
        self.init_decoration = init_decoration

        dx,dy,dz = voxel_size["x"], voxel_size["y"], voxel_size["z"]

        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range

        nx = int(np.round((x_max-x_min)/dx))
        ny = int(np.round((y_max-y_min)/dy))
        nz = int(np.round((z_max-z_min)/dz))

        self.spatial_shape_WHD = [nx,ny,nz]
        self.spatial_shape_DHW = self.spatial_shape_WHD[::-1]

        # make the range integer copy of the interval
        x_max = x_min + nx*dx
        y_max = y_min + ny*dy
        z_max = z_min + nz*dz

        self.x_range = [x_min, x_max]
        self.y_range = [y_min, y_max]
        self.z_range = [z_min, z_max]
        self.voxel_size_DHW = [dz, dy, dx]
        # self.max_voxel_pts = max_voxel_pts

        # expand axis
        self.lb_DHW = rearrange(np.array([z_min,y_min,x_min]),
                                "d->1 d")
        self.vox_sz_DHW = rearrange(np.array(self.voxel_size_DHW),
                                    "d->1 d")
        
        self.max_voxel_num = max_voxel_num

    def process_single_batch_pc(self, pc:np.ndarray, feature=None):
        # exclude out of range points
        x,y,z = pc[:,0],pc[:,1],pc[:,2]
        n_feat = pc.shape[-1] # feature dim

        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        z_min, z_max = self.z_range

        keep = (x>=x_min) & (x<x_max) & (y>=y_min) & (y<y_max) & \
        (z>=z_min) & (z<z_max)
        keep = np.argwhere(keep).squeeze()

        pc = pc[keep,:]
        # rearrange to DHW
        # pc[:,:3] = pc[:,[2,1,0]]
        
        coord = ((pc[:,:3] - self.lb_DHW)/self.vox_sz_DHW).astype(np.int32)

        _, inverse_ind, counts = np.unique(coord,
                                              return_inverse=True,
                                              return_counts=True,
                                              axis=0)
        
        isort = np.argsort(inverse_ind)
        pc_sort = pc[isort,:]
        ind_slice = np.concatenate([np.array([0]),np.cumsum(counts)])[:-1]
        pc_mean = np.add.reduceat(pc_sort, ind_slice, axis=0) / \
            counts[:, np.newaxis]
        
        return pc_mean

    def __call__(self, point_cloud : np.ndarray):
        """parse point cloud into voxels

        Args:
            point_cloud (_type_): a single point cloud, a 2D numpy array with
            shape (npoints, nfeatures) 
        """

        return self.process_single_batch_pc(point_cloud)