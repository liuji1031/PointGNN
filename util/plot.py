import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from nuscenes.utils.data_classes import LidarPointCloud

def to_lidar_point_cloud(data:Data):
    return LidarPointCloud(np.vstack([data.pos.T.numpy(),data.x.T.numpy()]))

def plot_pc(pc:LidarPointCloud, rng=(-45,45)):
    pc.render_height(ax=plt.gca(),x_lim=rng,y_lim=rng,marker_size=0.1)

def plot_torch_geo_data(data:Data,fig=None, rng=(-45,45),fig_size=(10,10)):
    if fig is None:
        fig = plt.figure(figsize=fig_size)
    else:
        plt.figure(fig.number)
    pc = to_lidar_point_cloud(data)
    plot_pc(pc,rng=rng)
    for gt_box in data.gt_boxes:
        gt_box.render(axis=plt.gca(), linewidth=1)    
    plt.show()