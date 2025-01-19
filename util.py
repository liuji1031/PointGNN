from typing_extensions import Self
import itertools
import typing
import numpy as np

class BoundingBox3D:
    """
    class describing a bounding box
    """
    def __init__(self, x, y, z, l, w, h, r) :
        self.x = x
        self.y = y
        self.z = z
        self.l = l
        self.w = w
        self.h = h
        self.r = r

    def find_gt_box_normals(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        given a bounding box description (x, y, z, l, w, h, r), return the
        normals of the box faces, as well as the lower and upper bound of
        projected values of the box corners onto the faces.

        Returns:
            np.ndarray: a 3x3 array with each column representing the normals of
            the box faces 
            np.ndarray: a 3x2 array representing the lower and upper bound of
                        projected values of the box corners onto. each row is 
                        the upper and lower bound of 1 dimension, not sorted
        """
        # compute the normals of the box faces. the columns of the rotation matrix
        # are the normals of the box faces
        R = np.array([[np.cos(self.r), -np.sin(self.r), 0],
                        [np.sin(self.r), np.cos(self.r), 0],
                        [0, 0, 1]])
        
        # get 2 extreme corners
        corners = np.array([[self.l/2, self.w/2, self.h/2],
                            [-self.l/2, -self.w/2, -self.h/2]]).T
        
        corners_ = np.dot(R, corners)
        corners_ += np.array([[self.x, self.y, self.z]]).T

        # compute the projections of the corners
        proj = np.dot(R.T, corners_) # 3x2

        return R, proj

    def within_gt_box(self, points : np.ndarray) -> np.ndarray:
        """check if a point is within the bounding box

        Args:
            points (np.ndarray): a Nx3 array representing the points to be 
            checked
        
        Returns:
            np.ndarray: a 1D boolean array with each element representing 
            whether the corresponding point is within the bounding box
        """
        R, proj_bounds = self.find_gt_box_normals()

        # project the points onto box normal
        proj = np.dot(R.T, points.T) # 3xN

        # compute lower and upper bound of the projected points
        min_val = np.min(proj_bounds, axis=1, keepdims=True) # 3x1
        max_val = np.max(proj_bounds, axis=1, keepdims=True) # 3x1

        # check if the point is within the box, i.e., all 3 boolean values along
        # a column are True
        return np.all((proj>=min_val) & (proj<=max_val), axis=0, keepdims=False)
    
    def iou_2D(self, other : Self) -> float:
        """compute the 2D intersection over union between two bounding boxes

        Args:
            other (Self): _description_

        Returns:
            float: _description_
        """
        pass

