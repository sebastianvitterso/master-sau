from typing import List
import cv2
from helpers import CORNER_TOP_LEFT, CORNER_BOTTOM_RIGHT, RAW_SIZE_IR, RAW_SIZE_RGB, GET_PARTITION_TOP_LEFT_CORNER
import numpy as np
from skimage import transform

CAMERA_MATRIX_K = np.load("./parameters/camera_matrix_K.npy")
CAMERA_DIST_COEFFS = np.load("./parameters/camera_dist_coeffs.npy")
TRANSFORM_VIS_TO_IR = np.load("./parameters/Transform_vis_to_IR.npy")
TRANSFORM_IR_TO_VIS = np.load("./parameters/Transform_IR_to_Vis.npy")

class Image:
    def __init__(self, img:np.ndarray, is_distorted:bool=False, is_cropped:bool=False, partition_coordinates:tuple[int, int]=None):
        self.img = img
        self.is_distorted = is_distorted
        self.is_cropped = is_cropped
        self.partition_coordinates = partition_coordinates
        
    def __str__(self):
        return f"<Image img.shape={self.img.shape}, is_distorted={self.is_distorted}, is_cropped={self.is_cropped}, partition_coordinates={self.partition_coordinates}>"
    def __repr__(self):
        return self.__str__()

    @classmethod
    def loadFromImagePath(cls, image_path:str, is_distorted:bool=False, is_cropped:bool=False, partition_coordinates:tuple[int, int]=None):
        img = cv2.imread(image_path)
        return cls(img, is_distorted, is_cropped, partition_coordinates)

    def saveToImagePath(self, image_path:str):
        cv2.imwrite(image_path, self.img)

    def undistort(self) -> 'Image':
        ''' @source: https://github.com/thaiKari/sheepDetector/blob/master/transformations.py '''

        assert self.is_distorted, "Don't undistort an already undistorted image!"

        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX_K, CAMERA_DIST_COEFFS, RAW_SIZE_IR, 1, RAW_SIZE_IR)    
        img = cv2.undistort(self.img, CAMERA_MATRIX_K, CAMERA_DIST_COEFFS, None, newcameramtx)
        T = transform.AffineTransform(TRANSFORM_VIS_TO_IR)
        img = (transform.warp(img, T, output_shape=(RAW_SIZE_RGB[1], RAW_SIZE_RGB[0])) * 255).astype(int)
        return Image(img, is_distorted=False, is_cropped=self.is_cropped, partition_coordinates=self.partition_coordinates)

    def crop(self) -> 'Image':
        assert not self.is_cropped, "Don't crop an already cropped image!"
        img = self.img[ 
            CORNER_TOP_LEFT[1]:CORNER_BOTTOM_RIGHT[1],
            CORNER_TOP_LEFT[0]:CORNER_BOTTOM_RIGHT[0],
        ]
        return Image(img, is_distorted=self.is_distorted, is_cropped=True, partition_coordinates=self.partition_coordinates)


    def partitions(self) -> List['Image']:
        assert self.partition_coordinates is None, "Don't partition a partition!"

        x_partition_count = 3 if self.is_cropped else 4
        y_partition_count = 2 if self.is_cropped else 3

        partitions = []
        for y in range(y_partition_count):
            for x in range(x_partition_count):
                partition_top_left = GET_PARTITION_TOP_LEFT_CORNER(x, y, is_cropped=self.is_cropped)
                partition_img = self.img[
                    partition_top_left[1]:partition_top_left[1] + 1280,
                    partition_top_left[0]:partition_top_left[0] + 1280,
                ]
                partition_image = Image(partition_img, is_distorted=self.is_distorted, is_cropped=self.is_cropped, partition_coordinates=(x,y))
                partitions.append(partition_image)

        return partitions
    
    
