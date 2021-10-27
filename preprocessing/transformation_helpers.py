"""
@source: https://github.com/thaiKari/sheepDetector/blob/master/transformations.py
"""

import numpy as np
import cv2
from skimage import transform
import matplotlib.pyplot as plt

K = np.load("./parameters/camera_matrix_K.npy")
dist = np.load("./parameters/camera_dist_coeffs.npy")
T_v2IR = np.load("./parameters/Transform_vis_to_IR.npy")
T_IR2v = np.load("./parameters/Transform_IR_to_Vis.npy")
wv, hv = (4056, 3040)
wIR,hIR = (640, 480)

# coordinates are (x, y), measured in pixels, respectively from left and top.
# coordinates were retrieved simply by measuring in photopea.com
CORNER_TOP_LEFT = (480, 285)
CORNER_BOTTOM_RIGHT = (3680, 2608)
RAW_SIZE = (4056, 3040)
CROPPED_SIZE = (
    CORNER_BOTTOM_RIGHT[0] - CORNER_TOP_LEFT[0],
    CORNER_BOTTOM_RIGHT[1] - CORNER_TOP_LEFT[1],
)

def undistort_IR_im(im):
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(wIR, hIR),1,(wIR, hIR))    
    return cv2.undistort(im, K, dist, None, newcameramtx)

def transform_IR_im_to_vis_coordinate_system(im):
    im = undistort_IR_im(im)
    T = transform.AffineTransform(T_v2IR)
    return (transform.warp(im, T, output_shape=(hv,wv)) * 255).astype(int)

def crop_img(img):
    # img should be a cv2 image, which means a np.array(3040, 4056, 3)
    # note that cv2 reads and writes images as numpyarrays with (y, x) not (x, y).
    # that's why [1] is before [0] here.
    return img[ 
        CORNER_TOP_LEFT[1]:CORNER_BOTTOM_RIGHT[1],
        CORNER_TOP_LEFT[0]:CORNER_BOTTOM_RIGHT[0],
    ]

def show_image_pair(img_rgb, img_ir, labels=None):

    blended_img = np.maximum(img_rgb, img_ir)

    if labels is not None:
        for label in labels:
            blended_img = cv2.rectangle(blended_img, (label.left, label.top), (label.right, label.bottom), (0,0,255), 2)
    
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.subplot(2, 3, 4)
    plt.imshow(img_ir)
    plt.subplot(2, 3, (2,6))
    plt.imshow(blended_img)
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()

def read_label_file_lines(filepath: str):
    with open(filepath) as file:
        return list(map(lambda line: line.strip(), file.readlines()))

def write_label_file(filepath: str, labels: list):
    label_lines = list(map(lambda label: label.toLabelLine(), labels))
    label_file_text = '/n'.join(label_lines)
    with open(filepath, 'w') as file:
        file.write(label_file_text)

    
