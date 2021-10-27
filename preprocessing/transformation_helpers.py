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

def show_image_pair(img_rgb, img_ir):
    transformed_img_ir = transform_IR_im_to_vis_coordinate_system(img_ir)

    cropped_img_rgb = crop_img(img_rgb)
    cropped_img_ir = crop_img(transformed_img_ir)

    cropped_blended_img = np.maximum(cropped_img_rgb, cropped_img_ir)

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(cropped_img_rgb)
    plt.subplot(2, 3, 4)
    plt.imshow(cropped_img_ir)
    plt.subplot(2, 3, (2,6))
    plt.imshow(cropped_blended_img)
    plt.show()

def read_label_file_lines(filepath: str):
    with open(filepath) as file:
        return list(map(lambda line: line.strip(), file.readlines()))

def write_label_file(filepath: str, labels: list):
    label_lines = list(map(lambda label: label.toLabelLine(), labels))
    label_file_text = '\n'.join(label_lines)
    with open(filepath, 'w') as file:
        file.write(label_file_text)



# plt.imshow(cropped)

#scaleX = wIR/ cropped.shape[1]
#print(scaleX)
#scaleY = hIR/ cropped.shape[0]
#print(scaleY)
#cropped = cv2.resize(cropped, (wIR, hIR))
#cropped = distort_im(cropped)
#cropped = cropped[33:420, 57:566]
#cropped =  cv2.remap(cropped, mapx2, mapy2, cv2.INTER_LINEAR)

#plt.figure()
#plt.imshow(cropped)

#min_corner = [ 396, 2752]
#max_corner = [3748,  176]
#print(cropped.shape)



#corners = select_coordinates_from_image(cropped*255)
#print(corners)

#def distort_point(p):
#
#    cx = K[0,2]
#    cy = K[1,2]
#    fx = K[0,0]
#    fy = K[1,1]
#    k1 = dist[0][0] *-1
#    k2 = dist[0][1] * -1
#    k3 = dist[0][-1] *-1
#    p1 = dist[0][2] * -1
#    p2 = dist[0][3] *-1
#    
#    x = ( p[0]- cx) / fx
#    y = (p[1]- cy) / fy
#
#    
#    r2 = x*x + y*y
#        
#    xDistort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
#    yDistort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
#    
#    xDistort = xDistort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x))
#    yDistort = yDistort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
#    
#    xDistort = xDistort * fx + cx
#    yDistort = yDistort * fy + cy
#    
#    
#    return[xDistort, yDistort]
#
#
#def normalise_p(p):
#    cx = K[0,2]
#    cy = K[1,2]
#    fx = K[0,0]
#    fy = K[1,1]
#    
#    x = ( float(p[0])- float(cx)) / float(fx)
#    y = ( float(p[1])- float(cy)) / float(fy)
#    
#    
#    return np.array([x,y, 1], np.float32)
#
#
#
# 
#
#def transform_vis_points_to_IR(pts):
#    print(T_v2IR)
#
#    T = T_v2IR
#    #scale x:
#    T[0,0] = T[0,0]
#    #scale y:
#    T[1,1] = T[1,1]
#    #trans x:
#    T[0,2] = T[0,2]
#    #trans y:
#    T[1,2] = T[1,2]
#
#    pts = transform.AffineTransform( T )(pts)
#    pts =np.asarray(pts, np.float32)
#    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist*-1,(wIR, hIR),1,(wIR, hIR))
#    undistorted = cv2.undistortPoints(pts.reshape((pts.shape[0],1,2)), K, dist, P=newcameramtx)
#    return undistorted.reshape(undistorted.shape[0], undistorted.shape[2])
#

