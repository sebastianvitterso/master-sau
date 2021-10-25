"""
@source: https://github.com/thaiKari/sheepDetector/blob/master/transformations.py
"""

import numpy as np
import cv2
from skimage import transform
import matplotlib.pyplot as plt
# from utils import resize_by_scale, get_line_mask, read_pts, select_coordinates_from_image

K = np.load("./parameters/camera_matrix_K.npy")
dist = np.load("./parameters/camera_dist_coeffs.npy")
T_v2IR = np.load("./parameters/Transform_vis_to_IR.npy")
T_IR2v = np.load("./parameters/Transform_IR_to_Vis.npy")
wv, hv = (4056, 3040)
wIR,hIR = (640, 480)

def undistort_IR_im(im):
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(wIR, hIR),1,(wIR, hIR))    
    return cv2.undistort(im, K, dist, None, newcameramtx)

def transform_IR_im_to_vis_coordinate_system(im):
    im = undistort_IR_im(im)
    T = transform.AffineTransform(T_v2IR)
    return transform.warp(im, T, output_shape=(hv,wv))

def undistort_IR_pt_list(pts):
    pts =np.asarray(pts, np.float32)
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(wIR, hIR),1,(wIR, hIR))
    undistorted = cv2.undistortPoints(pts.reshape((pts.shape[0],1,2)), K, dist, P=newcameramtx)
    return undistorted.reshape(undistorted.shape[0], undistorted.shape[2])

def transform_IR_pt_list_to_vis_coordinate_system(pts):
    pts = undistort_IR_pt_list(pts)
    return transform.AffineTransform(T_IR2v)(pts)

def transform_vis_pt_list_to_IR_coordinate_system(pts):
    pts = transform.AffineTransform(T_v2IR)(pts)
    for pt in pts:
        if pt[0] < 0:
            pt[0] = 0
        if pt[0] > wIR:
            pt[0] = wIR-1
        if pt[1]< 0:
            pt[1] = 0
        if pt[1] > hIR:
            pt[1] = hIR -1
        
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(wIR, hIR),1,(wIR, hIR))
    mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(wIR,hIR),5) 
    pts = list(map( lambda p: [mapx[ int(p[1]), int(p[0])], mapy[int(p[1]), int(p[0])] ] ,pts))
    return np.asarray(pts)

def transform_vis_im_to_IR_coordinate_system(im):
    T = transform.AffineTransform(T_IR2v)
    im = transform.warp(im, T, output_shape=(hIR,wIR))
    imNew = np.zeros_like(im)
    
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(wIR, hIR),1,(wIR, hIR))
    mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(wIR,hIR),5) 
    SCALE = 1.15
    mapx = cv2.resize(mapx,None,fx=SCALE, fy=SCALE, interpolation = cv2.INTER_LINEAR)
    mapy = cv2.resize(mapy,None,fx=SCALE, fy=SCALE, interpolation = cv2.INTER_LINEAR)
    
    
    
    for x in range(mapx.shape[1]):
        for y in range(mapx.shape[0]):
            try:
                x2 = int(mapx[y,x])
                y2 = int(mapy[y,x])
                imNew[y2,x2,:] = im[int(y/SCALE), int(x/SCALE),:]
            except:
                no_match=True#just do nothing
            
    return imNew

# coordinates are (x, y), measured in pixels, respectively from left and top.
# coordinates were retrieved simply by measuring in photopea.com
corner_top_left = (480, 285)
corner_bottom_right = (3680, 2608)
cropped_size = (
    corner_bottom_right[0] - corner_top_left[0],
    corner_bottom_right[1] - corner_top_left[1],
)

def crop_img(img):
    # img should be np.array(3040, 4056, 3)
    # note that cv2 reads and writes images as numpyarrays with (y, x) not (x, y).
    # that's why [1] is before [0] here.
    return img[ 
        corner_top_left[1]:corner_bottom_right[1],
        corner_top_left[0]:corner_bottom_right[0],
    ]

img_ir = cv2.imread('../data/ir/2021_09_holtan_0487.JPG')
img_rgb = cv2.imread('../data/rgb/2021_09_holtan_0487.JPG')
plt.figure()

# plt.imshow(img_rgb)


transformed_img_ir = transform_IR_im_to_vis_coordinate_system(img_ir)



print(cropped_size)
cropped_img_ir = crop_img(transformed_img_ir)
cropped_img_rgb = crop_img(img_rgb)
plt.imshow(cropped_img_rgb)
plt.show()

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

