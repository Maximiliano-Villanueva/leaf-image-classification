# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw
import numpy as np
import math
from pprint import pprint
import os
import matplotlib.pyplot as plt
import cv2
import sys
from image_process import * 



data_path = os.path.join('..', 'mrcnn', 'data', 'dataset', 'images', 'field')

"""
top
"""

#im_path = os.path.join(data_path, 'acer_campestre', '13291732970169.jpg') # top
#im_path = os.path.join(data_path, 'acer_campestre', '13291732970228.jpg') # top
#im_path = os.path.join(data_path, 'acer_campestre', '13291732970717.jpg') # top
#im_path = os.path.join(data_path, 'acer_campestre', '13291732972088.jpg') # top
#im_path = os.path.join(data_path, 'acer_campestre', '13291732972114.jpg') # top

#im_path = os.path.join(data_path, 'acer_ginnala', '13291762511387.jpg') # top
#im_path = os.path.join(data_path, 'acer_ginnala', '13291762516851.jpg') # top
#im_path = os.path.join(data_path, 'acer_ginnala', '13291762517273.jpg') # top


"""
bottom
"""

#im_path = os.path.join(data_path, 'acer_campestre', '13291732970228.jpg') #bottom
#im_path = os.path.join(data_path, 'acer_campestre', '13291732970717.jpg') #bottom
#im_path = os.path.join(data_path, 'acer_campestre', '13291732971024.jpg') #bottom
#im_path = os.path.join(data_path, 'acer_campestre', '13291732972031.jpg') #bottom
#im_path = os.path.join(data_path, 'acer_campestre', '13291732972088.jpg') #bottom


#im_path = os.path.join(data_path, 'acer_ginnala', '13291762515777.jpg') #bottom
#im_path = os.path.join(data_path, 'acer_ginnala', '13291762516851.jpg') #bottom
#im_path = os.path.join(data_path, 'acer_ginnala', '13291762517273.jpg') #bottom


"""
left
"""

#im_path = os.path.join(data_path, 'acer_ginnala', '13291762511614.jpg') # left


#im_path = os.path.join(data_path, 'acer_ginnala', '13291762510376.jpg') # left
#im_path = os.path.join(data_path, 'acer_ginnala', '13291762511614.jpg') # left
#im_path = os.path.join(data_path, 'acer_ginnala', '13291762512231.jpg') # left
#im_path = os.path.join(data_path, 'acer_ginnala', '13291762512931.jpg') # left --stack

#im_path = os.path.join(data_path, 'acer_campestre', '13291732970222.jpg') #left --maaaal
#im_path = os.path.join(data_path, 'acer_campestre', '13291732971753.jpg') #left
#im_path = os.path.join(data_path, 'acer_campestre', '13291732973114.jpg') #left
#im_path = os.path.join(data_path, 'acer_campestre', '13291732973267.jpg') #left

"""
def findCountours(thresh):
    
    thresh = crop_image(thresh)
    
    horizontal = int((thresh.shape[1] - 1) / 4)
    vertical = int((thresh.shape[0] - 1) / 4)
    #find countours and keep max
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    
    #find center of image with moments (given contour)
    moment = cv2.moments(big_contour)
    
    cX = int(moment["m10"] / moment["m00"])
    
    cY = int(moment["m01"] / moment["m00"])
    
    #split image in right and left 
    thresh_right = thresh[cY:,:]
    thresh_left = thresh[:cY,:]
    init_ind_vert = 0
    end_ind_vert = 0
    
    init_ind_hor = 0
    end_ind_hor = 0
    
    min_val = 999999999999999
    
    tol = 25000
    for i in range(0,4):
        
        final_ind = (i+1) * vertical if (i+1) * vertical < thresh.shape[0]-1 else thresh.shape[0]-1
        
        for j in range(0,4):
            
            final_ind_hor = (j+1) * horizontal if (j+1) * horizontal < thresh.shape[1]-1 else thresh.shape[1]-1
            
            val = thresh[i*vertical: (i+1) * vertical, j*horizontal: (j+1) * horizontal].sum()
            
            if val < min_val and val > tol:
                min_val = val
                init_ind_vert = i*vertical
                end_ind_vert = final_ind
                
                init_ind_hor = i*horizontal
                end_ind_hor = final_ind_hor 
            
    print ((init_ind_vert, init_ind_hor, min_val))
    #flip right side
    #thresh_right = cv2.flip(thresh_right, 0)
    
    #overlay left and flipped right
    #thresh_overlay = thresh_left + thresh_right
    
    result = thresh.copy()
    
    #cv2.circle(result, (int(cX),int(cY)), 10, (0, 0, 0), -1)
    
    #moment
    return thresh[init_ind_vert :  end_ind_vert, init_ind_hor : end_ind_hor]#thresh[300:400,250:300]
    
"""

def process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 2)
    img_canny = cv2.Canny(img_blur, 127, 47)
    kernel = np.ones((5, 5))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode

def get_contours(img):
    contours, _ = cv2.findContours(process(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, 0.01 * peri, True)

def get_angle(a, b, c):
    ba, bc = a - b, c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cos_angle))
    
def get_rot_angle(img):
    contours = get_contours(img)
    length = len(contours)
    min_angle = 180
    for i in cv2.convexHull(contours, returnPoints=False).ravel():
        a, b, c = contours[[i - 1, i, (i + 1) % length], 0]
        angle = get_angle(a, b, c)
        if angle < min_angle:
            min_angle = angle
            pts = a, b, c
    a, b, c = pts
    return 180 - np.degrees(np.arctan2(*(np.mean((a, c), 0) - b)))

def rotate(img):
    h, w, _ = img.shape
    angle = get_rot_angle(img)
    rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR), angle

img = cv2.imread(im_path)

plt.imshow(img)
plt.figure()
#cv2.imshow("Image", rotate(img))
rot_img, angle = rotate(img)

mask = get_mask(im_path)
mask = Image.fromarray(mask)
mask = mask.rotate(angle)



plt.imshow(rot_img)
plt.figure()

plt.imshow(mask)
plt.figure()
#cv2.waitKey(0)


"""
#clr = Image.open(im_path).resize((640,480))
clr = Image.open(im_path)
#data = np.asarray(clr)


data = get_mask(im_path)

res = findCountours(data)
print(res.sum())
plt.imshow(res)
plt.figure()
"""