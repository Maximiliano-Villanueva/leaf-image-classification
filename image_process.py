import numpy as np
import math

from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
# -*- coding: utf-8 -*-
def rads(degs):
    return degs * math.pi / 180.0

def brightness_distortion(I, mu, sigma):
    return np.sum(I*mu/sigma**2, axis=-1) / np.sum((mu/sigma)**2, axis=-1)


def chromacity_distortion(I, mu, sigma):
    alpha = brightness_distortion(I, mu, sigma)[...,None]
    return np.sqrt(np.sum(((I - alpha * mu)/sigma)**2, axis=-1))

def bwareafilt ( image ):
    image = image.astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255

    return img2


def get_mask(im_path):
    #read image
    img = cv2.imread(im_path)
    
    #img = cv2.resize(img, (600, 800), interpolation = Image.BILINEAR)
    
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1]
    val = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2]
    sat = cv2.medianBlur(sat, 11)
    val = cv2.medianBlur(val, 11)
    
    #create threshold
    thresh_S = cv2.adaptiveThreshold(sat , 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 401, 10);
    thresh_V = cv2.adaptiveThreshold(val , 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 401, 10);
    
    #mean, std
    mean_S, stdev_S = cv2.meanStdDev(img, mask = 255 - thresh_S)
    mean_S = mean_S.ravel().flatten()
    stdev_S = stdev_S.ravel()
    
    #chromacity
    chrom_S = chromacity_distortion(img, mean_S, stdev_S)
    chrom255_S = cv2.normalize(chrom_S, chrom_S, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)[:,:,None]
    
    mean_V, stdev_V = cv2.meanStdDev(img, mask = 255 - thresh_V)
    mean_V = mean_V.ravel().flatten()
    stdev_V = stdev_V.ravel()
    chrom_V = chromacity_distortion(img, mean_V, stdev_V)
    chrom255_V = cv2.normalize(chrom_V, chrom_V, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)[:,:,None]
    
    #create different thresholds
    thresh2_S = cv2.adaptiveThreshold(chrom255_S , 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 401, 10);
    thresh2_V = cv2.adaptiveThreshold(chrom255_V , 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 401, 10);
        

    #thresholded image
    thresh = cv2.bitwise_and(thresh2_S, cv2.bitwise_not(thresh2_V))
    
    return thresh



def crop_image(img,tol=5000, window=3):
    # img is 2D image data
    # tol  is tolerance
    #img = img / 255
    y = img.shape[1]-1
    x = img.shape[0]-1
    print(img.shape)
    print('x: ', x)
    print('y: ', y)
    test = None
    test_y = None
    for ix in range(0, x, window):
        
        i = ix-window if (ix + window) > x else ix
        
        if img[i : i + window, :].sum() >= tol:
            
            if test is None:
                test = img[i:i+window, : ]
                
            else:
                test = np.append(test, img[i:i+window, : ], 0)
                
    y = test.shape[1]-1
    
    
    for ix in range(0, y, window):
        
        i = ix-window if (ix + window) > y else ix
        
        if test[:,i : i + window].sum() >= tol:
            
            if test_y is None:
                test_y = test[:,i:i+window]
                
            else:
                test_y = np.append(test_y, test[:,i:i+window ], 1)
            

            
    
                    
    print('img ', test_y.shape, test_y.sum())
    return test_y


def rot_min_old(mask):
        
    mask = crop_image(mask)
    
    plt.imshow(mask)
    plt.figure()
    max_val = 0
    angle = 0
    
    x = int(mask.shape[1] / 2)
    
    mask = Image.fromarray(mask)
    
    for a in range(-181,181, 1):
                
        rot_img = mask.copy()
        
        rot_img = rot_img.rotate(a)
        
        res = np.asarray(rot_img)
        
        x = int(res.shape[1] / 2)
        
        curr_val = res[:,x-2 : x+2].sum()
        if a == -82:
            print('-82: ', curr_val)
        if curr_val > max_val:
            
            y = int(res.shape[0] / 2)
            #careful with reverted image
            
            if res[:y,x-2 : x+2].sum() >= res[y:,x-2 : x+2].sum():
                max_val = curr_val
                angle = a
            


    print('value: ', max_val)
    return angle




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