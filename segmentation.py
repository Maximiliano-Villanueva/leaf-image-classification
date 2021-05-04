



    
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from PIL import Image
import os
import math, sys

data_path = os.path.join('..', 'mrcnn', 'data', 'dataset', 'images', 'field')

#cap aon apunta fulla

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
right
"""
#im_path = os.path.join(data_path, 'acer_griseum', '13001148650053.jpg') # right
#im_path = os.path.join(data_path, 'acer_griseum', '13001148651083.jpg') # right
#im_path = os.path.join(data_path, 'acer_griseum', '13001148651245.jpg') # right
#im_path = os.path.join(data_path, 'acer_griseum', '13001148651818.jpg') # right


#im_path = os.path.join(data_path, 'acer_ginnala', '13291762512991.jpg') # right
#im_path = os.path.join(data_path, 'acer_ginnala', '13291762514877.jpg') # right
#im_path = os.path.join(data_path, 'acer_ginnala', '13291762516205.jpg') # right
#im_path = os.path.join(data_path, 'acer_ginnala', '13291762517069.jpg') # right



#im_path = os.path.join(data_path, 'acer_negundo', '13001151160340.jpg') # right
#im_path = os.path.join(data_path, 'acer_negundo', '13001151161516.jpg') 

im_path = os.path.join(data_path, 'acer_palmatum', '1249061360_0000.jpg')






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



    
    
def get_thresholded_rotated(im_path):
    
    #read image
    img = cv2.imread(im_path)
    
    img = cv2.resize(img, (600, 800), interpolation = Image.BILINEAR)
    
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
    
    #find countours and keep max
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
        
    # fit ellipse to leaf contours
    ellipse = cv2.fitEllipse(big_contour)
    (xc,yc), (d1,d2), angle = ellipse
    
    print('thresh shape: ', thresh.shape)
    #print(xc,yc,d1,d2,angle)
    
    rmajor = max(d1,d2)/2
    
    rminor = min(d1,d2)/2
    
    origi_angle = angle
    
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90
        
    #calc major axis line
    xtop = xc + math.cos(math.radians(angle))*rmajor
    ytop = yc + math.sin(math.radians(angle))*rmajor
    xbot = xc + math.cos(math.radians(angle+180))*rmajor
    ybot = yc + math.sin(math.radians(angle+180))*rmajor
    
    #calc minor axis line
    xtop_m = xc + math.cos(math.radians(origi_angle))*rminor
    ytop_m = yc + math.sin(math.radians(origi_angle))*rminor
    xbot_m = xc + math.cos(math.radians(origi_angle+180))*rminor
    ybot_m = yc + math.sin(math.radians(origi_angle+180))*rminor
    
    #determine which region is up and which is down
    if max(xtop, xbot) == xtop :
        x_tij = xtop
        y_tij = ytop
        
        x_b_tij = xbot
        y_b_tij = ybot
    else:
        x_tij = xbot
        y_tij = ybot
        
        x_b_tij = xtop
        y_b_tij = ytop
        
    
    if max(xtop_m, xbot_m) == xtop_m :
        x_tij_m = xtop_m
        y_tij_m = ytop_m
        
        x_b_tij_m = xbot_m
        y_b_tij_m = ybot_m
    else:
        x_tij_m = xbot_m
        y_tij_m = ybot_m
        
        x_b_tij_m = xtop_m
        y_b_tij_m = ytop_m
        
        
    print('-----')
    print(x_tij, y_tij)
    
    

    
    rect_size = 100
    
    """
    calculate regions of edges of major axis of ellipse
    this is done by creating a squared region of rect_size x rect_size, being the edge the center of the square
    """
    x_min_tij = int(0 if x_tij - rect_size < 0 else x_tij - rect_size)
    x_max_tij = int(thresh.shape[1]-1 if x_tij + rect_size > thresh.shape[1] else x_tij + rect_size)
    
    y_min_tij = int(0 if y_tij - rect_size < 0 else y_tij - rect_size)
    y_max_tij = int(thresh.shape[0] - 1 if y_tij + rect_size > thresh.shape[0] else y_tij + rect_size)
  
    
    x_b_min_tij = int(0 if x_b_tij - rect_size < 0 else x_b_tij - rect_size)
    x_b_max_tij = int(thresh.shape[1] - 1 if x_b_tij + rect_size > thresh.shape[1] else x_b_tij + rect_size)
    
    y_b_min_tij = int(0 if y_b_tij - rect_size < 0 else y_b_tij - rect_size)
    y_b_max_tij = int(thresh.shape[0] - 1 if y_b_tij + rect_size > thresh.shape[0] else y_b_tij + rect_size)
    

    sum_red_region =   np.sum(thresh[y_min_tij:y_max_tij, x_min_tij:x_max_tij])

    sum_yellow_region =   np.sum(thresh[y_b_min_tij:y_b_max_tij, x_b_min_tij:x_b_max_tij])
    
    
    """
    calculate regions of edges of minor axis of ellipse
    this is done by creating a squared region of rect_size x rect_size, being the edge the center of the square
    """
    x_min_tij_m = int(0 if x_tij_m - rect_size < 0 else x_tij_m - rect_size)
    x_max_tij_m = int(thresh.shape[1]-1 if x_tij_m + rect_size > thresh.shape[1] else x_tij_m + rect_size)
    
    y_min_tij_m = int(0 if y_tij_m - rect_size < 0 else y_tij_m - rect_size)
    y_max_tij_m = int(thresh.shape[0] - 1 if y_tij_m + rect_size > thresh.shape[0] else y_tij_m + rect_size)
  
    
    x_b_min_tij_m = int(0 if x_b_tij_m - rect_size < 0 else x_b_tij_m - rect_size)
    x_b_max_tij_m = int(thresh.shape[1] - 1 if x_b_tij_m + rect_size > thresh.shape[1] else x_b_tij_m + rect_size)
    
    y_b_min_tij_m = int(0 if y_b_tij_m - rect_size < 0 else y_b_tij_m - rect_size)
    y_b_max_tij_m = int(thresh.shape[0] - 1 if y_b_tij_m + rect_size > thresh.shape[0] else y_b_tij_m + rect_size)
    
    #value of the regions, the names of the variables are related to the color of the rectangles drawn at the end of the function
    sum_red_region_m =   np.sum(thresh[y_min_tij_m:y_max_tij_m, x_min_tij_m:x_max_tij_m])

    sum_yellow_region_m =   np.sum(thresh[y_b_min_tij_m:y_b_max_tij_m, x_b_min_tij_m:x_b_max_tij_m])
    
 
    #print(sum_red_region, sum_yellow_region, sum_red_region_m, sum_yellow_region_m)
    
    
    min_arg = np.argmin(np.array([sum_red_region, sum_yellow_region, sum_red_region_m, sum_yellow_region_m]))
    
    print('min: ', min_arg)
       
    
    if min_arg == 1: #sum_yellow_region < sum_red_region :
        
        
        left_quartile = x_b_tij < thresh.shape[0] /2 
        upper_quartile = y_b_tij < thresh.shape[1] /2

        center_x = x_b_min_tij + ((x_b_max_tij - x_b_min_tij) / 2)
        center_y = y_b_min_tij + (y_b_max_tij - y_b_min_tij / 2)
        

        center_x = x_b_min_tij + np.argmax(thresh[y_b_min_tij:y_b_max_tij, x_b_min_tij:x_b_max_tij].mean(axis=0))
        center_y = y_b_min_tij + np.argmax(thresh[y_b_min_tij:y_b_max_tij, x_b_min_tij:x_b_max_tij].mean(axis=1))

    elif min_arg == 0:
        
        left_quartile = x_tij < thresh.shape[0] /2 
        upper_quartile = y_tij < thresh.shape[1] /2


        center_x = x_min_tij + ((x_b_max_tij - x_b_min_tij) / 2)
        center_y = y_min_tij + ((y_b_max_tij - y_b_min_tij) / 2)

        
        center_x = x_min_tij + np.argmax(thresh[y_min_tij:y_max_tij, x_min_tij:x_max_tij].mean(axis=0))
        center_y = y_min_tij + np.argmax(thresh[y_min_tij:y_max_tij, x_min_tij:x_max_tij].mean(axis=1))
        
    elif min_arg == 3:
        
        
        left_quartile = x_b_tij_m < thresh.shape[0] /2 
        upper_quartile = y_b_tij_m < thresh.shape[1] /2

        center_x = x_b_min_tij_m + ((x_b_max_tij_m - x_b_min_tij_m) / 2)
        center_y = y_b_min_tij_m + (y_b_max_tij_m - y_b_min_tij_m / 2)
        

        center_x = x_b_min_tij_m + np.argmax(thresh[y_b_min_tij_m:y_b_max_tij_m, x_b_min_tij_m:x_b_max_tij_m].mean(axis=0))
        center_y = y_b_min_tij_m + np.argmax(thresh[y_b_min_tij_m:y_b_max_tij_m, x_b_min_tij_m:x_b_max_tij_m].mean(axis=1))

    else:
        

        
        left_quartile = x_tij_m < thresh.shape[0] /2 
        upper_quartile = y_tij_m < thresh.shape[1] /2


        center_x = x_min_tij_m + ((x_b_max_tij_m - x_b_min_tij_m) / 2)
        center_y = y_min_tij_m + ((y_b_max_tij_m - y_b_min_tij_m) / 2)

        
        center_x = x_min_tij_m + np.argmax(thresh[y_min_tij_m:y_max_tij_m, x_min_tij_m:x_max_tij_m].mean(axis=0))
        center_y = y_min_tij_m + np.argmax(thresh[y_min_tij_m:y_max_tij_m, x_min_tij_m:x_max_tij_m].mean(axis=1))
        
                      

    # draw ellipse on copy of input
    result = img.copy() 
    cv2.ellipse(result, ellipse, (0,0,255), 1)
    
    
    cv2.line(result, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (255, 0, 0), 1)
    cv2.circle(result, (int(xc),int(yc)), 10, (255, 255, 255), -1)

    cv2.circle(result, (int(center_x),int(center_y)), 10, (255, 0, 255), 5)

    cv2.circle(result, (int(thresh.shape[1] / 2),int(thresh.shape[0] - 1)), 10, (255, 0, 0), 5)


    cv2.rectangle(result,(x_min_tij,y_min_tij),(x_max_tij,y_max_tij),(255,0,0),3)
    cv2.rectangle(result,(x_b_min_tij,y_b_min_tij),(x_b_max_tij,y_b_max_tij),(255,255,0),3)
    
    cv2.rectangle(result,(x_min_tij_m,y_min_tij_m),(x_max_tij_m,y_max_tij_m),(255,0,0),3)
    cv2.rectangle(result,(x_b_min_tij_m,y_b_min_tij_m),(x_b_max_tij_m,y_b_max_tij_m),(255,255,0),3)
    
    
        
    
    
    
    
    plt.imshow(result)
    plt.figure()
    #rotate the image    
    rot_img = Image.fromarray(thresh)
        
    #180
    bot_point_x = int(thresh.shape[1] / 2)
    bot_point_y = int(thresh.shape[0] - 1)
    
    #poi
    poi_x = int(center_x)
    poi_y = int(center_y)
    
    #image_center
    im_center_x = int(thresh.shape[1] / 2)
    im_center_y = int(thresh.shape[0] - 1) / 2
    
    #a - adalt, b - abaix, c - dreta
    #ba = a - b
    #bc = c - a(b en realitat) 
    
    ba = np.array([im_center_x, im_center_y]) - np.array([bot_point_x, bot_point_y])
    bc = np.array([poi_x, poi_y]) - np.array([im_center_x, im_center_y])
    
    

    #angle 3 punts    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.arccos(cosine_angle)
    
    cos_angle = np.degrees(cos_angle)
    
    print('cos angle: ', cos_angle)
    
    print('print: ', abs(poi_x- bot_point_x))
    
    m = (int(thresh.shape[1] / 2)-int(center_x) / int(thresh.shape[0] - 1)-int(center_y))
    
    ttan = math.tan(m)
    
    theta = math.atan(ttan)
        
    print('theta: ', theta) 
    


    result = Image.fromarray(result)
    
    result = result.rotate(cos_angle)
    
    plt.imshow(result)
    plt.figure()

    #rot_img = rot_img.rotate(origi_angle)

    rot_img = rot_img.rotate(cos_angle)

    return rot_img


rot_img = get_thresholded_rotated(im_path)

plt.imshow(rot_img)

