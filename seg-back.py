



    
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
    
    #images = [img, thresh_S, thresh_V, cv2.bitwise_and(thresh2_S, cv2.bitwise_not(thresh2_V))]
    #titles = ['Original Image', 'Mask S', 'Mask V', 'S + V']
    
    
    
    
    
    
    #thresholded image
    thresh = cv2.bitwise_and(thresh2_S, cv2.bitwise_not(thresh2_V))
    
    """
    lower=(0,0,0)
    upper=(130,190,140)
    thresh = cv2.inRange(img, lower, upper)
    """
    #find countours and keep max
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
        
    # fit ellipse to leaf contours
    ellipse = cv2.fitEllipse(big_contour)
    (xc,yc), (d1,d2), angle = ellipse
    
    print('thresh shape: ', thresh.shape)
    print(xc,yc,d1,d2,angle)
    
    rmajor = max(d1,d2)/2
    
    origi_angle = angle
    
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90
    print('luli angle: ', angle)
    
    xtop = xc + math.cos(math.radians(angle))*rmajor
    ytop = yc + math.sin(math.radians(angle))*rmajor
    xbot = xc + math.cos(math.radians(angle+180))*rmajor
    ybot = yc + math.sin(math.radians(angle+180))*rmajor
    
    print(xtop, ytop, xbot, ybot)
    
    suml = np.sum(thresh[:int(xc), :])
    sumr = np.sum(thresh[int(xc):, :])
    
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
        
        
    print('-----')
    print('tij')
    print(x_tij, y_tij)
    
    x_min_tij = int((x_tij - 100) + x_tij if x_tij - 100 < 0 else x_tij - 100)
    x_max_tij = int(thresh.shape[0] if x_tij + 100 > thresh.shape[0] else x_tij + 100)
    
    y_min_tij = int(0 if y_tij - 100 < 0 else y_tij - 100)
    y_max_tij = int(thresh.shape[1] if y_tij + 100 > thresh.shape[1] else y_tij + 100)
  
    
    x_b_min_tij = int(0 if x_b_tij - 100 < 0 else x_b_tij - 100)
    x_b_max_tij = int(thresh.shape[0] if x_b_tij + 100 > thresh.shape[0] else x_b_tij + 100)
    
    y_b_min_tij = int(0 if y_b_tij - 100 < 0 else y_b_tij - 100)
    y_b_max_tij = int(thresh.shape[1] if y_b_tij + 100 > thresh.shape[1] else y_b_tij + 100)
    

    
    sum_left_region =   np.sum(thresh[x_min_tij:x_max_tij, y_min_tij:y_max_tij])
    
    sum_right_region =   np.sum(thresh[x_b_min_tij:x_b_max_tij, y_b_min_tij:y_b_max_tij])
    
    print('sum_right_region: ', sum_right_region, 'sum_left_region: ', sum_left_region)
    
    print('tija left: ', sum_left_region < sum_right_region)
    
    #yellow is lower and right
    #red is left and red
    if sum_left_region < sum_right_region:
        left_quartile = x_tij < thresh.shape[0] /2 
        upper_quartile = y_tij < thresh.shape[1] /2 
    else:
        
        left_quartile = x_b_tij < thresh.shape[0] /2 
        upper_quartile = y_b_tij < thresh.shape[1] /2
        
    print('left_quartile: ', left_quartile)
    print('upper_quartile', upper_quartile)
    
    #xbot < xtop
    if  suml > sumr:
        origi_angle = - origi_angle
        print ('esquerra')
    else:
        print ('dreta')
                
    
    
    #convert angle to negative in order to get the stem in vertical position at the bottom

    
    print('sum left: ', suml, 'sum right: ', sumr)
    """
    center_x = np.argmax(thresh.mean(axis=0))
    center_y = np.argmax( thresh.mean(axis=1))
    
    print ('center_x: ', center_x, 'center_y: ', center_y)
    
    if center_x < thresh.shape[0] / 2:
        print('centre masses esquerra')
        #angle = - angle
    else:
        print('centre masses dreta')
    
    
    if center_y < thresh.shape[1] / 2:
        print('centre masses amunt')
    else:
        print('centre masses avall')
        
    if xc < thresh.shape[0] / 2:
        print('centre ellipse esquerra')

    else:
        print('centre ellipse dreta')
        
    if center_y < thresh.shape[1] / 2:
        print('centre ellipse amunt')
    else:
        print('centre ellipse avall')
    
    twidth = sum((thresh > 21).any(axis=0))
    theight = sum((thresh > 21).any(axis=1))
    
    print('vertical: ', theight > twidth)

    print('mean raro x: ', center_x)
    print('mean raro y : ', center_y)
    """
    """
    if xc < yc:
        
        print('vertical')
        if angle > 90:
            print('up')
            #angle = angle + 180
        else:
            print('down')
            #angle = angle #angle - 180
    else:
        print('horizontal')
        
        if xc < thresh.shape[0] / 2:
            
            print('centre esquerra')
            
            
        else:
            print('centre dreta')
            angle = angle - 180
    """
    
    
    
    """
    if angle < 85:
        print('angle < 85')
        angle = angle - 90
    else:
        angle = angle + 180
        print('angle > 85')
    """
            
    
            
    
    #angle = angle
    
    # draw ellipse on copy of input
    result = img.copy() 
    cv2.ellipse(result, ellipse, (0,0,255), 1)
    
    print(x_min_tij, x_max_tij)
    print(y_min_tij, y_max_tij)
    
    print(x_b_min_tij, x_b_max_tij)
    print(y_b_min_tij, y_b_max_tij)
    
    cv2.line(result, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (255, 0, 0), 1)
    cv2.circle(result, (int(xc),int(yc)), 10, (255, 255, 255), -1)
    cv2.rectangle(result,(x_min_tij,y_min_tij),(x_max_tij,y_max_tij),(255,0,0),3)
    cv2.rectangle(result,(x_b_min_tij,y_b_min_tij),(x_b_max_tij,y_b_max_tij),(255,255,0),3)
    plt.imshow(result)
    plt.figure()
    #rotate the image    
    rot_img = Image.fromarray(thresh)
        
    rot_img = rot_img.rotate(origi_angle)

    return rot_img


rot_img = get_thresholded_rotated(im_path)
#rot_img.save('x.jpg', 'JPEG')
plt.imshow(rot_img)
#plt.figure()
#rot_img = get_rotation(np.array(rot_img))
#plt.imshow(rot_img)