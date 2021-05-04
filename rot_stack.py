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
im_path = os.path.join(data_path, 'acer_campestre', '13291732973114.jpg') #left
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

#img_path = os.path.join(data_path, 'acer_palmatum', '1249061360_0000.jpg')


#im_path = os.path.join(data_path, 'acer_ginnala', '13291762512991.jpg') # right




#clr = Image.open(im_path).resize((640,480))
clr = Image.open(im_path)
#data = np.asarray(clr)


data = get_mask(im_path)


#meeeeeeeeeeeeeeeeee

midx = data.shape[1] // 2
midy = data.shape[0] // 2




def isTopMin(mask):
    
    #mask = crop_image(mask, 10)
    y = int(mask.shape[0] / 2)

            
    
    return mask[:, :y].sum() < mask[:, y:].sum()


data = get_mask(im_path)
clr = Image.fromarray(data.copy())
rot_min_angle = rot_min(data)
#rot_min_angle = - rot_min_angle
print('angle found: ', rot_min_angle)

plt.imshow(clr)
plt.figure()

clr = Image.fromarray(data.copy())
#data = crop_image(data)


res_clr = clr.copy()
res_clr = res_clr.rotate(rot_min_angle)

#data_draw = ImageDraw.Draw(clr)
#data_draw.rectangle([int(data.shape[1] / 2) - 5, 0, int(data.shape[1] / 2) + 5, data.shape[1] -1], fill="black")

#data_draw = ImageDraw.Draw(res_clr)
#data_draw.rectangle([int(res_clr.width / 2) - 2, 0, int(res_clr.width  / 2) + 2, res_clr.height - 1], fill="white")


print('luli value: ', np.asarray(res_clr)[int(res_clr.width / 2 - 2) : int(res_clr.width / 2 + 2), :].sum() )

#res_clr = res_clr.rotate(rot_min_angle)

plt.imshow(res_clr)
plt.figure()

print('is top min: ', isTopMin(data))

