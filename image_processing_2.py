# -*- coding: utf-8 -*-
"""
Handling Images with OpenCV (and a bit of PIL / imutil)
Created on Thu Apr 29 19:42:53 2021

@author: alex
"""


### Libraries
# pip install imutils
import cv2                # OpenCV
import numpy as np
from PIL import Image
import imutils

# Read the image
#image_name = 'alex'
image_name = '2alex'
path = 'C:/Users/alex/CAS_ML_local/B_Deeplearning/03_Project/x_image_process/'
image = cv2.imread(path + image_name + '.jpg')
pil_image = Image.open(path + image_name + '.jpg')


### Crop Image to square form

# Settings for Crop and shrink 
target_pix = 256
reduce_width = 0.16
shift_down = 380            # plus moves down, minus up
shift_right = 200
# width, height = pil_image.size 
height, width = pil_image.size 
print(height, width)

# apply margins
left = (int(width * reduce_width)) - shift_right
right = (int(width * (1 - reduce_width))) - shift_right
bottom = (int((height - (right - left))) // 2) - shift_down
top = height - (bottom + 2 * shift_down)

# no margins
'''
left = 0
right = width
bottom = 0
top = height
'''

# apply cropping
cropped_img = image[bottom:top, left:right]
print(cropped_img.shape[0], cropped_img.shape[1])

# Resize/shrink based on width (fixed aspect ratio)
r = target_pix / cropped_img.shape[1]
dim = (target_pix, int(cropped_img.shape[0] * r))
resized1 = cv2.resize(cropped_img, dim, interpolation = cv2.INTER_AREA)

# View result
cv2.imshow("Cropped and Shrinked", resized1)
cv2.waitKey(0)

# Save result
cv2.imwrite(path + image_name + '_crop.jpg', resized1)


##############################
###
### other image functions
###
##############################

# Simple display
cv2.imshow("Image", image)
cv2.waitKey(0)

# Fit image in resizable window
cv2.namedWindow('Resize Window', cv2.WINDOW_NORMAL)
cv2.imshow('Resize Window', image)
cv2.waitKey(0)

# Resize window to fit to the screen
screen_res = 1080, 720
scale_width = screen_res[0] / image.shape[1]
scale_height = screen_res[1] / image.shape[0]
scale = min(scale_width, scale_height)
window_width = int(image.shape[1] * scale)
window_height = int(image.shape[0] * scale)
cv2.namedWindow('Scaled to fit', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Scaled to fit', window_width, window_height)
cv2.imshow('Scaled to fit', image)
cv2.waitKey(0)

# Convert to gray
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cv2.namedWindow('Gray Image', cv2.WINDOW_NORMAL)
cv2.imshow('Gray Image', gray_image)
cv2.waitKey(0)

#Rotate by 180'
rotated_img = cv2.rotate(image, cv2.ROTATE_180 )
cv2.namedWindow("Rotated Image", cv2.WINDOW_NORMAL)
cv2.imshow("Rotated Image", rotated_img)
cv2.waitKey(0)

# Rotate image (in steps of 60')
for angle in np.arange(0, 360, 60):
    rotated = imutils.rotate(image, angle)
    cv2.namedWindow("Rotated", cv2.WINDOW_NORMAL)
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)

# Rotate using PIL
rotate_img_pil=pil_image.rotate(90)
rotate_img_pil.show()

# Blow up (fixed aspect ratio)
scale_percent = 200
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized2 = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Resize", resized2)
cv2.waitKey(0)


### Denoising an image

# Original
cv2.namedWindow("Noised Image", cv2.WINDOW_NORMAL)
cv2.imshow("Noised Image", image)
cv2.waitKey(0)

# Denoise options
cv2.namedWindow("Denoised Image", cv2.WINDOW_NORMAL)
denoised_image = cv2.fastNlMeansDenoisingColored(image,None, h=5)
# denoised_image = cv2.GaussianBlur(image, (5,5), 0 )
# denoised_image = cv2.Canny(image, threshold1 =100,threshold2=200 )
cv2.imshow("Denoised Image", denoised_image)
cv2.waitKey(0)
