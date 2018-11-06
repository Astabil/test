import numpy as np
import cv2
from matplotlib import pyplot as plt
import subprocess


imgl = cv2.imread("/home/minneyar/Desktop/stereocam2/TAMNE_SLIJE/dir_left/LEFT_UNC1.jpg" )  # downscale images for faster processing if you like
imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/TAMNE_SLIJE/dir_left/LEFT_UNC22.jpg" )  # downscale images for faster processing if you like

im_gray = cv2.cvtColor(imgl , cv2.COLOR_BGR2GRAY)

(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
thresh = 167
im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

#image = imgl
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(image, (5, 5), 0)
#cv2.imshow("Image", image)
#thresh = cv2.adaptiveThreshold(blurred, 255,
#cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 3)
#cv2.imshow("Mean Thresh", thresh)
#
#thresh = cv2.adaptiveThreshold(blurred, 255,
#cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
cv2.imshow("Gau",im_bw)
cv2.waitKey(0)