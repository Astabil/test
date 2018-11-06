#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
import time
#import imutils
import time
import os, errno



def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)
    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1
        flat = np.sort(flat)
        n_cols = flat.shape[0]

        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

img1 = cv2.imread("/home/minneyar/Desktop/stereocam2/dir_right/right_RGB26.jpg") #MARIN USPOEREDBA
img2 = cv2.imread("/home/minneyar/Desktop/stereocam2/rectified imgs matlab/left4.jpg")  # MARIN USPOEREDBA
img3 = cv2.imread("/home/minneyar/Desktop/stereocam2/Depth Slike/Depth_luka.jpg")  # MARIN USPOEREDBA


stackedFrames = np.concatenate((cv2.pyrDown(img1), cv2.pyrDown(simplest_cb(img1, 1))), axis=1)
stackedFrames = np.concatenate((cv2.pyrDown(img2), cv2.pyrDown(simplest_cb(img2, 1))), axis=1)
stackedFrames = np.concatenate((cv2.pyrDown(img3), cv2.pyrDown(simplest_cb(img3, 1))), axis=1)


save = cv2.imwrite('/home/minneyar/Desktop/stereocam2/calib_bouguet/Color conversion Matrix CCM/CCM6.jpg',stackedFrames)

#cv2.imshow("pic", simplest_cb(imgL,1))
cv2.imshow("pic", stackedFrames)

cv2.waitKey()
cv2.destroyAllWindows()