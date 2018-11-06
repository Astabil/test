#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
import time
#import imutils
import time
import subprocess


try:
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_absolute=20", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c brightness=80", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c exposure_absolute=20", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c brightness=60", shell=True)
except:
    print("Error occured")

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
import numpy as np
import cv2
import subprocess
import os, errno


def conversion(frame):
    curr_frame = f8(frame)
    IR = curr_frame[1::2, 0::2]
    curr_frame[1::2, 0::2] = curr_frame[0::2, 1::2]
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BayerRG2BGR_EA)
    return IR, curr_frame

def f8(frame): return np.array(frame//4, dtype = np.uint8)

def make_720p(cap):
    cap.set(3, 1280)
    cap.set(4, 720)

def take_pictures():

    dir_left = "/home/minneyar/Desktop/stereocam2/Slije za usporedbu/dir_left"
    dir_right = "/home/minneyar/Desktop/stereocam2/Slije za usporedbu/dir_right"    #if not os.path.exists(directory):
    #    os.makedirs(directory)
    #else:
    #    pass

    try:
        os.makedirs(dir_left)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    try:
        os.makedirs(dir_right)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    camL = cv2.VideoCapture(0)
    camR = cv2.VideoCapture(1)
    camL.set(cv2.CAP_PROP_CONVERT_RGB, False)
    camR.set(cv2.CAP_PROP_CONVERT_RGB, False)

    make_720p(camL)
    make_720p(camR)

    frameID = 1
    while camL.grab() and camR.grab():

        retL, frameL = camL.retrieve()
        retR, frameR = camR.retrieve()

        IR_L, RGB_L = conversion(frameL)
        IR_R, RGB_R = conversion(frameR)

        # stackedFrames = np.concatenate((RGB_L,RGB_R), axis = 0)
        # stackedFrames = np.concatenate((RGB_L, RGB_R), axis=1)
        # cv2.imshow('Capture', stackedFrames)
        cv2.imshow("FRAME LEFT", cv2.pyrDown(simplest_cb(RGB_L,1)))
        cv2.imshow("FRAME RIGHT", cv2.pyrDown(simplest_cb(RGB_R,1)))

        key = cv2.waitKey(40) & 0xFF

        if retL and retR:
            if key == 32:  # Press Space to save img
                print("Saving images")
                img_l = dir_left + "/LEFT_UNC{}.jpg".format(frameID)
                img_r = dir_right + "/RIGHT_UNC{}.jpg".format(frameID)
                # cv2.imwrite(img_l, frameL)  #orginal
                # cv2.imwrite(img_r, frameR)  #orginal
                cv2.imwrite(img_l, RGB_L)
                cv2.imwrite(img_r, RGB_R)

                frameID += 1
                if frameID > 34:
                    break

        if key == ord('q'):
            break

        elif key == ord('s'):
            tmp = camL
            camL = camR
            camR = tmp

    camL.release()
    camR.release()
    cv2.destroyAllWindows()

take_pictures()

