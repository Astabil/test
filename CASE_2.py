#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import os
import time
import imutils
import time
import subprocess


"""izračun stereo kalibracije rms visok"""

try:
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_absolute=70", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c brightness=20", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c exposure_absolute=70", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c brightness=20", shell=True)
except:
    print("Error occured")



#RADI!!!!
def undistorted(frame, data):
    data_in = data
    img = frame
    K, D, DIM = data_in['K'], data_in['D'], data_in['DIM']
    K = np.array(K)
    D = np.array(D)
    #h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)#cv2.CV_16SC2
   # undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT,  borderValue=29)

    return undistorted_img

def RGB_false(cap):
    return cap.set(cv2.CAP_PROP_CONVERT_RGB, False)

def f8(frame):
    #return cv2.convertScaleAbs(frame, 0.25)
    return np.array(frame//4, dtype = np.uint8)

def make_480p(cap):
    cap.set(3, 640)
    cap.set(4, 480)

def main():
    print(cv2.useOptimized())

    data_left = np.load('Left_calibrated.npy').item()
    data_right = np.load('Right_calibrated.npy').item()

    data = np.load('intrs.npy').item()
    objp_left,  objp_right, imgp_left, imgp_right = data['OBJL'], data['OBJR'], data['IMPL'], data['IMGR']
    KL, DL, DIML = data_left['K'], data_left['D'], data_left['DIM']
    KR, DR, DIMR = data_right['K'], data_right['D'], data_right['DIM']
    KL = np.array(KL)
    DL = np.array(DL)
    KR = np.array(KR)
    DR = np.array(DR)

    flags = 0
   # flags |= cv2.CALIB_ZERO_TANGENT_DIST
    flags |= cv2.fisheye.CALIB_FIX_SKEW
   # flags |= cv2.CALIB_ZERO_TANGENT_DIST
    flags |= cv2.fisheye.CALIB_CHECK_COND
    flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    flags |= cv2.fisheye.CALIB_FIX_K4


    termination_criteria_extrinsics = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    (rms_stereo, _, _, _, _, R, T, E, F) = \
        cv2.stereoCalibrate(objp_right, imgp_left, imgp_right, KL, DL, KR, DR, DIML,
                            criteria=termination_criteria_extrinsics, flags = flags)
    print(rms_stereo)

    camL = cv2.VideoCapture(0)
    camR = cv2.VideoCapture(1)

    RGB_false(camL)
    RGB_false(camR)
    make_480p(camL)
    make_480p(camR)
    while camL.grab() and camR.grab():

        e1 = cv2.getTickCount()


        _, frameL = camL.retrieve()
        _, frameR = camR.retrieve()

        #print(frameR.shape)

        left_calibrated = undistorted(f8(frameL), data_left)
        right_calibrated = undistorted(f8(frameR), data_right)

       #fps = camL.get(cv2.CAP_PROP_FPS)
       #print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


        stackedFrames = np.concatenate((left_calibrated,right_calibrated), axis = 1)
        cv2.imshow("calibrated", stackedFrames)

        #cv2.imshow("calibrated", left_calibrated)

        key = cv2.waitKey(40) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            tmp = camL
            camL = camR
            camR = tmp

        e2 = cv2.getTickCount()
        t = (e2 - e1) / cv2.getTickFrequency()
        print(t)

    camL.release()
    cv2.destroyAllWindows()



main()
np.seterr(divide='ignore', invalid='ignore')



