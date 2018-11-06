import numpy as np
import cv2



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



#def conversion(frame):
#    rows = frame.shape[0]
#    cols = frame.shape[1]
#    curr_frame = f8(frame)
#
#    bayer = np.copy(curr_frame)
#    #kreiraj IR frame upola manji od ulaznog zbog vidiljivosti"
#    IR = np.zeros([rows//2, cols//2], np.uint8)
#    #zamini svako IR komponentu s zelenom BGIRR to BGGR"
#    for x in range(0, rows, 2):
#        for y in range(0, cols, 2):
#            bayer[x+1, y] = curr_frame[x, y+1]
#            IR[x//2, y//2] = curr_frame[x+1, y]
#    BGGR =cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2BGR)
#    return IR, BGGR


#subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_absolute=40",shell=True)



try:
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_absolute=70", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c brightness=30", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c exposure_absolute=70", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c brightness=30", shell=True)
except:
    print("Error occured")


def make_1080p(cap):
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p(cap):
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p(cap):
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)


def conversion(frame):
    curr_frame = f8(frame)
    IR = curr_frame[1::2, 0::2]
    curr_frame[1::2, 0::2] = curr_frame[0::2, 1::2]
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BayerRG2BGR)
    return IR, curr_frame


def canny(frame):
    # manualTreshold = 40
    # manualEdges = cv2.Canny(blur, 0, manualTreshold)
    value = blur(frame)
    return imutils.auto_canny(value)

def f8(frame):
    #return cv2.convertScaleAbs(frame, 0.25)
    return np.array(frame//4, dtype = np.uint8)

def hsv(frame):
    value = cv2.cvtColor(f8(frame), cv2.COLOR_BGR2GRAY)
    return value

def blur(frame):
    # gaussianblur(image, (kernel lenght, width), sigma)
    # cv2.GaussianBlur(frame8bit(frame), (5,5),0)
    return cv2.GaussianBlur(f8(frame), (5, 5), 0)

def gray(frame):
    return cv2.cvtColor(f8(frame), cv2.COLOR_BGR2GRAY)


def dataFrame(left, right):
    print("L frame data is %s \nR frame data is %s" % (
        (left.shape, left.size, left.dtype), (right.shape, right.size, right.dtype)))

def update(stereo):
    # disparity range is tuned for 'aloe' image pair
    stereo.setBlockSize(cv2.getTrackbarPos('window_size', 'disparity'))
    stereo.setUniquenessRatio(cv2.getTrackbarPos('uniquenessRatio', 'disparity'))
    stereo.setSpeckleWindowSize(cv2.getTrackbarPos('speckleWindowSize', 'disparity'))
    stereo.setSpeckleRange(cv2.getTrackbarPos('speckleRange', 'disparity'))
    stereo.setDisp12MaxDiff(cv2.getTrackbarPos('disp12MaxDiff', 'disparity'))


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

def main():
    #data_left = np.load('Left_calibrated.npy').item()
    #Kl, Dl, DIMl = data_left['K'], data_left['D'], data_left['DIM']
    #print(np.array(Kl))
    #print(np.array(Dl))
    #print(DIMl)
    #keys, values = zip(*values_left.items())
    #print(keys,values)
    #stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

    data_left = np.load('Left_calibrated.npy').item()
    data_right = np.load('Right_calibrated.npy').item()

    window_size = 5
    min_disp = 16
    num_disp = 192 - min_disp
    blockSize = window_size
    uniquenessRatio = 1
    speckleRange = 3
    speckleWindowSize = 3
    disp12MaxDiff = 200
    P1 = 600
    P2 = 2400
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=num_disp,
        blockSize=window_size,
        uniquenessRatio=uniquenessRatio,
        speckleRange=speckleRange,
        speckleWindowSize=speckleWindowSize,
        disp12MaxDiff=disp12MaxDiff,
        P1=P1,
        P2=P2
    )
    stereo.setBlockSize(cv2.getTrackbarPos('window_size', 'disparity'))
    stereo.setUniquenessRatio(cv2.getTrackbarPos('uniquenessRatio', 'disparity'))
    stereo.setSpeckleWindowSize(cv2.getTrackbarPos('speckleWindowSize', 'disparity'))
    stereo.setSpeckleRange(cv2.getTrackbarPos('speckleRange', 'disparity'))
    stereo.setDisp12MaxDiff(cv2.getTrackbarPos('disp12MaxDiff', 'disparity'))

    camL = cv2.VideoCapture(0)
    camR = cv2.VideoCapture(1)

    camL.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camL.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    camR.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camR.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

    camL.set(cv2.CAP_PROP_CONVERT_RGB, False)
    camR.set(cv2.CAP_PROP_CONVERT_RGB, False)

    while camL.grab() == True and camR.grab() == True:

        _, frameL = camL.retrieve()
        _, frameR = camR.retrieve()

        IR_L, BGGR_L = conversion(frameL)
        IR_R, BGGR_R = conversion(frameR)

        left_calibrated = undistorted(BGGR_L, data_left)
        right_calibrated = undistorted(BGGR_R, data_right)


       # fps = camL.get(cv2.CAP_PROP_FPS)
        #print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

        #print(IR_L.dtype, BGGR_L.shape)

        #print ("ir_data_function", IR_R)
        #print("BR_data_function", BGGR_L)

        print('computing disparity...')
        disp = stereo.compute(BGGR_L, BGGR_R).astype(np.float32) / 16.0

       # cv2.imshow('left', imgL)
        cv2.imshow('disparity', (disp - min_disp) / num_disp)
        update(stereo=stereo)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



        cv2.namedWindow('disparity')
        cv2.createTrackbar('speckleRange', 'disparity', speckleRange, 50, update)
        cv2.createTrackbar('window_size', 'disparity', window_size, 21, update)
        cv2.createTrackbar('speckleWindowSize', 'disparity', speckleWindowSize, 200, update)
        cv2.createTrackbar('uniquenessRatio', 'disparity', uniquenessRatio, 50, update)
        cv2.createTrackbar('disp12MaxDiff', 'disparity', disp12MaxDiff, 250, update)

        update(stereo=stereo)
        cv2.waitKey()

    camL.release()
    cv2.destroyAllWindows()



main()
np.seterr(divide='ignore', invalid='ignore')



