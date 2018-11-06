#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import numpy as np
import cv2
import sys
import os
import subprocess


# default values for brightness and exposure:
# brightness        (int)    : min=0 max=40 step=1 default=10 value=40
# exposure_absolute (int)    : min=1 max=10000 step=1 default=156 value=150

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

def f8(frame):
    return np.array(frame//4, dtype = np.uint8)

def main():
    dir_left = "/home/ivan/PycharmProjects/StereoVision/see3cam/IVAN_stereo/dir_left"
    dir_right = "/home/ivan/PycharmProjects/StereoVision/see3cam/IVAN_stereo/dir_right"

    camL = cv2.VideoCapture(0)
    camR = cv2.VideoCapture(1)
    make_480p(camL)
    make_480p(camR)

   # width = 1280
   # height = 720
   # camL.set(cv2.CAP_PROP_FRAME_WIDTH, width)
   # camL.set(cv2.CAP_PROP_FRAME_WIDTH, height)
   # camR.set(cv2.CAP_PROP_FRAME_WIDTH, width)
   # camR.set(cv2.CAP_PROP_FRAME_WIDTH, height)
#
    camL.set(cv2.CAP_PROP_CONVERT_RGB, False)
    camR.set(cv2.CAP_PROP_CONVERT_RGB, False)



    frameID = 1
    while camL.grab() and camR.grab():

        retL, frameL = camL.retrieve()
        retR, frameR = camR.retrieve()

        IR_L, RGB_L = conversion(frameL)
        IR_R, RGB_R = conversion(frameR)

       # stackedFrames = np.concatenate((RGB_L,RGB_R), axis = 0)
        stackedFrames = np.concatenate((RGB_L, RGB_R), axis=0)
        cv2.imshow('Capture', stackedFrames)

        key = cv2.waitKey(40) & 0xFF
        if (retL == True ) and (retR == True):

            if key == 32:  # Press Space to save img
                print("Saving images:")
                t = str(frameID)
                img_l= 'chessboard-L'+t+'.jpg'.format(frameID).format(frameID)
                img_r= 'chessboard-R'+t+'.jpg'.format(frameID).format(frameID)
                cv2.imwrite(img_l, f8(frameL))
                cv2.imwrite(img_r, f8(frameR))
                print("Image number {}:\n{} is captured".format(frameID, img_l))
                print("Image number {}:\n{} is captured\n".format(frameID, img_r))

                frameID += 1
                if frameID > 30:
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

main()