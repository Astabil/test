#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import numpy as np
import cv2
import sys
import subprocess
import time
from threading import Timer


# default values for brightness and exposure:
# brightness        (int)    : min=0 max=40 step=1 default=10 value=40
# exposure_absolute (int)    : min=1 max=10000 step=1 default=156 value=150

try:
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_absolute=40", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c brightness=30", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c exposure_absolute=40", shell=True)
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


def blur(frame):
    # gaussianblur(image, (kernel lenght, width), sigma)
    # cv2.GaussianBlur(frame8bit(frame), (5,5),0)
    return cv2.GaussianBlur(f8(frame), (5, 5), 0)

def conversion(frame):
    curr_frame = f8(frame)
    IR = curr_frame[1::2, 0::2]
    curr_frame[1::2, 0::2] = curr_frame[0::2, 1::2]
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BayerRG2BGR)
    return IR, curr_frame

def f8(frame):
    return np.array(frame//4, dtype = np.uint8)

def main():
    dir_left  = "/home/minneyar/Desktop/stereocam2/calib_bouguet/Slike - Treshold/dir_left"
    dir_right = "/home/minneyar/Desktop/stereocam2/calib_bouguet/Slike - Treshold/dir_right"

    camL = cv2.VideoCapture(0)
    camR = cv2.VideoCapture(1)
    make_720p(camL)
    make_720p(camR)




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

        BGGR_L = cv2.cvtColor(RGB_L, cv2.COLOR_BGR2GRAY)
        BGGR_R = cv2.cvtColor(RGB_R , cv2.COLOR_BGR2GRAY)

        retL, thresh1L = cv2.threshold(BGGR_L, 220, 255, cv2.THRESH_TOZERO)  # cv2.THRESH_BINARY)
        retR, thresh1R = cv2.threshold(BGGR_R, 220, 255, cv2.THRESH_TOZERO)  # cv2.THRESH_BINARY)

       # stackedFrames = np.concatenate((RGB_L,RGB_R), axis = 0)
        #stackedFrames = np.concatenate((RGB_L, RGB_R), axis=1)
       # cv2.imshow('Capture', stackedFrames)
        #cv2.imshow("FRAME LEFT", RGB_L)
      #  cv2.imshow("FRAME LEFT", frameL.astype("uint8"))

        #cv2.imshow("FRAME RIGHT", RGB_R)
      #  cv2.imshow("FRAME RIGHT",frameR.astype("uint8"))

        #cv2.imshow("IR LEFT", IR_L)
       # cv2.imshow("IR RIGHT", IR_R)

        cv2.imshow("left", thresh1L)
        cv2.imshow("right", thresh1R)

        key = cv2.waitKey(40) & 0xFF
        if retL and retR:

            if key == 32:  # Press Space to save img
                print("Saving images:")
                imgLEFTRGB= dir_left +"/left_RGB{}.jpg".format(frameID)
                imgRIGHTRGB= dir_right +"/right_RGB{}.jpg".format(frameID)
                imgLEFTir = dir_left + "/left_IR{}.jpg".format(frameID)
                imgRIGHir = dir_right + "/right_IR{}.jpg".format(frameID)
                #cv2.imwrite(img_l, frameL)  #orginal
                #cv2.imwrite(img_r, frameR)  #orginal
                cv2.imwrite(imgLEFTRGB, thresh1L)
              #  cv2.imwrite(imgLEFTRGB, frameL.astype("uint8"))
                cv2.imwrite(imgRIGHTRGB, thresh1R)
              #  cv2.imwrite(imgRIGHTRGB, frameR.astype("uint8"))
                cv2.imwrite(imgLEFTir, IR_L)
                cv2.imwrite(imgRIGHir, IR_R)

                print("Image number {}:\n{} is captured".format(frameID, imgLEFTRGB))
                print("Image number {}:\n{} is captured\n".format(frameID, imgRIGHTRGB))
                print("Image number {}:\n{} is captured".format(frameID, imgLEFTir))
                print("Image number {}:\n{} is captured\n".format(frameID, imgRIGHir))

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

main()