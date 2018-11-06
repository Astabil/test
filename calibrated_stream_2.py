#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import subprocess

try:
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_absolute=20", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c brightness=10", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c exposure_absolute=20", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c brightness=10", shell=True)
except:
    print("Error occured")


def conversion(frame):
    curr_frame = f8(frame)
    IR = curr_frame[1::2, 0::2]
    curr_frame[1::2, 0::2] = curr_frame[0::2, 1::2]
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BayerRG2BGR_EA)
    return IR, curr_frame

def f8(frame): return cv2.convertScaleAbs(frame, 0.25)

def undistorted(frame, data):
    data_in = data
    img = frame
    K, D, DIM = data_in['K'], data_in['D'], data_in['DIM']
    K = np.array(K)
    D = np.array(D)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)#cv2.CV_16SC2
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT,  borderValue=29)

    return undistorted_img

def fps(cap, num):
    cap.set(cv2.CAP_PROP_FPS, num)
    return print(cap.get(cv2.CAP_PROP_FPS))

def make_1080p(cap):
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p(cap):
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p(cap):
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(cap,width, height):
    cap.set(3, width)
    cap.set(4, height)



def main():

    data_left = np.load('/home/minneyar/Desktop/stereocam2/Parameters_old/Left_calibrated.npy').item()
    data_right = np.load('/home/minneyar/Desktop/stereocam2/Parameters_old/Right_calibrated.npy').item()



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

        stackedFrames = np.concatenate((left_calibrated,right_calibrated), axis = 1)

        #######TEST##################

        bf8 = np.array(frameL//4, dtype = np.uint8)
        bRGGB = np.copy(bf8)


        cols, rows = frameR.shape[:2]

        IR = np.zeros([rows // 2, cols // 2], np.uint8)

        # copy out IR pixels
        IR = bf8[1::2, 0::2]
        # copy over IR pixels with nearest G pixel
        bf8[1::2, 0::2] = bf8[0::2, 1::2]
        BGRim = cv2.cvtColor(bRGGB, cv2.COLOR_BayerRG2BGR)

        ###################test kraj #########################
        #cv2.imshow("IR_data", BGRim )
        cv2.imshow("IR_data", stackedFrames)
        cv2.imshow("IR_dat2a", frameL.astype("uint8"))
        cv2.imshow("IR_dat2a", IR_R)
        cv2.imshow("as", BGGR_L)




        print(frameR.dtype)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camL.release()
    cv2.destroyAllWindows()



main()
np.seterr(divide='ignore', invalid='ignore')


