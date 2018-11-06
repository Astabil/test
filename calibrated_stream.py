#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
import time
#import imutils
import time
import subprocess



# default values for brightness and exposure:
# brightness        (int)    : min=0 max=40 step=1 default=10 value=40
# exposure_absolute (int)    : min=1 max=10000 step=1 default=156 value=150

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
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_absolute=10", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c brightness=40", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c exposure_absolute=33", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c brightness=40", shell=True)
except:
    print("Error occured")

def conversion(frame):
    curr_frame = f8(frame)
    IR = curr_frame[1::2, 0::2]
    curr_frame[1::2, 0::2] = curr_frame[0::2, 1::2]
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BayerRG2BGR_EA)
    return IR, curr_frame


#def canny(frame):
#    # manualTreshold = 40
#    # manualEdges = cv2.Canny(blur, 0, manualTreshold)
#    value = blur(frame)
#    return imutils.auto_canny(value)

def f8(frame):
    return cv2.convertScaleAbs(frame, 0.25)
    #return np.array(frame//4, dtype = np.uint8)

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

#RADI!!!!
def undistorted(frame, data):
    data_in = data
    img = frame
    K, D, DIM = data_in['K'], data_in['D'], data_in['DIM']
    K = np.array(K)
    D = np.array(D)

   # h, w = img.shape[:2]

    #map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3),
    #                                                 K, (h,w), cv2.CV_16SC2)  # cv2.CV_16SC2

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3),
                                                     K, DIM, cv2.CV_16SC2)#cv2.CV_16SC2
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



    camL = cv2.VideoCapture(0)
    camR = cv2.VideoCapture(1)

    #camL.set(cv2.CAP_PROP_MODE, 1)
    #camR.set(cv2.CAP_PROP_MODE, 1)

    L = make_720p(camL)
    R = make_720p(camR)


   # camL.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
   # camL.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
   # camR.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
   # camR.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

    camL.set(cv2.CAP_PROP_CONVERT_RGB, False)
    camR.set(cv2.CAP_PROP_CONVERT_RGB, False)
    fps2 = camL.get(cv2.CAP_PROP_FPS)

    while camL.grab() and camR.grab():
        time.sleep(0.025)
        timer = cv2.getTickCount()

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer) //3600

        _, frameL = camL.retrieve()
        _, frameR = camR.retrieve()

        print(fps2)


        ###############TEST################################3



        IR_L, BGGR_L = conversion(frameL)
        IR_R, BGGR_R = conversion(frameR)

        left_calibrated = undistorted(BGGR_L, data_left)
        right_calibrated = undistorted(BGGR_R, data_right)


        img = cv2.imread('gradient.png', 0)

        #adaptive trehs

        retL, thresh1L = cv2.threshold(BGGR_L, 160, 255, cv2.THRESH_TOZERO)# cv2.THRESH_BINARY)
        retR, thresh1R = cv2.threshold(BGGR_R, 160, 255,  cv2.THRESH_TOZERO)# cv2.THRESH_BINARY)

        retL2, thresh2L = cv2.threshold(IR_L, 160, 255, cv2.THRESH_TOZERO)  # cv2.THRESH_BINARY)
        retR2, thresh2R = cv2.threshold(IR_R, 160, 255, cv2.THRESH_TOZERO)  # cv2.THRESH_BINARY)

        #adaptive trehs
        th = cv2.adaptiveThreshold(IR_L, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 115, 1)
        #cv2.imshow("adaptive", th)
       # print(thresh1R.shape)


       # ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
       # ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
       # ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
       # ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
#
       # titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
       # images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
#
       # for i in xrange(6):
       #     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
       #     plt.title(titles[i])
       #     plt.xticks([]), plt.yticks([])
#
       # plt.show()




#        stackedFrames = np.concatenate((cv2.pyrUp(IR_L),cv2.pyrUp(IR_R)), axis = 1)
        stackedFrames1 = np.concatenate ((thresh1L,thresh1R), axis = 1)
        stackedFrames2 = np.concatenate ((thresh2L,thresh2R), axis = 1)


        stackedFrames = np.concatenate((IR_L,IR_R), axis=1)
        #stackedFrames_un = np.concatenate((BGGR_L, BGGR_R), axis=1)

        cv2.putText(BGGR_L, "FPS : " + str(int(fps)), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 25, 255), 2)
        cv2.imshow("fgmask", IR_L)


        #cv2.imshow("calibrated", stackedFrames1)
        #cv2.imshow("calibrated2", stackedFrames2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camL.release()
    cv2.destroyAllWindows()



main()
np.seterr(divide='ignore', invalid='ignore')



#d = dict(p1=1, p2=2)
#def f2(p1,p2):
#    print p1, p2
#f2(**d)#f2(**d)