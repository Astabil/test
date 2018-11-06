import numpy as np
import cv2
from matplotlib import pyplot as plt
import subprocess


#imgl = cv2.imread("/home/minneyar/Desktop/stereocam2/SLIKE_BIG_IR/dir_right/RIGHT_UNC15.jpg" )  # downscale images for faster processing if you like
#    #imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/TAMNE_SLIJE/dir_left/LEFT_UNC22.jpg" )  # downscale images for faster processing if you like
#
##ret,imgl = cv2.threshold(imgL, 20, 25, cv2.THRESH_TOZERO)  # cv2.THRESH_BINARY)
#
#
#ret, thresh2 = cv2.threshold(imgl, 127, 255, cv2.THRESH_BINARY_INV)
#ret, thresh3 = cv2.threshold(imgl, 127, 255, cv2.THRESH_TRUNC)
#ret, thresh4 = cv2.threshold(imgl, 222, 255, cv2.THRESH_TOZERO)
#ret, thresh5 = cv2.threshold(imgl, 127, 255, cv2.THRESH_TOZERO_INV)
## titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
## images = [imgl, thresh2, thresh2, thresh3, thresh4, thresh5]
## for i in range(6):
##    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
##    plt.title(titles[i])
##    plt.xticks([]), plt.yticks([])
## plt.show()
#
#cv2.imshow("a", thresh4)
#cv2.waitKey()
#cv2.destroyAllWindows()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# default values for brightness and exposure:
# brightness        (int)    : min=0 max=40 step=1 default=10 value=40
# exposure_absolute (int)    : min=1 max=10000 step=1 default=156 value=150



#subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_absolute=40",shell=True)

try:
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_absolute=10", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c brightness=40", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c exposure_absolute=10", shell=True)
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
    data_left = np.load('/home/minneyar/Desktop/stereocam2/Parameters_old/Left_calibrated.npy').item()
    data_right = np.load('/home/minneyar/Desktop/stereocam2/Parameters_old/Right_calibrated.npy').item()

    camL = cv2.VideoCapture(0)
    camR = cv2.VideoCapture(1)

    L = make_720p(camL)
    R = make_720p(camR)

    camL.set(cv2.CAP_PROP_CONVERT_RGB, False)
    camR.set(cv2.CAP_PROP_CONVERT_RGB, False)

    while camL.grab() and camR.grab():

        _, frameL = camL.retrieve()
        _, frameR = camR.retrieve()

        ###############TEST################################3
        IR_L, BGGR_L = conversion(frameL)
        IR_R, BGGR_R = conversion(frameR)

        left_calibrated = undistorted(BGGR_L, data_left)
        right_calibrated = undistorted(BGGR_R, data_right)

        img = cv2.imread('gradient.png', 0)

        #adaptive trehs

        retL, thresh1L = cv2.threshold(BGGR_L, 160, 255, cv2.THRESH_TOZERO)# cv2.THRESH_BINARY)
        retR, thresh1R = cv2.threshold(BGGR_R, 160, 255,  cv2.THRESH_TOZERO)# cv2.THRESH_BINARY)

        retL2, thresh2L = cv2.threshold(IR_L, 160, 200, cv2.THRESH_TOZERO)  # cv2.THRESH_BINARY)
        retL2, thresh3L = cv2.threshold(IR_L, 200, 255, cv2.THRESH_TOZERO)  # cv2.THRESH_BINARY)

        retR2, thresh2R = cv2.threshold(IR_R, 160, 255, cv2.THRESH_TOZERO)  # cv2.THRESH_BINARY)

        #adaptive trehs
        th = cv2.adaptiveThreshold(IR_L, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 115, 1)
        cv2.imshow("adaptive", th)
       # print(thresh1R.shape)

        hsv = cv2.cvtColor(BGGR_L, cv2.COLOR_BGR2HSV)

        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])

        mask = cv2.inRange(IR_L, thresh2L, thresh3L)
        res = cv2.bitwise_and(BGGR_L, BGGR_L, mask=mask)

        kernel = np.ones((5, 5), np.uint8)

        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cv2.imshow('Original', IR_L)
        cv2.imshow('Mask', mask)
        cv2.imshow('Opening', opening)
        cv2.imshow('Closing', closing)

        stackedFrames = np.concatenate ((thresh1L,thresh1R), axis = 1)
        stackedFrames2 = np.concatenate ((thresh2L,thresh2R), axis = 1)
        stackedFrames = np.concatenate((IR_L,IR_R), axis=1)
        #stackedFrames_un = np.concatenate((BGGR_L, BGGR_R), axis=1)

        #cv2.imshow("calibrated", stackedFrames)
        cv2.imshow("calibrated2", stackedFrames2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camL.release()
    cv2.destroyAllWindows()

main()
np.seterr(divide='ignore', invalid='ignore')
