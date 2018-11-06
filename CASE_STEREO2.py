#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import subprocess
import glob
from sklearn.preprocessing import normalize

np.seterr(divide='ignore', invalid='ignore')

try:
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_absolute=70", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c brightness=30", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c exposure_absolute=70", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c brightness=30", shell=True)
except:
    print("Error occured")

CHESSBOARD_SIZE = (9,6)
img_dir_left = "/home/ivan/PycharmProjects/StereoVision/see3cam/IVAN_stereo/dir_left"
img_dir_right = "/home/ivan/PycharmProjects/StereoVision/see3cam/IVAN_stereo/dir_right"

#critera for termination
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((1, CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

def conversion(frame):
    curr_frame = f8(frame)
    IR = curr_frame[1::2, 0::2]
    curr_frame[1::2, 0::2] = curr_frame[0::2, 1::2]
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BayerRG2BGR)
    return IR, curr_frame

def f8(frame): return np.array(frame//4, dtype = np.uint8)

def calculate(imgDir):
    imgPath = glob.glob(imgDir+'/*jpg')
    img_size = None # useful for testing image size
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for image in sorted(imgPath):
        img = cv2.imread(image)
        if img_size == None:
            img_size = img.shape[:2]
        else:
            assert img_size == img.shape[:2], "All images must share the same size."

        #img = cv2.imread(image)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        #if ret true append coordinates of verticies in objpoints, and append coordinates of checkboard corners to imgpoint
        if ret:
            objpoints.append(objp)
            cornersM = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria) #increase precision of corners coordinates
            imgpoints.append(corners)

        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, cornersM, ret)
       # cv2.imshow(imgDir, img)
       # cv2.waitKey(1)

        #cv2.destroyAllWindows(imgDir)
    return objpoints, imgpoints, img_size, gray

def calibrate(objpoints, imgpoints, img_size, gray):
    #objpoints, imgpoints, img_size, gray = calibrate(img_dir_left)
    img_number = len(objpoints) #number of the pictures
    #print("image points", m_ok)
    #floating point camera matrix default [3x3] => default values ==  ([[fx,0,cx], [0,fy,cy], [0,0,1]])
    K = np.zeros((3, 3))
    #output vector of distortion coefficients (k1,k2,k3,k4)
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(img_number)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(img_number)]

    print("rvecs", rvecs)
    rms, _, _, _, _ = cv2.fisheye.calibrate(objpoints,imgpoints,gray.shape[::-1],K,D,rvecs,tvecs,calib_flags,
         (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))


    print("RMS", rms)
    print("PRINTAMMMMMM" , rvecs, tvecs)

    print("Found " + str(img_number) + " valid images for calibration")
    print("DIM=" + str(img_size[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")\n")

    #vrati K i d
    dimension = img_size[::-1]
    k = K.tolist()
    d = D.tolist()

    #print ("IZ FUNKCIJE" ,k , d)
    return  dimension , k , d

def Scalibrate():

    objp_left, imgp_left, imgS_left, gray_left = calculate(img_dir_left)
    objp_right, imgp_right, imgS_right, gray_right = calculate(img_dir_right)

    DIM_left, KL, DL = calibrate(objp_left, imgp_left, imgS_left, gray_left)
    DIM_right, KR, DR = calibrate(objp_right, imgp_right, imgS_right, gray_right)

    KL = np.array(KL)
    DL = np.array(DL)
    KR = np.array(KR)
    DR = np.array(DR)

    (_, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
        objp_left, imgp_left, imgp_right,
        KL, DL,
        KR, DR,
        imgS_left, None, None, None, None,
        cv2.CALIB_FIX_INTRINSIC, criteria)

    rectify_scale = 0

    (leftRectification, rightRectification, leftProjection, rightProjection,
     dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
        KL, DL,
        KR, DR,
        imgS_left, rotationMatrix, translationVector,
        None, None, None, None, None,
        cv2.CALIB_ZERO_DISPARITY,  rectify_scale,(0,0))

    print("NOVO", leftRectification, rightRectification, leftProjection, rightProjection,
     dispartityToDepthMap, leftROI, rightROI)

    leftMapX, leftMapY = cv2.initUndistortRectifyMap(
        KL, DL, leftRectification,
        leftProjection, imgS_left, cv2.CV_32FC1)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        KR, DR, rightRectification,
        rightProjection, imgS_right, cv2.CV_32FC1)

    return rotationMatrix, translationVector , leftMapX, leftMapY , rightMapX, rightMapY, leftROI, rightROI


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


def stereo():
    REMAP_INTERPOLATION = cv2.INTER_LINEAR

    DEPTH_VISUALIZATION_SCALE = 2048
    rotationMatrix, translationVector, leftMapX, leftMapY, rightMapX, rightMapY, leftROI, rightROI = Scalibrate()

    stereoMatcher = cv2.StereoBM_create()
    stereoMatcher.setMinDisparity(2)
    stereoMatcher.setNumDisparities(144)
    stereoMatcher.setBlockSize(31)
    stereoMatcher.setROI1(leftROI)
    stereoMatcher.setROI2(rightROI)
    stereoMatcher.setSpeckleRange(12)
    stereoMatcher.setSpeckleWindowSize(25)



    camL = cv2.VideoCapture(0)
    camR = cv2.VideoCapture(1)

    camL.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camL.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    camR.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camR.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

    camL.set(cv2.CAP_PROP_CONVERT_RGB, False)
    camR.set(cv2.CAP_PROP_CONVERT_RGB, False)

    data_left = np.load('Left_calibrated.npy').item()
    data_right = np.load('Right_calibrated.npy').item()

    while camL.grab() == True and camR.grab() == True:

        _, frameL = camL.retrieve()
        _, frameR = camR.retrieve()

       # IR_L, BGGR_L = conversion(frameL)
       # IR_R, BGGR_R = conversion(frameR)

        fl = f8(frameL)
        fr = f8(frameR)

        left_calibrated = undistorted(fl, data_left)
        right_calibrated = undistorted(fr, data_right)

        undistorted_img_L = cv2.remap(fl, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT,
                                    borderValue=29)
        undistorted_img_R = cv2.remap(fr, rightMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT,
                                    borderValue=29)

        fixedLeft = left_calibrated
        fixedRight = right_calibrated
#
       # grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
      #  grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
        depth = stereoMatcher.compute(fixedLeft, fixedRight).astype(np.float32) / 16.0
      #  depth = stereoMatcher.compute(fixedRight, fixedLeft).astype(np.float32) / 16.0

       #grayLeft = cv2.cvtColor(undistorted_img_L, cv2.COLOR_BGR2GRAY)
       #grayRight = cv2.cvtColor(undistorted_img_R, cv2.COLOR_BGR2GRAY)
       #depth = stereoMatcher.compute(grayLeft, grayRight).astype(np.float32) / 16.0



       # cv2.imshow('left', fixedLeft)
       # cv2.imshow('right', fixedRight)
        Window_name = "Stereo"
        cv2.namedWindow(Window_name)
        cv2.imshow(Window_name, depth / DEPTH_VISUALIZATION_SCALE)
        # left_calibrated = undistorted(BGGR_L, data_left)
       # right_calibrated = undistorted(BGGR_R, data_right)


        #stackedFrames = np.concatenate((left_calibrated,right_calibrated), axis = 1)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        elif cv2.waitKey(1) & 0xFF == ord('s'):
            tmp = camL
            camL = camR
            camR = tmp

    camL.release()
    cv2.destroyAllWindows()


def main():
    stereo()

main()




