#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import glob
import random



"""Kombinacija fisheye kalibracije i standardne stereo kalibracije"""





np.seterr(divide='ignore', invalid='ignore')

#critera for termination
CHESSBOARD_SIZE = (9,6)
img_dir_left = "/home/ivan/PycharmProjects/StereoVision/see3cam/IVAN_stereo/dir_left"
img_dir_right = "/home/ivan/PycharmProjects/StereoVision/see3cam/IVAN_stereo/dir_right"

#critera for termination
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((1, CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

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
        cv2.imshow(imgDir, img)
        cv2.waitKey(1)

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

    (leftRectification, rightRectification, leftProjection, rightProjection,
     dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
        KL, DL,
        KR, DR,
        imgS_left, rotationMatrix, translationVector,
        None, None, None, None, None,
        cv2.CALIB_ZERO_DISPARITY, 0.25)

    print("NOVO", leftRectification, rightRectification, leftProjection, rightProjection,
     dispartityToDepthMap, leftROI, rightROI)

    leftMapX, leftMapY = cv2.initUndistortRectifyMap(
        KL, DL, leftRectification,
        leftProjection, imgS_left, cv2.CV_32FC1)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        KR, DR, rightRectification,
        rightProjection, imgS_right, cv2.CV_32FC1)

    return rotationMatrix, translationVector

    #flag = 0
    #flag = cv2.fisheye.CALIB_FIX_INTRINSIC
    #flag |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    ## flag |= cv2.fisheye.CALIB_CHECK_COND
#
    #flag |= cv2.fisheye.CALIB_FIX_SKEW
#

    #(_,_,_,_,_,R,T) = cv2.fisheye.stereoCalibrate(objp_left,
    #                                              imgp_left,
    #                                              imgp_right,
    #                                              KL,DL,
    #                                              KR,DR,
    #                                              imgS_left,None, None,calib_flags, criteria)


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
    objp_left, imgp_left, imgS_left, gray_left = calculate(img_dir_left)
    objp_right, imgp_right, imgS_right, gray_right = calculate(img_dir_right)

    RM, TM = Scalibrate()

    print("Stereo RM " , RM)
    print("Stereo TM " , TM)

main()