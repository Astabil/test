#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import glob
import random

np.seterr(divide='ignore', invalid='ignore')

#critera for termination
CHESSBOARD_SIZE = (8,6)
img_dir_left = "/home/ivan/PycharmProjects/StereoVision/see3cam/IVAN_stereo/left"
img_dir_right = "/home/ivan/PycharmProjects/StereoVision/see3cam/IVAN_stereo/right"

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
        img = cv2.imread(image).astype(np.uint8)
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
            cornersM = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(cornersM)

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

    #k = np.array(K)
    #d = np.array(D)
    print("RMS", rms)
    print("PRINTAMMMMMM" , rvecs, tvecs)

    print("Found " + str(img_number) + " valid images for calibration")
    print("DIM=" + str(img_size[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")", )
    print("D=np.array(" + str(D.tolist()) + ")\n", )

    #vrati K i d
    dimension = img_size[::-1]
    k = K.tolist()
    d = D.tolist()

    #print ("IZ FUNKCIJE" ,k , d)
    return  dimension , k , d

def main():
    objp_left, imgp_left, imgS_left, gray_left = calculate(img_dir_left)
    objp_right, imgp_right, imgS_right, gray_right = calculate(img_dir_right)

    print("IMG POINTS LEFT", np.asarray(imgp_left).shape)
    print("IMG POINTS RIGHT", np.asarray(imgp_right).shape)
    print("\nVALUES FOR LEFT CAMERA" )
    calib_left = calibrate(objp_left, imgp_left, imgS_left, gray_left)
    print("VALUES FOR RIGHT CAMERA")
    calib_right = calibrate(objp_right, imgp_right, imgS_right, gray_right)

    if calib_left != None:
        DIM, K, D = calib_left
        save_left = np.save('Left_calibrated_default', {'DIM': DIM, 'K': K, 'D': D})
    else:
        print("Error LEFT")
    if calib_right != None:
        DIM, K, D = calib_right
        save_right = np.save('Right_calibrated_default', {'DIM' : DIM, 'K': K, 'D':D })
    else:
        print('Error RIGHT')



main()