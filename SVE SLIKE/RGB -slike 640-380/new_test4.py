import numpy as np
import cv2
from sklearn.preprocessing import normalize
import subprocess


CHECKERBOARD = (9,6)

criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

# Arrays to store object points and image points from all images
objpoints= []   # 3d points in real world space
imgpointsR= []   # 2d points in image plane
imgpointsL= []

K_left = np.zeros((3, 3))
D_left = np.zeros((4, 1))

K_right = np.zeros((3, 3))
D_right = np.zeros((4, 1))

R = np.zeros((1, 1, 3), dtype=np.float64)
T = np.zeros((1, 1, 3), dtype=np.float64)

N_OK = len(imgpointsL)

objp = np.zeros( (CHECKERBOARD[0]*CHECKERBOARD[1], 1, 3) , np.float64)
objp[:,0, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = np.array([objp]*len(imgpointsL), dtype=np.float64)
imgpointsL = np.asarray(imgpointsL, dtype=np.float64)
imgpointsR = np.asarray(imgpointsR, dtype=np.float64)

objpoints = np.reshape(objpoints, (N_OK, 1, CHECKERBOARD[0]*CHECKERBOARD[1], 3))
imgpoints_left = np.reshape(imgpointsL, (N_OK, 1, CHECKERBOARD[0]*CHECKERBOARD[1], 2))
imgpoints_right = np.reshape(imgpointsR, (N_OK, 1, CHECKERBOARD[0]*CHECKERBOARD[1], 2))





# Start calibration from the camera
print('Starting calibration for the 2 cameras... ')
# Call all saved images
for i in range(1,27):   # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
    t= str(i)
    ChessImaR = cv2.imread('chessboard-R' + t + '.jpg', 0)  # Right side
    ChessImaL = cv2.imread('chessboard-L' + t + '.jpg', 0)  # Left side


    retR, cornersR = cv2.findChessboardCorners(ChessImaR,
                                               CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)  # Define the number of chees corners we are looking for
    retL, cornersL = cv2.findChessboardCorners(ChessImaL,
                                               CHECKERBOARD,  cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)  # Left side
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR,cornersR,(11,11),(-1,-1),criteria)
        cv2.cornerSubPix(ChessImaL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)




(rms, K_left, D_left, K_right, D_right, R, T) = \
        cv2.fisheye.stereoCalibrate(
            objpoints,
            imgpoints_left,
            imgpoints_right,
            K_left,
            D_left,
            K_right,
            D_right,
            ChessImaL[::-1],
            R,
            T,
            calib_flags
        )