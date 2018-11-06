import numpy as np
import cv2
from sklearn.preprocessing import normalize
import subprocess

"""Standardni stereo-pipeline bez fisheye funkcija"""



try:
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_absolute=70", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c brightness=30", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c exposure_absolute=70", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c brightness=30", shell=True)
except:
    print("Error occured")


def make_480p(cap):
    cap.set(3, 640)
    cap.set(4, 480)

def f8(frame):
    return np.array(frame//4, dtype = np.uint8)

kernel= np.ones((3,3),np.uint8)

criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all images
objpoints= []   # 3d points in real world space
imgpointsR= []   # 2d points in image plane
imgpointsL= []

# Start calibration from the camera
print('Starting calibration for the 2 cameras... ')
# Call all saved images
for i in range(1,27):   # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
    t= str(i)
    ChessImaR = cv2.imread('chessboard-R' + t + '.jpg', 0)  # Right side
    ChessImaL = cv2.imread('chessboard-L' + t + '.jpg', 0)  # Left side
    retR, cornersR = cv2.findChessboardCorners(ChessImaR,
                                               (9, 6), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)  # Define the number of chees corners we are looking for
    retL, cornersL = cv2.findChessboardCorners(ChessImaL,
                                               (9, 6),  cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)  # Left side
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR,cornersR,(11,11),(-1,-1),criteria)
        cv2.cornerSubPix(ChessImaL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

# Determine the new values for different parameters
#   Right Side
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        ChessImaR.shape[::-1],None,None)
hR,wR= ChessImaR.shape[:2]
OmtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,
                                                   (wR,hR),1,(wR,hR))

#   Left Side
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        ChessImaL.shape[::-1],None,None)
hL,wL= ChessImaL.shape[:2]
OmtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

print("RetR", retR)
print("RetL", retL)
print('Cameras Ready to use')


calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW


retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,
                                                          imgpointsL,
                                                          imgpointsR,
                                                          mtxL,
                                                          distL,
                                                          mtxR,
                                                          distR,
                                                          ChessImaR.shape[::-1],
                                                          criteria_stereo,
calib_flags)

print("retS", retS)

rectify_scale = 0  # if 0 image croped, if 1 image nor croped
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                  ChessImaR.shape[::-1], R, T,
                                                  rectify_scale,
                                                  (0, 0),1)  # last paramater is alpha, if 0= croped, if 1= not croped
# initUndistortRectifyMap function
Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                              ChessImaR.shape[::-1],
                                              cv2.CV_16SC2)  # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                               ChessImaR.shape[::-1], cv2.CV_16SC2)

REMAP_INTERPOLATION = cv2.INTER_LINEAR


DEPTH_VISUALIZATION_SCALE = 2048


CAMERA_WIDTH = 680
CAMERA_HEIGHT = 480

camL = cv2.VideoCapture(0)
camR = cv2.VideoCapture(1)

make_480p(camL)
make_480p(camR)

camL.set(cv2.CAP_PROP_CONVERT_RGB, False)
camR.set(cv2.CAP_PROP_CONVERT_RGB, False)

camL.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camL.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
camR.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camR.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)


window_size = 15
min_disp = 16
num_disp = 96 - min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=16,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=150,
                                   speckleRange=32
)    

# morphology settings
kernel = np.ones((12, 12), np.uint8)

counter = 450



while camL.grab()  and camR.grab():

    _, frameL = camL.retrieve()
    _, frameR = camR.retrieve()

    Left_nice = cv2.remap(f8(frameL), Left_Stereo_Map[0], Left_Stereo_Map[1], REMAP_INTERPOLATION)                           # Rectify the image using the kalibration parameters founds during the initialisation
    Right_nice = cv2.remap(f8(frameR), Right_Stereo_Map[0], Right_Stereo_Map[1],REMAP_INTERPOLATION)

    disparity = stereo.compute(Left_nice, Right_nice).astype(np.float32) / 16.0
    disparity = (disparity - min_disp) / num_disp

    cv2.imshow("L", Left_nice)
    cv2.imshow("r", Right_nice)
    #cv2.imshow('depth',disparity)
    #cv2.imshow('Filtered Color Depth', filt_Color)

    # Mouse click

    # End the Programme
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

camR.release()
camL.release()
cv2.destroyAllWindows()