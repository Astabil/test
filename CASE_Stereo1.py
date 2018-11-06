import numpy as np
import cv2
import glob
import pprint
import re
from sklearn.preprocessing import normalize


def conversion(frame):
    curr_frame = f8(frame)
    IR = curr_frame[1::2, 0::2]
    curr_frame[1::2, 0::2] = curr_frame[0::2, 1::2]
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BayerRG2BGR)
    return IR, curr_frame


def f8(frame):
    #return cv2.convertScaleAbs(frame, 0.25)
    return np.array(frame//4, dtype = np.uint8)

data_left = np.load('Left_calibrated.npy').item()
cm1, dc1, DIM1 = data_left['K'], data_left['D'], data_left['DIM']
data_right = np.load('Right_calibrated.npy').item()
cm2, dc2, DIM2 = data_right['K'], data_right['D'], data_right['DIM']

cm1 = np.array(cm1)
dc1 = np.array(dc1)
cm2 = np.array(cm2)
dc2 = np.array(dc2)


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.

img_dir_left = "/home/ivan/PycharmProjects/StereoVision/see3cam/IVAN_stereo/dir_left"
img_dir_right = "/home/ivan/PycharmProjects/StereoVision/see3cam/IVAN_stereo/dir_right"

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
        ret, corners = cv2.findChessboardCorners(gray, (9,6), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

        #if ret true append coordinates of verticies in objpoints, and append coordinates of checkboard corners to imgpoint
        if ret:
            objpoints.append(objp)
            cornersM = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(cornersM)

       # cv2.drawChessboardCorners(img, (9,6), cornersM, ret)
       # cv2.imshow(imgDir, img)
        cv2.waitKey(1)

    return objpoints, imgpoints

Objpoints, ImgpointsL = calculate(img_dir_left)
ObjpointsR, ImgpointsR = calculate(img_dir_right)


# retVal = returned value, cm1 = Camera Matrix 1
# dc1 = Distortion Coefficients matrix 1, cm2 = Camera Matrix 2
# dc2 = Distortion Coefficients matrix 2
# r = rotation matrix, t = translation vector
# e = essential matrix, f = fundamental matrix
retVal, cm1, dc1, cm2, dc2, r, t, e, f = cv2.stereoCalibrate(Objpoints, ImgpointsL, ImgpointsR, cm1, dc1, cm2, dc2, (640,480), None, None, cv2.CALIB_FIX_INTRINSIC, criteria)

#print (r, t, e, f)
#
#print ("Calibration done.")

def main():
    kernel = np.ones((3, 3), np.uint8)
    rectify_scale= 0 # if 0 image croped, if 1 image nor croped
    RL, RR, PL, PR, Q, _, _= cv2.stereoRectify(cm1, dc1, cm2, dc2,
                                                     (640,480), r, t,
    rectify_scale,(0,0)) # last paramater is alpha, if 0= croped, if 1= not croped
    Left_Stereo_Map= cv2.initUndistortRectifyMap(cm1, dc1, RL, PL,
                                                 (640,480), cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
    Right_Stereo_Map= cv2.initUndistortRectifyMap(cm2, dc1, RR, PR,
    (640,480), cv2.CV_16SC2)

    window_size = 5
    min_disp = 0
    num_disp = 160 - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=window_size,
                                   uniquenessRatio=15,
                                   speckleWindowSize=0,
                                   speckleRange=2,
                                   disp12MaxDiff=5,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2)

    # Used for the filtered image
    stereoR = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time

    # WLS FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)


    # Call the two cameras
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



        Left_nice = cv2.remap(BGGR_L, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT,
                              0)
        Right_nice = cv2.remap(BGGR_R, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)


        grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

        disp = stereo.compute(grayL, grayR)  # .astype(np.float32)/ 16
        dispL = disp
       # dispR = cv2.resize(disp, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
        dispR = stereoR.compute(grayR, grayL)
        dispL = np.int16(dispL)
        dispR = np.int16(dispR)
      #

            # Using the WLS filter
        filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
        # cv2.imshow('Disparity Map', filteredImg)
        disp = ((disp.astype(
            np.float32) / 16) - min_disp) / num_disp
        ##    # Resize the image for faster executions

        # Filtering the Results with a closing filter
        closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE,
                                   kernel)  # Apply an morphological filter for closing little "black" holes in the picture(Remove noise)

        # Colors map
        dispc = (closing - closing.min()) * 255
        dispC = dispc.astype(
            np.uint8)  # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
        disp_Color = cv2.applyColorMap(dispC, cv2.COLORMAP_OCEAN)  # Change the Color of the Picture into an Ocean Color_Map
        filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)

        # Show the result for the Depth_image
        # cv2.imshow('Disparity', disp)
        # cv2.imshow('Closing',closing)
        # cv2.imshow('Color Depth',disp_Color)
        cv2.imshow('Filtered Color Depth', filt_Color)

        # Mouse click

        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

# Release the Cameras
    camR.release()
    camL.release()
    cv2.destroyAllWindows()


main()
np.seterr(divide='ignore', invalid='ignore')