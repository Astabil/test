import cv2
import numpy as np
import glob


"""koristene slike za kalibraciju s OpenCV sitea, puca kod kod stereo kalibracije"""

CHECKERBOARD = (8,6)

img_dir_left = "/home/ivan/PycharmProjects/StereoVision/see3cam/IVAN_stereo/left"
img_dir_right = "/home/ivan/PycharmProjects/StereoVision/see3cam/IVAN_stereo/right"

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_CHECK_COND

K_left = np.zeros((3, 3))
D_left = np.zeros((4, 1))

K_right = np.zeros((3, 3))
D_right = np.zeros((4, 1))

R = np.zeros((1, 1, 3), dtype=np.float64)
T = np.zeros((1, 1, 3), dtype=np.float64)


objp = np.zeros( (CHECKERBOARD[0]*CHECKERBOARD[1], 1, 3) , np.float64)
objp[:,0, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)




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
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

        #if ret true append coordinates of verticies in objpoints, and append coordinates of checkboard corners to imgpoint
        if ret:
            objpoints.append(objp)
            cornersM = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(cornersM)

        cv2.drawChessboardCorners(img, CHECKERBOARD, cornersM, ret)
        cv2.imshow(imgDir, img)
        cv2.waitKey(1)

        #cv2.destroyAllWindows(imgDir)
    return objpoints, imgpoints, img_size, gray

objp_left, imgp_left, imgS_left, gray_left = calculate(img_dir_left)
objp_right, imgp_right, imgS_right, gray_right = calculate(img_dir_right)


obj = np.array(imgp_left)
print(obj.shape)

obj2 = np.array(imgp_right)
print(obj2.shape)

N_OK = len(imgp_left)

objpoints = np.array([objp]*len(imgp_left), dtype=np.float64)
imgpoints_left = np.asarray(imgp_left, dtype=np.float64)
imgpoints_right = np.asarray(imgp_right, dtype=np.float64)

objpoints = np.reshape(objpoints, (N_OK, 1, CHECKERBOARD[0]*CHECKERBOARD[1], 3))
imgpoints_left = np.reshape(imgpoints_left, (N_OK, 1, CHECKERBOARD[0]*CHECKERBOARD[1], 2))
imgpoints_right = np.reshape(imgpoints_right, (N_OK, 1, CHECKERBOARD[0]*CHECKERBOARD[1], 2))

(rms, K_left, D_left, K_right, D_right, R, T) = \
        cv2.fisheye.stereoCalibrate(
            objpoints,
            imgpoints_left,
            imgpoints_right,
            K_left,
            D_left,
            K_right,
            D_right,
            imgS_left,
            R,
            T,
            calib_flags
        )