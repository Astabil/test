import numpy as np
import cv2
import subprocess
import os, errno

try:
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_absolute=10", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c brightness=60", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c exposure_absolute=10", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c brightness=60", shell=True)
except:
    print("Error occured")

def conversion(frame):
    curr_frame = f8(frame)
    IR = curr_frame[1::2, 0::2]
    curr_frame[1::2, 0::2] = curr_frame[0::2, 1::2]
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BayerRG2BGR_EA)
    return IR, curr_frame

def f8(frame): return np.array(frame//4, dtype = np.uint8)

def make_720p(cap):
    cap.set(3, 1280)
    cap.set(4, 720)

def take_pictures():

    dir_left = "/home/minneyar/Desktop/stereocam2/Slije za usporedbu/dir_left"
    dir_right = "/home/minneyar/Desktop/stereocam2/Slije za usporedbu/dir_right"    #if not os.path.exists(directory):
    #    os.makedirs(directory)
    #else:
    #    pass

    try:
        os.makedirs(dir_left)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    try:
        os.makedirs(dir_right)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    camL = cv2.VideoCapture(0)
    camR = cv2.VideoCapture(1)
    camL.set(cv2.CAP_PROP_CONVERT_RGB, False)
    camR.set(cv2.CAP_PROP_CONVERT_RGB, False)

    make_720p(camL)
    make_720p(camR)

    frameID = 1
    while camL.grab() and camR.grab():

        retL, frameL = camL.retrieve()
        retR, frameR = camR.retrieve()

        IR_L, RGB_L = conversion(frameL)
        IR_R, RGB_R = conversion(frameR)

        # stackedFrames = np.concatenate((RGB_L,RGB_R), axis = 0)
        # stackedFrames = np.concatenate((RGB_L, RGB_R), axis=1)
        # cv2.imshow('Capture', stackedFrames)
        cv2.imshow("FRAME LEFT", RGB_L)
        cv2.imshow("FRAME RIGHT", RGB_R)

        key = cv2.waitKey(40) & 0xFF

        if retL and retR:
            if key == 32:  # Press Space to save img
                print("Saving images")
                img_l = dir_left + "/LEFT_UNC{}.jpg".format(frameID)
                img_r = dir_right + "/RIGHT_UNC{}.jpg".format(frameID)
                # cv2.imwrite(img_l, frameL)  #orginal
                # cv2.imwrite(img_r, frameR)  #orginal
                cv2.imwrite(img_l, RGB_L)
                cv2.imwrite(img_r, RGB_R)

                frameID += 1
                if frameID > 34:
                    break

        if key == ord('q'):
            break

        elif key == ord('s'):
            tmp = camL
            camL = camR
            camR = tmp

    camL.release()
    camR.release()
    cv2.destroyAllWindows()

take_pictures()


def stereo():

    data = np.load('All_parameters.npy').item()
    camera1_matrix = data['camera1_matrix']
    camera2_matrix = data['camera2_matrix']
    dist1_coeff4V = data['dist1_coeff4V']
    dist1_coeff5V = data['dist1_coeff5V']
    dist2_coeff4V = data['dist2_coeff4V']
    dist2_coeff5V = data['dist2_coeff5V']
    R = data['R']
    T = data['T']
    #print(camera1_matrix)
    #print(camera2_matrix)
    #print(R)
    #print(T)
    #print(dist1_coeff4V)
    #print(dist2_coeff5V)

    dataK = np.load('KL_parameters.npy').item()
    K1 = dataK['K1']
    K2 = dataK['K2']
    K3 = dataK['K3']

    D = (np.array([K1, K2, K3])).reshape(1,3)
    print(D)

    #skaliranje slike baza cv2.pyrDown
    #imgL = cv2.pyrDown(cv2.imread("/home/minneyar/Desktop/stereocam2/dir_left/left_RGB7.jpg", ))  # downscale images for faster processing if you like
    imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/dir_left/left_RGB7.jpg" )  # downscale images for faster processing if you like
    cv2.imshow("uncalibrated", imgL)

    hl, wl = imgL.shape[:2] # both frames should be of same shape
    imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/dir_right/right_RGB7.jpg") # downscale images for faster processing if you like
    hr, wr = imgR.shape[:2] # both frames should be of same shape
  #  cv2.imshow("undistorted", imgL)
    print(imgL.shape)



    img_size = imgR.shape
    print("Size", imgR.shape)
    kernel= np.ones((3,3),np.uint8)
    rectify_scale= 0 # if 0 image croped, if 1 image nor croped

    (R1, R2, P1, P2, Q, leftROI, rightROI) = cv2.stereoRectify(camera1_matrix, dist1_coeff5V, camera2_matrix, dist2_coeff5V, (wl, hl), R, T, None, None, None, None, None, cv2.CALIB_ZERO_DISPARITY, 0)#
    map1L, map2L = cv2.initUndistortRectifyMap(camera1_matrix, dist1_coeff5V, R1, P1,  (wl, hl), cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
    map1R, map2R = cv2.initUndistortRectifyMap(camera2_matrix, dist2_coeff5V, R2, P2,(wr, hr), cv2.CV_16SC2)

    undistorted_imgL = cv2.remap(imgL, map1L, map2L, interpolation=cv2.INTER_LANCZOS4)
    undistorted_imgR = cv2.remap(imgR, map1R, map2R, interpolation=cv2.INTER_LINEAR)

    cv2.imshow("nulta", undistorted_imgL)
    imageL = cv2.remap(imgL, map1L, map2L, interpolation = cv2.INTER_LANCZOS4)
    imageR = cv2.remap(imgR, map1R, map2R, interpolation = cv2.INTER_LANCZOS4)
    cv2.imshow("undistortedL", cv2.pyrDown(undistorted_imgL))

    #imageL = cv2.pyrDown(undistorted_imgL)
    #imageR = cv2.pyrDown(undistorted_imgR)

    #cv2.imshow("undistortedR", undistorted_imgR)



#******** TEST*************************#
    #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera1_matrix, dist1_coeff5V, (wl, hl), 1, (wl, hl))
    #mapx, mapy = cv2.initUndistortRectifyMap(camera1_matrix, dist1_coeff5V, R1,P1, newcameramtx, (wl,hl), cv2.CV_16SC2)
    #image = cv2.remap(imgL, mapx, mapy, cv2.INTER_LINEAR)
#
    #x, y, w, h = roi
    #image = image[y:y + h, x:x + w]

    #cv2.imshow("undistorted", image)

#
 #   """ SGBM Parameters ------"""
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    print('computing disparity...')
    displ = left_matcher.compute(imageL, imageR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imageR, imageL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imageL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    image = cv2.applyColorMap(filteredImg, cv2.COLORMAP_AUTUMN)
   # m = cv2.pyrDown(image)
    cv2.imshow('Disparity Map_', image)
    #print(m.shape)
   # print(image.shape, image.dtype)
   # save = cv2.imwrite('/home/minneyar/Desktop/stereocam2/matTOcv2.jpg', cv2.applyColorMap(filteredImg,  cv2.COLORMAP_HOT))


    cv2.waitKey()
    cv2.destroyAllWindows()

