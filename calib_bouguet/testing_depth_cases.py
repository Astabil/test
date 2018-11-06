import numpy as np
import cv2
import subprocess

dir_case1 = "/home/minneyar/Desktop/stereocam2/calib_bouguet/CASE_DEPTH/CASE1_CLOUDY/"
dir_case2 = "/home/minneyar/Desktop/stereocam2/calib_bouguet/CASE_DEPTH/CASE_BRIGHT/"
dir_case3 = "/home/minneyar/Desktop/stereocam2/calib_bouguet/CASE_DEPTH/CASE_DARK/"
dir_case3_a = "/home/minneyar/Desktop/stereocam2/calib_bouguet/CASE_DEPTH/CASE_DARK/use_case/"
dir_case4 = "/home/minneyar/Desktop/stereocam2/calib_bouguet/CASE_DEPTH/CASE_ZED/case8/"

def save_img(dir, img, ID):
    IMG = dir + "{}.jpg".format(ID)
    cv2.imwrite(IMG, img)

def main():
    data = np.load('All_parameters_NEW.npy').item()  # 1
   # data = np.load('All_parameters_MACI.npy').item()  # 1
   # data = np.load('All_parameters_BRIGHT_left.npy').item()  # 1
    data = np.load('All_parameters_BRIGHT_RIGHT.npy').item()  # 1

    camera1_matrix = data['camera1_matrix']
    camera2_matrix = data['camera2_matrix']
    dist1_coeff4V = data['dist1_coeff4V']
    # dist1_coeff5V = data['dist1_coeff5V']
    dist2_coeff4V = data['dist2_coeff4V']
    # dist2_coeff5V = data['dist2_coeff5V']
    R = data['R']
    T = data['T']


    # skaliranje slike baza cv2.pyrDown
    # imgL = cv2.pyrDown(cv2.imread("/home/minneyar/Desktop/stereocam2/dir_left/left_RGB7.jpg", ))  # downscale images for faster processing if you like
    #imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/dir_right/right_RGB33.jpg" )  # downscale images for faster processing if you like    #1
   # imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/SLIKE_BIG_IR/dir_left/LEFT_UNC22.jpg" )  # downscale images for faster processing if you like
    imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/TAMNE_SLIJE/dir_right/RIGHT_UNC1.jpg")  # downscale images for faster processing if you like
    imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/Slije za usporedbu/dir_right/LEFT_UNC8.jpg")

    ###TRESHOLDAT SLIKE
    #ret, imgL = cv2.threshold(imgL, 120, 255, cv2.THRESH_TOZERO)  # cv2.THRESH_BINARY)
    #imgL = np.asarray(imgL)
    hl, wl = imgL.shape[:2]  # both frames should be of same shape

   # imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/dir_left/left_RGB33.jpg")  # downscale images for faster processing if you like
   # imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/SLIKE_BIG_IR/dir_right/RIGHT_UNC22.jpg")  # downscale images for faster processing if you like
    imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/TAMNE_SLIJE/dir_left/LEFT_UNC6.jpg")


    imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/Slije za usporedbu/dir_right/RIGHT_UNC8.jpg")# downscale images for faster processing if you like
    ###TRESHOLDAT SLIKE
    #ret, imgR = cv2.threshold(imgR, 120, 255, cv2.THRESH_TOZERO)  # cv2.THRESH_BINARY)
    #imgR = np.asarray(imgR)

    # imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/TAMNE_SLIJE/dir_right/RIGHT_UNC2.jpg") # downscale images for faster processing if you like
    hr, wr = imgR.shape[:2]  # both frames should be of same shape
    #  cv2.imshow("undistorted", imgL)

    img_size = imgR.shape
    print("Size", imgR.shape)
    kernel = np.ones((3, 3), np.uint8)
    rectify_scale = 0  # if 0 image croped, if 1 image nor croped

    (R1, R2, P1, P2, Q, leftROI, rightROI) = cv2.stereoRectify(camera1_matrix, dist1_coeff4V, camera2_matrix,
                                                               dist2_coeff4V, (wl, hl), R, T, None, None, None, None,
                                                               None, cv2.CALIB_ZERO_DISPARITY, 0)  #
    map1L, map2L = cv2.initUndistortRectifyMap(camera1_matrix, dist1_coeff4V, R1, P1, (wl, hl),
                                               cv2.CV_16SC2)  # cv2.CV_16SC2 this format enables us the programme to work faster
    map1R, map2R = cv2.initUndistortRectifyMap(camera2_matrix, dist2_coeff4V, R2, P2, (wr, hr), cv2.CV_16SC2)

    undistorted_imgL = cv2.remap(imgL, map1L, map2L, interpolation=cv2.INTER_LANCZOS4)
    undistorted_imgR = cv2.remap(imgR, map1R, map2R, interpolation=cv2.INTER_LANCZOS4)

    imageL = cv2.remap(imgL, map1L, map2L, interpolation=cv2.INTER_LANCZOS4)
    imageR = cv2.remap(imgR, map1R, map2R, interpolation=cv2.INTER_LANCZOS4)

    stackedFrames_UNC = np.concatenate((cv2.pyrDown(imgL), cv2.pyrDown(imgR)), axis=1)
    stackedFrames_CAL = np.concatenate((cv2.pyrDown(imageL), cv2.pyrDown(imageR)), axis=1)

    cv2.imshow("unc ", stackedFrames_UNC)
    cv2.imshow("cal", stackedFrames_CAL)

    ID_unc = "unc"
    ID_cal = "cal"
    ID_stereo = "depth"
    save_img(dir_case4, stackedFrames_UNC, ID_unc )
    save_img(dir_case4, stackedFrames_CAL, ID_cal )



    # cv2.imshow("undistortedL", cv2.pyrDown(undistorted_imgL))

    # imageL = cv2.pyrDown(undistorted_imgL)
    # imageR = cv2.pyrDown(undistorted_imgR)

    # cv2.imshow("undistortedR", undistorted_imgR)

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
    displ = left_matcher.compute(imageL, imageR).astype(np.float32) / 16
    dispr = right_matcher.compute(imageR, imageL).astype(np.float32) / 16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imageL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    image = cv2.applyColorMap(filteredImg, cv2.COLORMAP_RAINBOW)
    # m = cv2.pyrDown(image)
    cv2.imshow('Disparity Map_', image)
    save_img(dir_case4, image, ID_stereo)

    # print(m.shape)
    # print(image.shape, image.dtype)
    # save = cv2.imwrite('/home/minneyar/Desktop/stereocam2/matTOcv2.jpg', cv2.applyColorMap(filteredImg,  cv2.COLORMAP_HOT))

   #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   #blurred = cv2.GaussianBlur(image, (5, 5), 0)
   #cv2.imshow("Image", image)
   #thresh = cv2.adaptiveThreshold(blurred, 255,
   #                               cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 3)
   #cv2.imshow("Mean Thresh", thresh)

   #thresh = cv2.adaptiveThreshold(blurred, 255,
   #                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
   #cv2.imshow("Gau", thresh)






    cv2.waitKey()
    cv2.destroyAllWindows()


main()
