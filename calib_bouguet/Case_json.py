import numpy as np
import cv2
import  matplotlib.pyplot as plt
import matplotlib as mpl

def main():
    data = np.load('/home/minneyar/Desktop/stereocam2/calib_bouguet/Parameters/All_parameters.npy').item()  # 1
    #data = np.load('All_parameters_MACI.npy').item()  # 1
   # data = np.load('/home/minneyar/Desktop/stereocam2/calib_bouguet/Parameters/All_parameters_BRIGHT_left.npy').item()  # 1
    #data = np.load('/home/minneyar/Desktop/stereocam2/calib_bouguet/Parameters/All_parameters_BRIGHT_RIGHT.npy').item()  # 1
   # data = np.load('/home/minneyar/Desktop/stereocam2/calib_bouguet/Parameters/stereoParams_cleaned.npy').item()  # 1


    camera1_matrix = data['camera1_matrix']
    camera2_matrix = data['camera2_matrix']
    dist1_coeff4V = data['dist1_coeff4V']
    # dist1_coeff5V = data['dist1_coeff5V']
    dist2_coeff4V = data['dist2_coeff4V']
    # dist2_coeff5V = data['dist2_coeff5V']
    R = data['R']
    T = data['T']
    # print(camera1_matrix)
    # print(camera2_matrix)
    # print(R)
    # print(T)
    # print(dist1_coeff4V)
    # print(dist2_coeff5V)

    # dataK = np.load('KL_parameters.npy').item()
    # K1 = dataK['K1']
    # K2 = dataK['K2']
    # K3 = dataK['K3']
    #
    # D = (np.array([K1, K2, K3])).reshape(1,3)
    # print(D+
    # skaliranje slike baza cv2.pyrDown
    # imgL = cv2.pyrDown(cv2.imread("/home/minneyar/Desktop/stereocam2/dir_left/left_RGB7.jpg", ))  # downscale images for faster processing if you like
    imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/dir_left/left_RGB32.jpg" )  # downscale images for faster processing if you like    #1
    imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike 25.9 jutro/dir_left/left_RGB1.jpg" )  # downscale images for faster processing if you like    #1
    imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike 25.9 jutro/dir_right/right_RGB1.jpg")
    #imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike zadnji dan/s/dir_right/right_RGB1.jpg")
    #imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike zadnji dan/s2/dir_right/right_RGB1.jpg")
    #imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike zadnji dan/s3/dir_right/right_RGB1.jpg")
    imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike zadnji dan/s4/dir_right/right_RGB3.jpg")
    #imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike zadnji dan/s5!/dir_right/right_RGB1.jpg")
    #imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/SLIKE ZADNJI DAN IR & RGB/dir_right/right_RGB1.jpg")
    #imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/SLIKE ZADNJI DAN IR & RGB/dir_right/right_IR1.jpg")
    imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/SLIKE ZADNJI DAN IR & RGB2/dir_right/right_RGB4.jpg")
    imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/Slike - Treshold/dir_right/right_RGB17.jpg")
    imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike zadnji dan/Stolice Test 1/dir_right/right_RGB7.jpg")




    ###TRESHOLDAT SLIKE
    #ret, imgL = cv2.threshold(imgL, 120, 255, cv2.THRESH_TOZERO)  # cv2.THRESH_BINARY)
    #imgL = np.asarray(imgL)
    hl, wl = imgL.shape[:2]  # both frames should be of same shape

    imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/dir_right/right_RGB32.jpg")
    imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike 25.9 jutro/dir_right/right_RGB1.jpg")
    imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike 25.9 jutro/dir_left/left_RGB1.jpg" )  # downscale images for faster processing if you like    #1
    #imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike zadnji dan/s/dir_left/left_RGB1.jpg" )  # downscale images for faster processing if you like    #1
    #imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike zadnji dan/s2/dir_left/left_RGB1.jpg" )  # downscale images for faster processing if you like    #1
    #imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike zadnji dan/s3/dir_left/left_RGB1.jpg" )  # downscale images for faster processing if you like    #1
    imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike zadnji dan/s4/dir_left/left_RGB3.jpg" )  # downscale images for faster processing if you like    #1
    #imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike zadnji dan/s5!/dir_left/left_RGB1.jpg" )  # downscale images for faster processing if you like    #1
    #imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/SLIKE ZADNJI DAN IR & RGB/dir_left/left_RGB1.jpg" )  # downscale images for faster processing if you like    #1
    #imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/SLIKE ZADNJI DAN IR & RGB/dir_left/left_IR1.jpg" )  # downscale images for faster processing if you like    #1
    imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/SLIKE ZADNJI DAN IR & RGB2/dir_left/left_RGB4.jpg" )  # downscale images for faster processing if you like    #1
    imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/Slike - Treshold/dir_left/left_RGB17.jpg" )  # downscale images for faster processing if you like    #1
    imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike zadnji dan/Stolice Test 1/dir_left/left_RGB7.jpg" )  # downscale images for faster processing if you like    #1



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
                                                               None, cv2.CALIB_ZERO_DISPARITY, -1)  #
    map1L, map2L = cv2.initUndistortRectifyMap(camera1_matrix, dist1_coeff4V, R1, P1, (wl, hl),
                                               cv2.CV_16SC2)  # cv2.CV_16SC2 this format enables us the programme to work faster
    map1R, map2R = cv2.initUndistortRectifyMap(camera2_matrix, dist2_coeff4V, R2, P2, (wr, hr), cv2.CV_16SC2)

    undistorted_imgL = cv2.remap(imgL, map1L, map2L, interpolation=cv2.INTER_LANCZOS4)
    undistorted_imgR = cv2.remap(imgR, map1R, map2R, interpolation=cv2.INTER_LANCZOS4)

    imageL = cv2.remap(imgL, map1L, map2L, interpolation=cv2.INTER_LANCZOS4)
    imageR = cv2.remap(imgR, map1R, map2R, interpolation=cv2.INTER_LANCZOS4)

    stackedFrames_UNC = np.concatenate((cv2.pyrDown(imgL), cv2.pyrDown(imgR)), axis=1)
    stackedFrames_CAL = np.concatenate((cv2.pyrDown(imageL), cv2.pyrDown(imageR)), axis=1)
    stackedFrames_CAL = cv2.flip(stackedFrames_CAL, -1)
    cv2.imshow("unc ", stackedFrames_UNC)
    cv2.imshow("cal", stackedFrames_CAL)
    # cv2.imshow("undistortedL", cv2.pyrDown(undistorted_imgL))

    print(Q)

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
    print("results", Q[2][3], 1/Q[3][2])


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
    filteredImg = cv2.flip(filteredImg, -1)
    image = cv2.applyColorMap(filteredImg, cv2.COLORMAP_HOT)
    # m = cv2.pyrDown(image)

    cv2.imshow('Disparity Map_', image)
    # print(m.shape)
    # print(image.shape, image.dtype)
    save_cal = cv2.imwrite('/home/minneyar/Desktop/stereocam2/calib_bouguet/DUBINA STOLICE/stolica7.jpg', image)
    #save = cv2.imwrite('/home/minneyar/Desktop/stereocam2/calib_bouguet/TEST DUBINE FINALNO/TEST3_1.jpg', cv2.applyColorMap(filteredImg,  cv2.COLORMAP_HOT))

   #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   #blurred = cv2.GaussianBlur(image, (5, 5), 0)
   #cv2.imshow("Image", image)
   #thresh = cv2.adaptiveThreshold(blurred, 255,
   #                               cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 3)
   #cv2.imshow("Mean Thresh", thresh)

   #thresh = cv2.adaptiveThreshold(blurred, 255,
   #                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
   #cv2.imshow("Gau", thresh)



    #depth = baseline * focal / disparity
    #To obtain depth, you need to convert disparity using the following formula:
    ##depth = baseline * focal / disparity
    ##For KITTI the baseline is 0.54m and the focal ~721 pixels. The relative disparity outputted by the model has to be scaled by 1242 which is the original image size.
    ##The final formula is:
    ##depth = 0.54 * 721 / (1242 * disp)

    # slika bez podataka u matplotlibu
   # plt.subplot(2, 1, 1)
    #plt.imshow(image)
    #plt.title("SLika")
    #plt.xticks([])
    #plt.yticks([])
    #plt.show()

    # Make a figure and axes with dimensions as desired.
    fig = plt.figure(figsize=(7, 6))
    ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
    # ax2 = fig.add_axes([0.05, 0.475, 0.9, 0.15])
    ax3 = fig.add_axes([0.05, -0.15, 0.9, 0.95])

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    cmap = mpl.cm.cool
    cmap = 'Blues'
    norm = mpl.colors.Normalize(vmin=0, vmax=480)

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('Centimeters')

    # The second example illustrates the use of a ListedColormap, a
    # BoundaryNorm, and extended ends to show the "over" and "under"
    # value colors.
    cmap = mpl.colors.ListedColormap(['r', 'g', 'b', 'c'])
    cmap.set_over('0.25')
    cmap.set_under('0.75')
    plt.imshow(image)
    plt.title("Depth image")
    plt.xticks([])
    plt.yticks([])
    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()


main()
