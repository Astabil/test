import numpy as np
import cv2
import subprocess
import  matplotlib.pyplot as plt
import matplotlib as mpl




def main():
    data = np.load('/home/minneyar/Desktop/stereocam2/calib_bouguet/Parameters/All_parameters.npy').item()  # 1
    #data = np.load('All_parameters_MACI.npy').item()  # 1
    #data = np.load('/home/minneyar/Desktop/stereocam2/calib_bouguet/Parameters/All_parameters_BRIGHT_left.npy').item()  # 1
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
    # print(D)

    # skaliranje slike baza cv2.pyrDown
    # imgL = cv2.pyrDown(cv2.imread("/home/minneyar/Desktop/stereocam2/dir_left/left_RGB7.jpg", ))  # downscale images for faster processing if you like
    imgL = cv2.imread(
        "/home/minneyar/Desktop/stereocam2/dir_left/left_RGB22.jpg")  # downscale images for faster processing if you like    #1

    ###TRESHOLDAT SLIKE
    # ret, imgL = cv2.threshold(imgL, 120, 255, cv2.THRESH_TOZERO)  # cv2.THRESH_BINARY)
    # imgL = np.asarray(imgL)
    hl, wl = imgL.shape[:2]  # both frames should be of same shape

    imgR = cv2.imread(
        "/home/minneyar/Desktop/stereocam2/dir_right/right_RGB22.jpg")  # downscale images for faster processing if you like

    ###TRESHOLDAT SLIKE
    #ret, imgR = cv2.threshold(imgR, 120, 255, cv2.THRESH_TOZERO)  # cv2.THRESH_BINARY)
    #imgR = np.asarray(imgR)

    # imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/TAMNE_SLIJE/dir_right/RIGHT_UNC2.jpg") # downscale images for faster processing if you like
    hr, wr = imgR.shape[:2]  # both frames should be of same shape
    #  cv2.imshow("undistorted", imgL)
    f = 0.29*wr
    print ("wr" , wr)
    print("f", f)
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

    # imageL = cv2.pyrDown(undistorted_imgL)
    # imageR = cv2.pyrDown(undistorted_imgR)

    # cv2.imshow("undistortedR", undistorted_imgR)


    window_size = 3
    min_disp = 2
    num_disp = 130 - min_disp
    left_matcher  = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=window_size,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32,
                                   disp12MaxDiff=5,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2)

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    #izracun focal lenght i baseline q[2][3] , 1/q[3][2]
    print("results", Q[2][3], 1/Q[3][2])
    print(Q)
   # print("udaljenost",  Q[2][3]*1/Q[3][2]/left_matcher )


    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    print('computing disparity...')
    disp = left_matcher.compute(imageL, imageR).astype(np.float32) / 16
    displ = disp
    dispr = right_matcher.compute(imageR, imageL).astype(np.float32) / 16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    b = 6.5
    print ("D", b*f/disp )

    #distance = cv2.reprojectImageTo3D(disp, Q, None, None, None)
    #print("nova distanca", distance)

    ### 72 x 60

    filteredImg = wls_filter.filter(displ, imageL, None, dispr)  # important to put "imgL" here!!!
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    filteredImg = cv2.flip(filteredImg, -1)

    image = cv2.applyColorMap(filteredImg, cv2.COLORMAP_HOT)
    # m = cv2.pyrDown(image)
    #cv2.imshow("moje", image)
    #dodano
    disp = ((disp.astype(np.float32) / 16) - min_disp) / num_disp
    closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE,kernel)  # Apply an morphological filter for closing little "black" holes in the picture(Remove noise)
    dispc = (closing - closing.min()) * 255
    dispC = dispc.astype(np.uint8)  # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
    disp_Color = cv2.applyColorMap(dispC, cv2.COLORMAP_HOT)  # Change the Color of the Picture into an Ocean Color_Map
    filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_HOT)

    #distance = cv2.reprojectImageTo3D(disp, Q, None, None, None)
   # print("nova distanca", distance)


    #print("udaljenost",  Q[2][3]*1/Q[3][2]/disp )
    #cv2.imshow("disp", dispC)
    #cv2.imshow('Disparity Map_', filt_Color)

    #fig ,ax = plt.subplots()
    #cax = ax.imshow(cv2.flip(disp,-1), interpolation='nearest', cmap=cm.coolwarm)
    #cax = ax.imshow(cv2.applyColorMap(filteredImg, cv2.COLORMAP_HOT), interpolation='nearest', cmap=cm.Blues)
#
    #ax.set_title('Depth')
  ###cbar = fig.colorbar(cax, ticks=range)
    #cbar = fig.colorbar(cax, ticks=range(60, 480, 20), label = 'depth in cm')
#
    #cbar.ax.set_yticklabels(['0', '480'])
    #plt.show()




    cv2.imshow("as", cv2.applyColorMap(filteredImg, cv2.COLORMAP_HOT))
    #plt.imshow(image)
    #plt.colorbar()
    #plt.show()

    #slika bez podataka u matplotlibu
   # plt.imshow(image)
   # plt.title("SLika")
   # plt.xticks([])
   # plt.yticks([])
   # plt.show()



    plt.subplot(2,1,1)
    plt.imshow(image, clim=(60, 480))
    ###colorbar
    fig = plt.figure(figsize=(6, 2))
    plt.subplot(2,1,2)
    ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
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
    cb1.set_label('Distance unit [cm]')
    plt.show()






    #ValueError: Colormap COLORMAP_JET is not recognized. Possible values are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, winter, winter_r

    #depth = baseline * focal / disparity
    #To obtain depth, you need to convert disparity using the following formula:
    ##depth = baseline * focal / disparity
    ##For KITTI the baseline is 0.54m and the focal ~721 pixels. The relative disparity outputted by the model has to be scaled by 1242 which is the original image size.
    ##The final formula is:
    ##depth = 0.54 * 721 / (1242 * disp)

    #my_image1 = image
    #my_image2 = np.sqrt(my_image1.T) + 3
    #plt.subplot(1, 2, 1)
    #plt.imshow(my_image1)
    #plt.subplot(1, 2, 2)
    #plt.imshow(fig, vmin=0, vmax=480, cmap='Blues', aspect='auto')
    #plt.colorbar()
#
    #plt.show()
#
    #cv2.waitKey()
    #cv2.destroyAllWindows()



main()

"""
def coords_mouse_disp(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #print x,y,disp[y,x],filteredImg[y,x]
        average=0
        for u in range (-1,2):
            for v in range (-1,2):
                average += disp[y+u,x+v]
        average=average/9
        Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
        Distance= np.around(Distance*0.01,decimals=2)
    print('Distance: '+ str(Distance)+' m')"""