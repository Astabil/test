import numpy as np
import cv2
import pandas as pd
import  matplotlib.pyplot as plt
from matplotlib import cm
from pyntcloud import PyntCloud


def main():
    data = np.load('/home/minneyar/Desktop/stereocam2/calib_bouguet/Parameters/All_parameters.npy').item()  # 1
    #data = np.load('All_parameters_MACI.npy').item()  # 1
    data = np.load('/home/minneyar/Desktop/stereocam2/calib_bouguet/Parameters/All_parameters_BRIGHT_left.npy').item()  # 1
    data = np.load('/home/minneyar/Desktop/stereocam2/calib_bouguet/Parameters/All_parameters_BRIGHT_RIGHT.npy').item()  # 1
    data = np.load('/home/minneyar/Desktop/stereocam2/calib_bouguet/Parameters/stereoParams_cleaned.npy').item()  # 1


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
    imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/dir_right/right_RGB33.jpg" )  # downscale images for faster processing if you like    #1
    imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/Slije za usporedbu/dir_left/LEFT_UNC8.jpg") #MARIN USPOEREDBA
    # imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/SLIKE_BIG_IR/dir_right/RIGHT_UNC15.jpg" )  # downscale images for faster processing if you like
   # imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/TAMNE_SLIJE/dir_left/LEFT_UNC1.jpg")  # downscale images for faster processing if you like
   # imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/Slije za usporedbu/dir_left/LEFT_UNC8.jpg") #MARIN USPOEREDBA
    imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/NOVE SLIKE/dir_left/LEFT_UNC26.jpg")



    ###TRESHOLDAT SLIKE
    #ret, imgL = cv2.threshold(imgL, 120, 255, cv2.THRESH_TOZERO)  # cv2.THRESH_BINARY)
    #imgL = np.asarray(imgL)
    hl, wl = imgL.shape[:2]  # both frames should be of same shape

    imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/dir_left/left_RGB33.jpg")  # downscale images for faster processing if you like
    imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/Slije za usporedbu/dir_right/RIGHT_UNC8.jpg")
    #imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/SLIKE_BIG_IR/dir_left/LEFT_UNC15.jpg")  # downscale images for faster processing if you like
   # imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/TAMNE_SLIJE/dir_right/RIGHT_UNC1.jpg")  # downscale images for faster processing if you like
    imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/NOVE SLIKE/dir_right/RIGHT_UNC26.jpg")

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
   # cv2.imshow("unc ", stackedFrames_UNC)
   # cv2.imshow("cal", stackedFrames_CAL)
    # cv2.imshow("undistortedL", cv2.pyrDown(undistorted_imgL))

    # imageL = cv2.pyrDown(undistorted_imgL)
    # imageR = cv2.pyrDown(undistorted_imgR)

    # cv2.imshow("undistortedR", undistorted_imgR)

    #
    #   """ SGBM Parameters ------"""
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    #left_matcher = cv2.StereoSGBM_create(
    #    minDisparity=0,
    #    numDisparities=160,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    #    blockSize=5,
    #    P1=8 * 3 * window_size ** 2,
    #    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    #    P2=32 * 3 * window_size ** 2,
    #    disp12MaxDiff=1,
    #    uniquenessRatio=15,
    #    speckleWindowSize=0,
    #    speckleRange=2,
    #    preFilterCap=63,
    #    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    #)
#
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
   # print("results", Q[2][3], 1/Q[3][2])
   # print(Q)
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



    filteredImg = wls_filter.filter(displ, imageL, None, dispr)  # important to put "imgL" here!!!
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    filteredImg = cv2.flip(filteredImg, -1)
    ### do odi je napisano learn tech with us

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


    print('generating 3d point cloud...', )
    h, w = imgL.shape[:2]
    f = 0.8 * w  # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                    [0, 0, 0, -f],  # so that y-axis looks up
                    [0, 0, 1, 0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    print('%s saved' % 'out.ply')

    cv.imshow('left', imgL)
    cv.imshow('disparity', (disp - min_disp) / num_disp)
    cv.waitKey()


cv.destroyAllWindows()



   #fig ,ax = plt.subplots()
   #cax = ax.imshow(cv2.flip(disp,-1), interpolation='nearest', cmap=cm.coolwarm)
   #cax = ax.imshow(cv2.applyColorMap(filteredImg, cv2.COLORMAP_HOT), interpolation='nearest', cmap=cm.Blues)

   #ax.set_title('test')
   #cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
   #cbar.ax.set_yticklabels(['<-1', '0', '>1'])
   #plt.show()


    #point cloud
    colourImg = imgL
    indicesArray = np.moveaxis(np.indices((hl, wl)), 0, 2)
    imageArray = np.dstack((indicesArray, colourImg)).reshape((-1, 5))
    df = pd.DataFrame(imageArray, columns=["x", "y", "red", "green", "blue"])
    depthImg = filteredImg
    depthArray = np.array(depthImg[::-2])
    df.insert(loc=2, column='z', value=depthArray)
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(float)
    df[['red', 'green', 'blue']] = df[['red', 'green', 'blue']].astype(np.uint)
    df['z'] = df['z'] * 0.5
    cloud = PyntCloud(df)
    cloud.plot()


    #cv2.imshow("as", filt_Color)
   #plt.imshow(image)
   #plt.colorbar()
   #plt.show()



    cv2.waitKey()
    cv2.destroyAllWindows()



main()

