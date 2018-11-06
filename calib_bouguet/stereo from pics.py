import numpy as np
import cv2
import subprocess
import  matplotlib.pyplot as plt
import matplotlib as mpl
from openpyxl import Workbook # Used for writing data into an Excel file


def coords_mouse_disp(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # print x,y,disp[y,x],filteredImg[y,x]
        average = 0
        for u in range(-1, 2):
            for v in range(-1, 2):
                average += disp[y + u, x + v]
        average = average / 9
        Distance = -593.97 * average ** (3) + 1506.8 * average ** (2) - 1373.1 * average + 522.06
        Distance = np.around(Distance * 0.01, decimals=2)
        print('Distance: ' + str(Distance) + ' m')


    #This section has to be uncommented if you want to take mesurements and store them in the excel
        ws.append([counterdist, average])
        print('Measure at '+str(counterdist)+' cm, the dispasrity is ' + str(average))
        if (counterdist <= 85):
            counterdist += 3
        elif(counterdist <= 120):
            counterdist += 5
        else:
            counterdist += 10
        print('Next distance to measure: '+str(counterdist)+'cm')


wb=Workbook()
ws=wb.active

def main():
    data = np.load('/home/minneyar/Desktop/stereocam2/calib_bouguet/Parameters/All_parameters.npy').item()  # 1

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
    imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/dir_left/left_RGB32.jpg" )  # downscale images for faster processing if you like    #1
    imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike 25.9 jutro/dir_left/left_RGB1.jpg" )  # downscale images for faster processing if you like    #1
    imgL = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike 25.9 jutro/dir_right/right_RGB1.jpg")
    hl, wl = imgL.shape[:2]  # both frames should be of same shape

    imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/dir_right/right_RGB32.jpg")
    imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike 25.9 jutro/dir_right/right_RGB1.jpg")
    imgR = cv2.imread("/home/minneyar/Desktop/stereocam2/calib_bouguet/slike 25.9 jutro/dir_left/left_RGB1.jpg" )  # downscale images for faster processing if you like    #1




    hr, wr = imgR.shape[:2]  # both frames should be of same shape

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

    window_size = 3
    min_disp = 2
    num_disp = 130 - min_disp
    left_matcher = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=window_size,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32,
                                   disp12MaxDiff=5,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2)

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
    disp = displ
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

    disp = ((disp.astype(np.float32) / 16) - min_disp) / num_disp  # Calculation allowing us to have 0 for the most distant object able to detect
    closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE,
                               kernel)  # Apply an morphological filter for closing little "black" holes in the picture(Remove noise)

    # Colors map
    dispc = (closing - closing.min()) * 255



    dispC = dispc.astype(np.uint8)  # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
    disp_Color = cv2.applyColorMap(dispC, cv2.COLORMAP_HOT)  # Change the Color of the Picture into an Ocean Color_Map
    filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_JET)


    #depth = baseline * focal / disparity
    #To obtain depth, you need to convert disparity using the following formula:
    ##depth = baseline * focal / disparity
    ##For KITTI the baseline is 0.54m and the focal ~721 pixels. The relative disparity outputted by the model has to be scaled by 1242 which is the original image size.
    ##The final formula is:
    ##depth = 0.54 * 721 / (1242 * disp)

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


    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('Centimeters')
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
