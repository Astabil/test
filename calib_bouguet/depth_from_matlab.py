import numpy as np
import cv2
import subprocess

try:
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_absolute=10", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c brightness=20", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c exposure_absolute=10", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c brightness=20", shell=True)
except:
    print("Error occured")


def conversion(frame):
    curr_frame = f8(frame)
    IR = curr_frame[1::2, 0::2]
    curr_frame[1::2, 0::2] = curr_frame[0::2, 1::2]
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BayerRG2BGR_EA)
    return IR, curr_frame


def f8(frame): return cv2.convertScaleAbs(frame, 0.25)


def make_720p(cap):
    cap.set(3, 1280)
    cap.set(4, 720)


def make_480p(cap):
    cap.set(3, 640)
    cap.set(4, 480)


def main():
    data = np.load('/home/minneyar/Desktop/stereocam2/calib_bouguet/Parameters/All_parameters_NEW.npy').item()
    camera1_matrix = data['camera1_matrix']
    camera2_matrix = data['camera2_matrix']
    dist1_coeff4V = data['dist1_coeff4V']
    dist2_coeff4V = data['dist2_coeff4V']
    R = data['R']
    T = data['T']

    imgL = cv2.imread(
        "/home/minneyar/Desktop/stereocam2/dir_right/right_RGB12.jpg")  # downscale images for faster processing if you like
    hl, wl = imgL.shape[:2]  # both frames should be of same shape

    imgR = cv2.imread(
        "/home/minneyar/Desktop/stereocam2/dir_left/left_RGB12.jpg")  # downscale images for faster processing if you like
    hr, wr = imgR.shape[:2]  # both frames should be of same shape

    #cv2.imshow("as", imgR)

    img_size = imgR.shape
    print("Size", imgR.shape)
    kernel = np.ones((3, 3), np.uint8)
    rectify_scale = 0  # if 0 image croped, if 1 image nor croped

   # (R1, R2, P1, P2, Q, leftROI, rightROI) = cv2.stereoRectify(camera1_matrix, dist1_coeff4V, camera2_matrix,
    #                                                           dist2_coeff4V, (wr, hr), R, T,  rectify_scale,(0,0))
    (R1, R2, P1, P2, Q, _, _) = cv2.stereoRectify(camera1_matrix, dist1_coeff4V, camera2_matrix,
                                                               dist2_coeff4V, (wr, hr), R, T, rectify_scale, (0, 0))  #
    #
    rectify_scale = 0
    map1L, map2L = cv2.initUndistortRectifyMap(camera1_matrix, dist1_coeff4V, R1, P1, (wl, hl), rectify_scale, (0,0), cv2.CV_16SC2)
                                               #cv2.CV_8U)  # cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
    map1R, map2R = cv2.initUndistortRectifyMap(camera2_matrix, dist2_coeff4V, R2, P1, (wr, hr), cv2.CV_16SC2)
                                              # cv2.CV_8U)  # cv2.CV_16SC2)

    undistorted_imgR = cv2.remap(imgL, map1L, map2L, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT,  borderValue=29)
    undistorted_imgL = cv2.remap(imgR, map1R, map2R, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT,  borderValue=29)


    cv2.imshow("L", undistorted_imgL)
    cv2.imshow("R", undistorted_imgR)

    # Create StereoSGBM and prepare all parameters
    window_size = 6
    min_disp = 2
    num_disp = 130 - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=window_size,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32,
                                   disp12MaxDiff=5,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2)

    # Used for the filtered image
    stereoR = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time

    # WLS FILTER Parameters
    lmbda = 80000
    sigma = 1.8
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    camL = cv2.VideoCapture(0)
    camR = cv2.VideoCapture(1)

    camL.set(cv2.CAP_PROP_CONVERT_RGB, False)
    camR.set(cv2.CAP_PROP_CONVERT_RGB, False)

    while camL.grab() and camR.grab():
        # Start Reading Camera images
        _, frameL = camL.retrieve()
        _, frameR = camR.retrieve()


        #cv2.imshow("asas", np.uint8(frameL))
        # frameL = frameL.shape[:2]
        # frameR = frameR.shape[:2]


        # Rectify the images on rotation and alignement
        Left_nice = cv2.remap(np.uint8(frameR), map1R, map2R, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        Right_nice = cv2.remap(np.uint8(frameL), map1L, map2L, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        # Left_nice = cv2.remap(np.uint8(frameL), map1L, map2L, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        # Right_nice =cv2.remap(np.uint8(frameR), map1R, map2R, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        # Left_nice = frameL.shape[:2]
        # Right_nice = frameR.shape[:2]

        # Left_nice =  cv2.remap(f8(frameL), map1L, map2L, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        # Right_nice = cv2.remap(f8(frameR), map1L, map2L, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        # Rectify the image using the kalibration parameters founds during the initialisation
        # Rectify the image using the kalibration parameters founds during the initialisation


        cv2.imshow("left ", Left_nice)
        cv2.imshow("right ",Right_nice)

        # Convert from color(BGR) to gray
        # grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
        # grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

        # Compute the 2 images for the Depth_image
        disp = stereo.compute(Left_nice, Right_nice)  # .astype(np.float32)/ 16
        dispL = disp
        dispR = stereoR.compute(Right_nice, Left_nice)
        dispL = np.int16(dispL)
        dispR = np.int16(dispR)

        # Using the WLS filter
        filteredImg = wls_filter.filter(dispL, Left_nice, None, dispR)
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
        # cv2.imshow('Disparity Map', filteredImg)
        disp = ((disp.astype(
            np.float32) / 16) - min_disp) / num_disp  # Calculation allowing us to have 0 for the most distant object able to detect

        ##    # Resize the image for faster executions
        ##    dispR= cv2.resize(disp,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)

        # Filtering the Results with a closing filter
        closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE,
                                   kernel)  # Apply an morphological filter for closing little "black" holes in the picture(Remove noise)

        # Colors map
        dispc = (closing - closing.min()) * 255
        dispC = dispc.astype(
            np.uint8)  # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
        disp_Color = cv2.applyColorMap(dispC,
                                       cv2.COLORMAP_OCEAN)  # Change the Color of the Picture into an Ocean Color_Map
        filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)

        # Show the result for the Depth_image
        #  cv2.imshow('Disparity', disp)
        # cv2.imshow('Closing',closing)
        # cv2.imshow('Color Depth',disp_Color)

        # IR_L, BGGR_L = conversion(frameL)
        # IR_R, BGGR_R = conversion(frameR)

        # cv2.imshow('Filtered Color Depth', filt_Color)

        #cv2.imshow('Filtered Color Depth', filt_Color)
        # cv2.imshow("nesto", IR_L )

        # Mouse click

        # End the Programme
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

        elif cv2.waitKey(1) & 0xFF == ord('s'):
            tmp = camL
            camL = camR
            camR = tmp

    # Save excel
    ##wb.save("data4.xlsx")

    # Release the Cameras
    camR.release()
    camL.release()
    cv2.destroyAllWindows()


main()
