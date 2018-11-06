import cv2
import pyzed.camera as zcam
import pyzed.types as tp
import pyzed.core as core
import pyzed.defines as sl
import math
import numpy as np
import sys
import imutils


bg = None

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (_, cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def main():
    # initialize weight for running average
    aWeight = 0.5

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590
    top, right, bottom, left = 50, 150, 250, 450 #me likes


    # initialize num of frames
    num_frames = 0

    #instance zeda
    i = 0
    depth = core.PyMat()
    point_cloud = core.PyMat()

    print("Running...")
    init_params = zcam.PyInitParameters()
    init_params.depth_mode = sl.PyDEPTH_MODE.PyDEPTH_MODE_PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.PyUNIT.PyUNIT_MILLIMETER  # Use milliliter units (for depth measurements)

    cam = zcam.PyZEDCamera()
    if not cam.is_opened():
        print("Opening ZED Camera...")
    status = cam.open(init_params)

    if status != tp.PyERROR_CODE.PySUCCESS:
        print(repr(status))
        exit()

    runtime = zcam.PyRuntimeParameters()
    runtime.sensing_mode = sl.PySENSING_MODE.PySENSING_MODE_STANDARD  # Use STANDARD sensing mode

    mat = core.PyMat()

    key = ''
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == tp.PyERROR_CODE.PySUCCESS:
            #cam.retrieve_image(mat, sl.PyVIEW.PyVIEW_LEFT)
           # cam.retrieve_image(mat, sl.PyVIEW.PyVIEW_DEPTH)
            cam.retrieve_image(mat, sl.PyVIEW.PyVIEW_NORMALS)
            cam.retrieve_image(mat, sl.PyVIEW.PyVIEW_LEFT_GRAY)

            cam.retrieve_measure(depth, sl.PyMEASURE.PyMEASURE_DEPTH)
            cam.retrieve_measure(point_cloud, sl.PyMEASURE.PyMEASURE_XYZRGBA)

            #euklinda distanca za mirit udaljenost objekta

            x = round(mat.get_width() / 2)
            y = round(mat.get_height() / 2)

            err, point_cloud_value = point_cloud.get_value(x, y)
            distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                 point_cloud_value[1] * point_cloud_value[1] +
                                 point_cloud_value[2] * point_cloud_value[2])

            if not np.isnan(distance) and not np.isinf(distance):
                distance = round(distance)
                print("Distance to Camera at ({0}, {1}): {2} mm\n".format(x, y, distance))
                # Increment the loop
                i = i + 1
            else:
                print("Can't estimate distance at this position, move the camera\n")
            sys.stdout.flush()
            #cam.retrieve_image(mat, sl.PyVIEW.PyVIEW_LEFT_UNRECTIFIED)
            cv2.imshow("ZED", mat.get_data())

            frame = mat.get_data()
            frame = imutils.resize(frame, width=700)
            frame = cv2.flip(frame, 1)
            clone = frame.copy()
            (height, width) = frame.shape[:2]

            print(height, width)
            roi = frame[top:bottom, right:left]
            gray = cv2.GaussianBlur(roi, (7, 7), 0)
            cv2.imshow("roi", roi)

            if num_frames < 30:
                run_avg(gray, aWeight)
            else:
                # segment the hand region
                hand = segment(gray)
                # check whether hand region is segmented
                if hand is not None:
                    # if yes, unpack the thresholded image and
                    # segmented region
                    (thresholded, segmented) = hand

                    # draw the segmented region and display the frame
                    cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                    cv2.imshow("Thesholded", thresholded)

                    # draw the segmented hand
            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
            # increment the number of frames
            num_frames += 1

            # display the frame with segmented hand
            cv2.imshow("Video Feed", clone)

            key = cv2.waitKey(5)
        else:
            key = cv2.waitKey(5)
    cv2.destroyAllWindows()

    cam.close()



main()
