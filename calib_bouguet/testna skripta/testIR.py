
# load required libraries
import math
import numpy as np
import cv2
import subprocess

try:
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_absolute=20", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video0 -c brightness=10", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c exposure_absolute=20", shell=True)
    subprocess.check_call("v4l2-ctl -d /dev/video1 -c brightness=10", shell=True)
except:
    print("Error occured")

def main():


    # open up camera on port 0
    camL = cv2.VideoCapture(0)
    camR = cv2.VideoCapture(1)

    # CU40 has the following possible resolution settings
    # 672 x 380
    # 1280 x 720
    # 1920 x 1080
    # 2688 x 1520

    rows = 1520
    cols = 2688

    #set dimensions
    camL.set(cv2.CAP_PROP_FRAME_HEIGHT, rows)
    camL.set(cv2.CAP_PROP_FRAME_WIDTH, cols)
    camR.set(cv2.CAP_PROP_FRAME_HEIGHT, rows)
    camR.set(cv2.CAP_PROP_FRAME_WIDTH, cols)


    # turn off automatic RGB conversion
    # this sets capture to 16bit
    camR.set(cv2.CAP_PROP_CONVERT_RGB, False)
    camL.set(cv2.CAP_PROP_CONVERT_RGB, False)

    while camL.grab() and camR.grab():

        _, frameL = camL.retrieve()
        _, frameR = camR.retrieve()


        bf8 = np.array(frameL//4, dtype = np.uint8)

        # uncomment this line to write out scaled (10 bit to 8 bit) raw imag

        # create a copy of the bayer array to replace IR with nearest G
        bRGGB = np.copy(bf8)

        # IR array is 1/4 of the pixels
        # create 8-bit array of 0s to fill

        cols, rows = frameR.shape[:2]
        IR = np.zeros([rows//2, cols//2], np.uint8)

        # copy out IR pixels
        IR = bf8[1::2, 0::2]
        # copy over IR pixels with nearest G pixel
        bf8[1::2, 0::2] = bf8[0::2, 1::2]

        # convert Bayer RGGB into BGR image [rows, cols, dim = 3]
        # cv2.COLOR_BayerRG2BGR bayer demosaicing
        # cv2.COLOR_BayerRG2BGR_EA edge aware demosaicing
        BGRim = cv2.cvtColor(bRGGB, cv2.COLOR_BayerRG2BGR_EA)

        cv2.imshow("IR_data", BGRim)

        # write out RGB image
        cv2.imshow("IR_data_a", IR)

        # write out IR image

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        camL.release()
        camR.release()
        cv2.destroyAllWindows()

main()