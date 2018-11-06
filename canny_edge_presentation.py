import sys
import cv2
import numpy as np


#def conversion(frame):
#    rows = frame.shape[0]
#    cols = frame.shape[1]
#    curr_frame = f8(frame)
#
#    bayer = np.copy(curr_frame)
#    #kreiraj IR frame upola manji od ulaznog zbog vidiljivosti"
#    IR = np.zeros([rows//2, cols//2], np.uint8)
#    #zamini svako IR komponentu s zelenom BGIRR to BGGR"
#    for x in range(0, rows, 2):
#        for y in range(0, cols, 2):
#            bayer[x+1, y] = curr_frame[x, y+1]
#            IR[x//2, y//2] = curr_frame[x+1, y]
#    BGGR =cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2BGR)
#    return IR, BGGR

def conversion(frame):
    curr_frame = f8(frame)
    IR = curr_frame[0::2, 1::2]
    curr_frame[1::2, 0::2] = curr_frame[0::2, 1::2]
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BayerRG2BGR)
    return IR, curr_frame

def f8(frame):
    #return cv2.convertScaleAbs(frame, 0.25)
    return np.array(frame//4, dtype = np.uint8)

def read_cam():

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        windowName = "Edge Detection"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName, 1280, 720)
        cv2.moveWindow(windowName, 0, 0)
        cv2.setWindowTitle(windowName, "InfraRed Stereo")

        cap.set(cv2.CAP_PROP_CONVERT_RGB, False)
        showWindow = 3  # Show all stages
        showHelp = True
        edgeThreshold = 40
        showFullScreen = False
        while True:
            if cv2.getWindowProperty(windowName, 0) < 0:
                break;
            ret_val, frame = cap.read();

            IR_L, BGGR_L = conversion(frame)
            hsv = cv2.cvtColor( BGGR_L, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(hsv, (7, 7), 1.5)
            edges = cv2.Canny(blur, 0, edgeThreshold)
            if showWindow == 3:
                frameRs = cv2.resize( BGGR_L, (640, 360))
                hsvRs = cv2.resize(hsv, (640, 360))
                vidBuf = np.concatenate((frameRs, cv2.cvtColor(hsvRs, cv2.COLOR_GRAY2BGR)), axis=1)
                blurRs = cv2.resize(blur, (640, 360))
                edgesRs = cv2.resize(edges, (640, 360))
                vidBuf1 = np.concatenate(
                    (cv2.cvtColor(blurRs, cv2.COLOR_GRAY2BGR), cv2.cvtColor(edgesRs, cv2.COLOR_GRAY2BGR)), axis=1)
                vidBuf = np.concatenate((vidBuf, vidBuf1), axis=0)
            if showWindow == 1:  # Show Camera Frame
                displayBuf =  BGGR_L
            elif showWindow == 2:  # Show Canny Edge Detection
                displayBuf = edges
            elif showWindow == 3:  # Show All Stages
                displayBuf = vidBuf

            cv2.imshow(windowName, displayBuf)
            key = cv2.waitKey(10)
            if key == 27:  # Check for ESC key
                cv2.destroyAllWindows()
                break;
            elif key == 49:  # 1 key, show frame
                cv2.setWindowTitle(windowName, "Camera Feed")
                showWindow = 1
            elif key == 50:  # 2 key, show Canny
                cv2.setWindowTitle(windowName, "Canny Edge Detection")
                showWindow = 2
            elif key == 51:  # 3 key, show Stages
                cv2.setWindowTitle(windowName, "Camera, Gray scale, Gaussian Blur, Canny Edge Detection")
                showWindow = 3
            elif key == 52:  # 4 key, toggle help
                showHelp = not showHelp
            elif key == 44:  # , lower canny edge threshold
                edgeThreshold = max(0, edgeThreshold - 1)
                print('Canny Edge Threshold Maximum: ', edgeThreshold)
            elif key == 46:  # , raise canny edge threshold
                edgeThreshold = edgeThreshold + 1
                print('Canny Edge Threshold Maximum: ', edgeThreshold)
            elif key == 74:  # Toggle fullscreen; This is the F3 key on this particular keyboard
                # Toggle full screen mode
                if showFullScreen == False:
                    cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                showFullScreen = not showFullScreen
    else:
        print ("camera open failed")

if __name__ == '__main__':
    read_cam()