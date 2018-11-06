import numpy as np
import cv2
import time
import os
import sys


def setup_tracker(ttype):
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[ttype]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()

    return tracker

#video = cv2.VideoCapture(os.path.join(IMAGES_FOLDER, 'moving_subject_scale.mp4'))
#video = cv2.VideoCapture("/home/minneyar/Desktop/Libri/Computer-Vision-Basics-with-Python-Keras-and-OpenCV-master/images/moving_subject_scale.mp4")
#DODANO
video = cv2.VideoCapture(0)

def f8(frame):
    return np.array(frame//4, dtype = np.uint8)

def conversion(frame):
    curr_frame = f8(frame)
    IR = curr_frame[1::2, 0::2]
    curr_frame[1::2, 0::2] = curr_frame[0::2, 1::2]
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BayerRG2BGR_EA)
    return IR, curr_frame

video.set(cv2.CAP_PROP_CONVERT_RGB, False)
######
# Read first frame


while True:
    time.sleep(0.025)
    timer = cv2.getTickCount()

    # Read a new frame
    success, frame = video.read()
    IR_L, BGGR_L = conversion(frame)
    frame = BGGR_L

    if not success:
        print("first frame not read")
        sys.exit()
        break
    if success:
        tracker = cv2.TrackerKCF_create()
        # Select roi for bbox
        bbox = cv2.selectROI(frame, False)
        cv2.destroyAllWindows()

    # Initialize tracker with first frame and bounding box
    tracking_success = tracker.init(frame, bbox)

    # Update tracker
    tracking_success, bbox = tracker.update(frame)

    # Draw bounding box
    if tracking_success:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

    cv2.imshow("frame", frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27: break # ESC pressed


cv2.destroyAllWindows()
video.release()