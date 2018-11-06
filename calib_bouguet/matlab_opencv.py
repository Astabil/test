import cv2
import numpy as np
import json
from pprint import *

def conversion(frame):
    curr_frame = f8(frame)
    IR = curr_frame[1::2, 0::2]
    curr_frame[1::2, 0::2] = curr_frame[0::2, 1::2]
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BayerRG2BGR_EA)
    return IR, curr_frame

def f8(frame): return cv2.convertScaleAbs(frame, 0.25)


#segment za otvaranje matlab json filea
with open("/home/minneyar/Desktop/stereocam2/calib_bouguet/StereoParams.json", 'r') as data_file:
    data = json.load(data_file)
pprint(data["CameraParameters1"])

