import cv2
import pyzed.camera as zcam
import pyzed.types as tp
import pyzed.core as core
import pyzed.defines as sl
import math
import numpy as np
import sys

def main():
    print("Running...")
    init = zcam.PyInitParameters()
    cam = zcam.PyZEDCamera()
    if not cam.is_opened():
        print("Opening ZED Camera...")
    status = cam.open(init)
    if status != tp.PyERROR_CODE.PySUCCESS:
        print(repr(status))
        exit()

    runtime = zcam.PyRuntimeParameters()
    mat = core.PyMat()

    key = ''
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == tp.PyERROR_CODE.PySUCCESS:
            cam.retrieve_image(mat, sl.PyVIEW.PyVIEW_LEFT)
            cv2.imshow("ZED", mat.get_data())
            key = cv2.waitKey(5)
        else:
            key = cv2.waitKey(5)
    cv2.destroyAllWindows()

    cam.close()



main()
