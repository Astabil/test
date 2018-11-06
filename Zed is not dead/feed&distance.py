import cv2
import pyzed.camera as zcam
import pyzed.types as tp
import pyzed.core as core
import pyzed.defines as sl
import math
import numpy as np
import sys
import imutils

def main():
    i = 0
    image = core.PyMat()
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
            #cam.retrieve_image(mat, sl.PyVIEW.PyVIEW_LEFT)

            cam.retrieve_image(mat, sl.PyVIEW.PyVIEW_NORMALS)
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
            key = cv2.waitKey(5)
        else:
            key = cv2.waitKey(5)
    cv2.destroyAllWindows()

    cam.close()



main()
