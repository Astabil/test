import numpy as np
import cv2
import glob


def undistorted(frame, data):
    data_in = data
    img = frame
    K, D, DIM = data_in['K'], data_in['D'], data_in['DIM']
    K = np.array(K)
    D = np.array(D)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3),
                                                     K, DIM, cv2.CV_16SC2)#cv2.CV_16SC2
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_TRANSPARENT,  borderValue=29)
    return undistorted_img


data_left = np.load('Left_calibrated.npy').item()
data_right = np.load('Right_calibrated.npy').item()

imgl = cv2.cvtColor(cv2.imread("/home/ivan/PycharmProjects/StereoVision/see3cam/IVAN_stereo/dir_left/left_RGB1.jpg")
                    , cv2.COLOR_BGR2GRAY)
imgr = cv2.cvtColor(cv2.imread("/home/ivan/PycharmProjects/StereoVision/see3cam/IVAN_stereo/dir_right/right_RGB1.jpg")
                    , cv2.COLOR_BGR2GRAY)

l = undistorted(imgl, data_left)
r = undistorted(imgr, data_right)

cv2.imshow("slika1", undistorted(imgl, data_left))
cv2.imwrite("/home/ivan/PycharmProjects/StereoVision/see3cam/IVAN_stereo/undistorted_liva.jpg",l)
cv2.imwrite("/home/ivan/PycharmProjects/StereoVision/see3cam/IVAN_stereo/undistorted_desna.jpg",r)



cv2.waitKey(0)
cv2.destroyAllWindows()


#stereo = cv2.StereoSGBM_create(minDisparity=4, numDisparities=16, blockSize=23)
#disparity = stereo.compute(imgl,imgr)
## Normalize the image for representation
#min = disparity.min()
#max = disparity.max()
#disparity = np.uint8(255 * (disparity - min) / (max - min))
#
#stacked = np.concatenate((imgl,undistorted(imgl, data_left)), axis= 0)
#
#fig = plt.figure(figsize=(16,9))
#ax = plt.Axes(fig, [0., 0., 1., 1.])
#ax.set_axis_off()
#fig.add_axes(ax)
#plt.imshow(np.hstack((imgl, imgr, disparity)),'gray')
#plt.show()

#cv2.imshow("stacked", disparity)
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()