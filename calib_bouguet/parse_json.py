import numpy as np
import json
from pprint import pprint
import cv2

with open("/home/minneyar/Desktop/stereocam2/calib_bouguet/StereoParams.json", 'r') as data_file:
    data = json.load(data_file)

#pprint(data["CameraParameters2"])


#distortionCoefficients is a vector or 4, 5, or 8 parameters:
# k1, k2, p1, p2 [, k3 [, k4, k5, k6]]

image_size = list(data["CameraParameters1"]["ImageSize"])
#image_size = list(data["WorldPoints"])

print(image_size)


"""Intrinsics:

    cameraMatrix1 = stereoParameters.CameraParameters1.IntrinsicMatrix'
    cameraMatrix2 = stereoParameters.CameraParameters2.IntrinsicMatrix'
    distCoeffs1 = [stereoParameters.CameraParameters1.RadialDistortion(1:2), stereoParameters.CameraParameters1.TangentialDistortion, stereoParameters.CameraParameters1.RadialDistortion(3)]
    distCoeffs2 = [stereoParameters.CameraParameters2.RadialDistortion(1:2), stereoParameters.CameraParameters2.TangentialDistortion, stereoParameters.CameraParameters2.RadialDistortion(3)]

Extrinsics:

    R = stereoParameters.RotationOfCamera2'
    T = stereoParameters.TranslationOfCamera2'
"""

"""PRETVORBE"""


"""Camera matrix for both cameras  K"""
"""cameraMatrix1 = stereoParams.CameraParameter1.intrinsicMatrix"""

camera_matrix1_mat = np.asarray(data["CameraParameters1"]["IntrinsicMatrix"])
camera_matrix1_cv = camera_matrix1_mat.T #transpose camera matrix because matlab/cv2
cm1 = np.copy(camera_matrix1_cv)   #copy camera matrix for later elimination of parameter skew[0][1] to zero because of cv2
cm1[0][1] = 0   # set skew to zero

print("mat", camera_matrix1_mat)
print("CV", camera_matrix1_cv)
print("NOVI M", cm1)
#print("CV_copy", m)

#print("CV_skew", camera_matrix1[0][1])

"""cameraMatrix2 = stereoParams.CameraParameter2.intrinsicMatrix"""
camera_matrix2_mat = np.asarray(data["CameraParameters2"]["IntrinsicMatrix"])#.reshape(3,3)
camera_matrix2_cv = camera_matrix2_mat.T #transpose matrix because of opencv
cm2 = np.copy(camera_matrix2_cv)
cm2[0][1] = 0


save_cameramatrix = np.save('Camera_matrices', {'camera1_matrix':cm1, 'camera2_matrix':cm2})
save = np.savez_compressed('outputFile', camera_matrix1 = cm1, camera_matrix2 = cm2)
calibration = np.load('outputFile.npz', allow_pickle=False)



"""dist coeficients elements"""

#Radijalni parametri K1, K2, K3, od prve kamere
radial_Dist1 = np.array(data["CameraParameters1"]["RadialDistortion"])
rd1_K1param = radial_Dist1[0]  # K1 prva kamera
rd1_K2param = radial_Dist1[1]  # K2 prva kamera
rd1_K3param = radial_Dist1[2] # K3 prva kamera

#radial_Dist2_K1 = np.array(data["CameraParameters2"]["RadialDistortion"])
radial_Dist2 = np.array(data["CameraParameters2"]["RadialDistortion"])
rd2_K1param = radial_Dist2[0]  # K1 prva kamera
rd2_K2param = radial_Dist2[1]  # K2 prva kamera
rd2_K3param = radial_Dist2[2] # K3 prva kamera


save_K = np.save("KL_parameters", {'K1':rd1_K1param, 'K2':rd1_K2param, 'K3': rd1_K3param})

"""PRINT distorzijski"""
#pprint(data["CameraParameters1"]["RadialDistortion"])
#
#print("#######################")
#print("K1",rd1_K1param)
#print("K2",rd1_K2param)
#print("K3", rd1_K3param)
#print("#######################")
#
#pprint(data["CameraParameters2"]["RadialDistortion"])
#
#print("#######################")
#print("K1", rd2_K1param)
#print("K2", rd2_K2param)
#print("K3", rd2_K3param)
#print("#######################")

"""Tangentinal distortion p1, p2"""

#tangencijalna distorzija prve kamere
tang_Dist1 = np.array(data["CameraParameters1"]["TangentialDistortion"])
td1_P1 = tang_Dist1[0]
td1_P2 = tang_Dist1[1]

#tangencijalna distorzija druge kamere
tang_Dist2 = np.array(data["CameraParameters2"]["TangentialDistortion"])
td2_P1 = tang_Dist2[0]
td2_P2 = tang_Dist2[1]

"""D1 s cetri i pet elemenata"""
D1_4 = (np.array([rd1_K1param, rd1_K2param, td1_P1, td1_P2])).reshape(4,1)                # četri elementa
D1_5 = (np.array([rd1_K1param, rd1_K2param, td1_P1, td1_P2, rd1_K3param])).reshape(5,1)   # pet elemenata fisheye
#print(D1_4)
#print(D1_5)


"""D2 s cetri i pet elemenata"""
D2_4 = (np.array([rd2_K1param, rd2_K2param, td2_P1, td2_P2])).reshape(4,1)                 # ovo je samo s cetri elementa
D2_5 = (np.array([rd2_K1param, rd2_K2param, td2_P1, td2_P2, rd2_K3param])).reshape(5,1)       # pet elemenata fisheye
#print("D2_4", D2_4.shape)

#print("D2_5", D2_5.shape)


#"""Print Tangencijalna"""
#pprint(data["CameraParameters1"]["TangentialDistortion"])
##print("#######################")
#print("P1",td1_P1)
#print("P2",td1_P2)
##print("#######################")
#
#pprint(data["CameraParameters2"]["TangentialDistortion"])
##print("#######################")
#print("P1",td2_P1)
#print("P2",td2_P2)
##print("#######################")




""" R = stereoParams.RotationOfCamera2 """
R = np.array(data["RotationOfCamera2"])
R = R.T                                                         #Transpose it
#print ("R", R.shape )
#print(R)

"""T = stereoParams.TranslationOfCamera2"""
T = np.asarray(data["TranslationOfCamera2"])
T = T.T                                                             #Transpose it
#print("T", T)

save_parameters = np.save("All_parameters", {'camera1_matrix':cm1, 'camera2_matrix':cm2, 'dist1_coeff4V': D1_4, 'dist1_coeff5V': D1_5,
                                             'dist2_coeff4V': D2_4, 'dist2_coeff5V': D2_5, 'R': R, 'T':T})