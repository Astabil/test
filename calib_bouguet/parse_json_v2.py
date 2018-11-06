import numpy as np
import json
from pprint import pprint
import cv2

with open("/home/minneyar/Desktop/stereocam2/calib_bouguet/stereoParams_cleaned/stereoParams_cleaned.json", 'r') as data_file:
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

#""cameraMatrix2 = stereoParams.CameraParameter2.intrinsicMatrix"""
camera_matrix2_mat = np.asarray(data["CameraParameters2"]["IntrinsicMatrix"])#.reshape(3,3)
camera_matrix2_cv = camera_matrix2_mat.T #transpose matrix because of opencv

#ave_cameramatrix = np.save('Camera_matrices', {'camera1_matrix':cm1, 'camera2_matrix':cm2})
#ave = np.savez_compressed('outputFile', camera_matrix1 = cm1, camera_matrix2 = cm2)
#alibration = np.load('outputFile.npz', allow_pickle=False)



#""dist coeficients elements"""

##Radijalni parametri K1, K2, K3, od prve kamere
radial_Dist1 = np.array(data["CameraParameters1"]["RadialDistortion"])
rd1_K1param = radial_Dist1[0]  # K1 prva kamera
rd1_K2param = radial_Dist1[1]  # K2 prva kamera

radial_Dist2 = np.array(data["CameraParameters2"]["RadialDistortion"])
rd2_K1param = radial_Dist2[0]  # K1 prva kamera
rd2_K2param = radial_Dist2[1]  # K2 prva kamera


#save_K = np.save("KL_parameters", {'K1':rd1_K1param, 'K2':rd1_K2param, 'K3': rd1_K3param})

#""PRINT distorzijski"""
#pprint(data["CameraParameters1"]["RadialDistortion"])
#
#print("#######################")
print("K1",rd1_K1param)
print("K2",rd1_K2param)
#print("#######################")
#
#pprint(data["CameraParameters2"]["RadialDistortion"])

#""Tangentinal distortion p1, p2"""
#tangencijalna distorzija prve kamere
tang_Dist1 = np.array(data["CameraParameters1"]["TangentialDistortion"])
td1_P1 = tang_Dist1[0]
td1_P2 = tang_Dist1[1]

#tangencijalna distorzija druge kamere
tang_Dist2 = np.array(data["CameraParameters2"]["TangentialDistortion"])
td2_P1 = tang_Dist2[0]
td2_P2 = tang_Dist2[1]

#""D1 s cetri """
D1_4 = (np.array([rd1_K1param, rd1_K2param, td1_P1, td1_P2])).reshape(4,1)                # četri elementa
#print(D1_4)


#""D2 s cetri i pet elemenata"""
D2_4 = (np.array([rd2_K1param, rd2_K2param, td2_P1, td2_P2])).reshape(4,1)                 # ovo je samo s cetri elementa
print("D2_4", D2_4.shape)

#"" R = stereoParams.RotationOfCamera2 """
R = np.array(data["RotationOfCamera2"])
R = R.T                                                         #Transpose it
print ("R", R.shape )
print(R)

#""T = stereoParams.TranslationOfCamera2"""
T = np.asarray(data["TranslationOfCamera2"])
T = T.T                                                             #Transpose it
#print("T", T)

save_parameters = np.save("Parameters/stereoParams_cleaned", {'camera1_matrix':camera_matrix1_cv, 'camera2_matrix':camera_matrix2_cv, 'dist1_coeff4V': D1_4,'dist2_coeff4V': D2_4, 'R': R, 'T':T})