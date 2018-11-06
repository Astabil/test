import numpy as np

data = np.load('Camera_matrices.npy').item()
camera1_matrix = data['camera1_matrix']
camera2_matrix = data['camera2_matrix']
print(camera1_matrix)
print(camera2_matrix)

# print(np.array(Kl))
# print(np.array(Dl))
# print(DIMl)