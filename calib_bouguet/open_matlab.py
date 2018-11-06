import scipy.io as spio

mat = spio.loadmat('latest.mat', squeeze_me=True, struct_as_record=True)

print(mat)


#import numpy as np
#import h5py
#f = h5py.File('/home/minneyar/Desktop/stereocam2/calib_bouguet/calibrationSession_stereo.mat','r')
#data = f.get('data/variable1')
##myvar = f['myvar'].value
#data = np.array(data) # For converting to numpy array
