import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('right_RGB15.jpg')
color = ('b','g','r')
cv2.imshow("img1", img)
#normalizacija histograma
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
#for i, col in enumerate(color):
#    histr = cv2.calcHist([img],[i],None,[256],[0,256])
#    plt.plot(histr,color = col)
#    plt.xlim([0,256])
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img2 = cdf[img]
cv2.imshow("img", img2)
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()