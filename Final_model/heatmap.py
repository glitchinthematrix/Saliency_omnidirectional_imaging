import cv2
from matplotlib import pyplot as plt

img_max =cv2.imread('SM_smooth_max.jpg',0)
img_median=cv2.imread('SM_smooth_median.jpg',0)

plt.imsave('HM_median.png', img_median, cmap='jet')
plt.imsave('HM_max.png', img_median, cmap='jet')
