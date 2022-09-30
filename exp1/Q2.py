import cv2
import numpy as np

fig = cv2.imread('./data/1.jpg', 0)
cv2.imshow('Figure', fig)
cv2.waitKey(0)

fig_array = (np.array(fig, dtype='int32') + 20) % 255
fig_array = fig_array.astype('int8')

cv2.imshow('Figure', fig_array)
cv2.waitKey(0)
