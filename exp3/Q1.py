import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./data/fig.jpg', cv2.IMREAD_GRAYSCALE)
dst = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dst_center = np.fft.fftshift(dst)
result = 20 * np.log(np.abs(cv2.magnitude(dst_center[:, :, 0], dst_center[:, :, 1])))

cv2.imwrite('./data/fig1.jpg', result)

plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.subplot(122)
plt.axis("off")
plt.imshow(result, cmap="gray")
plt.show()
