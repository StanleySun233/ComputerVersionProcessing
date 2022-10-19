import cv2
import numpy as np
import matplotlib.pyplot as plt


def lowPassFiltering(img, size):
    h, w = img.shape[0:2]
    h1, w1 = int(h / 2), int(w / 2)
    img2 = np.zeros((h, w), np.uint8)
    img2[h1 - int(size / 2):h1 + int(size / 2),
    w1 - int(size / 2):w1 + int(size / 2)] = 1
    img3 = img2 * img
    return img3


gray = cv2.imdecode(np.fromfile('./data/fig.jpg', dtype=np.uint8), 0)
gray = cv2.resize(gray, (256, 256))

h, w = gray.shape

img_dft = np.fft.fft2(gray)
dft_shift = np.fft.fftshift(img_dft)

dft_shift = lowPassFiltering(dft_shift, h // 3)
res = np.log(np.abs(dft_shift))


idft_shift = np.fft.ifftshift(dft_shift)
ifimg = np.fft.ifft2(idft_shift)
ifimg = np.abs(ifimg)

plt.subplot(131), plt.imshow(gray, 'gray'), plt.title('origin')
plt.axis('off')
plt.subplot(132), plt.imshow(res, 'gray'), plt.title('low filter')
plt.axis('off')
plt.subplot(133), plt.imshow(np.int8(ifimg), 'gray'), plt.title('after filter')
plt.axis('off')
plt.show()