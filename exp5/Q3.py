import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

img2 = cv2.imread("./data/pic2.jpg", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("./data/pic3.jpg", cv2.IMREAD_GRAYSCALE)


def matrix_conv(arr, kernel):
    n = len(kernel)
    ans = 0
    for i in range(n):
        for j in range(n):
            ans += arr[i, j] * float(kernel[i, j])
    return ans / arr.shape[0] / arr.shape[1]


def filter(image, kernel=np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])):
    n = len(kernel)
    image_1 = np.zeros((image.shape[0] + 2 * (n - 1), image.shape[1] + 2 * (n - 1)))
    image_1[(n - 1):(image.shape[0] + n - 1), (n - 1):(image.shape[1] + n - 1)] = image
    image_2 = np.zeros((image_1.shape[0] - n + 1, image_1.shape[1] - n + 1))
    for i in range(image_1.shape[0] - n + 1):
        for j in range(image_1.shape[1] - n + 1):
            temp = image_1[i:i + n, j:j + n]
            image_2[i, j] = matrix_conv(temp, kernel)
    new_image = image_2[(n - 1):(n + image.shape[0] - 1), (n - 1):(n + image.shape[1] - 1)]
    return new_image


conv = np.ones((7, 7), np.float32) / 25

img2_filter = filter(img2, conv)
img3_filter = filter(img3, conv)

plt.subplot(221)
plt.imshow(img2, cmap='gray')
plt.title('原始图2')

plt.subplot(222)
plt.imshow(img3, cmap='gray')
plt.title('原始图3')

plt.subplot(223)
plt.imshow(img2_filter, cmap='gray')
plt.title('卷积图2')

plt.subplot(224)
plt.imshow(img3_filter, cmap='gray')
plt.title('卷积图3')

plt.show()
