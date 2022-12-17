import cv2
import matplotlib.pyplot as plt
import numpy as np


def float2u8(num):
    if num > 255 or num < -255:
        return 255
    elif -255 <= num <= 255:
        if abs(num - int(num)) < 0.5:
            return np.uint8(abs(num))
        else:
            return np.uint8(abs(num)) + 1


def sobel(img, k=0):
    row = img.shape[0]
    col = img.shape[1]
    image = np.zeros((row, col), np.uint8)
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            y = int(img[i - 1, j + 1, k]) - int(img[i - 1, j - 1, k]) + 2 * (
                    int(img[i, j + 1, k]) - int(img[i, j - 1, k])) + int(img[i + 1, j + 1, k]) - int(
                img[i + 1, j - 1, k])
            x = int(img[i + 1, j - 1, k]) - int(img[i - 1, j - 1, k]) + 2 * (
                    int(img[i + 1, j, k]) - int(img[i - 1, j, k])) + int(img[i + 1, j + 1, k]) - int(
                img[i - 1, j + 1, k])
            image[i, j] = float2u8(abs(x) * 0.5 + abs(y) * 0.5)
    return image


img = cv2.imread('./data/fig1.png')
plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title("Origin Picture")

plt.subplot(222)
plt.imshow(sobel(img, 0), cmap='gray')
plt.title("Sobel Picture")
plt.subplot(223)
plt.imshow(cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1], cmap='gray')
plt.title("Origin Picture after binary")

plt.subplot(224)
plt.imshow(cv2.threshold(sobel(img), 128, 255, cv2.THRESH_BINARY)[1], cmap='gray')
plt.title("Sobel Picture after binary")

plt.show()