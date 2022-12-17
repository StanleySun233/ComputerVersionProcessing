import cv2
import matplotlib.pyplot as plt
import numpy as np


def LaplacianFilter(src):
    src = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    src = np.float64(src)

    rows, cols = src.shape
    g = np.zeros(src.shape, dtype=np.float64)
    tempLap = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            temp = src[i - 1:i + 2, j - 1:j + 2]
            g[i, j] = abs(np.sum(temp * tempLap))

    gmax = np.max(g)
    gmin = np.min(g)
    t = gmax - gmin
    g = (g - gmin) / t * 255.0

    g = np.uint8(g + 0.5)
    dst = g[1:rows - 1, 1:cols - 1]
    return dst


def NoneBack_Laplacian(src):
    src = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    src = np.float64(src)

    rows, cols = src.shape
    g = np.zeros(src.shape, dtype=np.float64)
    tempLap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)  # 仅仅修改了系数和，使之为0,即可保留背景
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            temp = src[i - 1:i + 2, j - 1:j + 2]
            g[i, j] = abs(np.sum(temp * tempLap))

    gmax = np.max(g)
    gmin = np.min(g)
    t = gmax - gmin
    g = (g - gmin) / t * 255.0

    g = np.uint8(g + 0.5)
    dst = g[1:rows - 1, 1:cols - 1]
    return dst


img = cv2.imread('./data/fig1.png',cv2.IMREAD_GRAYSCALE)
plt.subplot(321)
plt.imshow(img, cmap='gray')
plt.title("Origin Picture")

plt.subplot(322)
plt.imshow(LaplacianFilter(img), cmap='gray')
plt.title("Laplacian Picture")

plt.subplot(323)
plt.imshow(cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1], cmap='gray')
plt.title("Origin Picture after binary")

plt.subplot(324)
plt.imshow(cv2.threshold(LaplacianFilter(img), 128, 255, cv2.THRESH_BINARY)[1], cmap='gray')
plt.title("Laplacian Picture after binary")

plt.subplot(325)
plt.imshow(NoneBack_Laplacian(img), cmap='gray')
plt.title("Laplacian No Blank Picture")

plt.subplot(326)
plt.imshow(cv2.threshold(NoneBack_Laplacian(img), 128, 255, cv2.THRESH_BINARY)[1], cmap='gray')
plt.title("Laplacian No Blank Picture after binary")

plt.show()
