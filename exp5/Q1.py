import cv2
import matplotlib.pyplot as plt
import numpy as np

img = np.array([[1, 5, 255, 225, 100, 200, 255, 200],
                [1, 7, 254, 255, 100, 10, 10, 9],
                [3, 7, 10, 100, 100, 2, 9, 6],
                [3, 6, 10, 10, 9, 2, 8, 2],
                [2, 1, 8, 8, 9, 3, 4, 2],
                [1, 0, 7, 8, 8, 3, 2, 1],
                [1, 1, 8, 8, 7, 2, 2, 1],
                [2, 3, 9, 8, 7, 2, 2, 0]])

f = np.array([[1, 5, 255, 225, 100, 200, 255, 200],
              [1, 7, 254, 255, 100, 10, 10, 9],
              [3, 7, 10, 100, 100, 2, 9, 6],
              [3, 6, 10, 10, 9, 2, 8, 2],
              [2, 1, 8, 8, 9, 3, 4, 2],
              [1, 0, 7, 8, 8, 3, 2, 1],
              [1, 1, 8, 8, 7, 2, 2, 1],
              [2, 3, 9, 8, 7, 2, 2, 0]])

plt.rc("font", family='Microsoft YaHei')
plt.subplot(221)
plt.title("1. 原图")
plt.axis('off')
plt.imshow(img, cmap="gray")

bar = np.zeros(256, dtype=int)
for i in img:
    for j in i:
        bar[j] += 1

fa = 90
fb = 150
ga = 50
gb = 210
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j] <= 80:
            img[i][j] *= 0.3125
        elif img[i][j] <= fb:
            img[i][j] = fa * (img[i][j] - 80) / 9 + ga
        else:
            img[i][j] = (img[i][j] - fb) * 3 / 13 + gb
plt.subplot(223)
plt.plot(bar)
plt.title('原始直方图')
bar = np.zeros(256, dtype=int)
for i in img:
    for j in i:
        bar[j] += 1
plt.subplot(224)
plt.plot(bar)
plt.title('线性展宽后的直方图')
plt.subplot(222)
plt.title("2. 线性展宽后")
plt.axis('off')
plt.imshow(img, cmap="gray")
plt.show()


def contrast(img1):
    m, n = img1.shape
    img1_ext = cv2.copyMakeBorder(img1, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    rows_ext, cols_ext = img1_ext.shape

    b = 0.0
    for i in range(1, rows_ext - 1):
        for j in range(1, cols_ext - 1):
            b += ((img1_ext[i, j] - img1_ext[i, j + 1]) ** 2 + (img1_ext[i, j] - img1_ext[i, j - 1]) ** 2 +
                  (img1_ext[i, j] - img1_ext[i + 1, j]) ** 2 + (img1_ext[i, j] - img1_ext[i - 1, j]) ** 2)

    cg = b / (4 * (m - 2) * (n - 2) + 3 * (2 * (m - 2) + 2 * (n - 2)) + 2 * 4)
    return cg


print("原图的对比度：", contrast(f))
print("三段式线性对比度展宽的对比度：", contrast(img))
