import matplotlib.pyplot as plt
import cv2
import numpy as np


def contrast(_img):
    m, n = _img.shape
    img1_ext = cv2.copyMakeBorder(_img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    rows_ext, cols_ext = img1_ext.shape

    b = 0.0
    for i in range(1, rows_ext - 1):
        for j in range(1, cols_ext - 1):
            b += ((img1_ext[i, j] - img1_ext[i, j + 1]) ** 2 + (img1_ext[i, j] - img1_ext[i, j - 1]) ** 2 +
                  (img1_ext[i, j] - img1_ext[i + 1, j]) ** 2 + (img1_ext[i, j] - img1_ext[i - 1, j]) ** 2)

    cg = b / (4 * (m - 2) * (n - 2) + 3 * (2 * (m - 2) + 2 * (n - 2)) + 2 * 4)
    return cg


def def_equalizehist(_img, L=256):
    _img = cv2.imread(_img, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("ori", _img)
    h, w = _img.shape

    hist = cv2.calcHist([_img], [0], None, [256], [0, 255])

    hist[0:255] = hist[0:255] / (h * w)

    sum_hist = np.zeros(hist.shape)

    for i in range(256):
        sum_hist[i] = sum(hist[0:i + 1])
    equal_hist = np.zeros(sum_hist.shape)

    for i in range(256):
        equal_hist[i] = int(((L - 1) - 0) * sum_hist[i] + 0.5)
    equal_img = _img.copy()

    for i in range(h):
        for j in range(w):
            equal_img[i, j] = equal_hist[_img[i, j]]

    equal_hist = cv2.calcHist([equal_img], [0], None, [256], [0, 256])
    equal_hist[0:255] = equal_hist[0:255] / (h * w)
    cv2.imshow("inverse", equal_img)
    plt.plot(hist, color='b')
    plt.show()
    plt.plot(equal_hist, color='r')
    plt.show()
    cv2.waitKey()
    return [equal_img, equal_hist]


img, hist = def_equalizehist("./data/pic1.jpg")
origin = cv2.imread("./data/pic1.jpg", cv2.IMREAD_GRAYSCALE)

print("直方图均衡化的均衡化：", contrast(img))
print("原始图片的均衡化：", contrast(origin))
