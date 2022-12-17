import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
gray = cv2.imread('./data/fig2.png', cv2.IMREAD_GRAYSCALE)

# 计算图像直方图
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# 初始化阈值
threshold = 0

# 初始化类内方差和
variance_sum = float("inf")

# 遍历每个阈值，计算类内方差和
for i in range(1, 255):
    # 计算前景和背景的像素值和
    foreground_sum = np.sum(hist[:i])
    background_sum = np.sum(hist[i:])

    # 计算前景和背景的平均灰度值
    foreground_mean = np.sum(np.arange(1, i + 1) * hist[:i]) / foreground_sum
    background_mean = np.sum(np.arange(i + 1, 256) * hist[i:]) / background_sum

    # 计算类内方差和
    class_variance_sum = foreground_sum * (foreground_mean - threshold) ** 2 + background_sum * (
            background_mean - threshold) ** 2

    # 如果类内方差和更小，则更新阈值
    if class_variance_sum < variance_sum:
        variance_sum = class_variance_sum
        threshold = i

ret, img = cv2.threshold(gray, threshold, 255, 0)

plt.subplot(121)
plt.imshow(gray, cmap='gray')
plt.title("Origin Picture")

plt.subplot(122)
plt.imshow(img, cmap='gray')
plt.title(f"Binary Picture while t = {threshold}")

plt.show()
