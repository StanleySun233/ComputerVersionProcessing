import cv2
from matplotlib import pyplot as plt

# 读取图像
img = cv2.imread('./data/fig2.png', cv2.IMREAD_GRAYSCALE)

# 计算图像的灰度分布
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# 确定划分阈值
p = 0.4
s = 0
threshold = 127
for i in range(256):
    s += hist[i]
    if s >= img.size * p:
        threshold = i
        break

# 使用计算出的阈值进行二值化分割
_, img_binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title("Origin Picture")

plt.subplot(132)
plt.imshow(img_binary, cmap='gray')
plt.title(f"Binary Picture while t = {threshold}")

plt.subplot(133)
plt.hist(img)
plt.title(f"Hist Img")

plt.show()
