import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 8)  # 设置figure_size尺寸
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
"""
import cv2


img = cv2.imread('./data/1.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (1920, 1080))
cv2.imwrite('./data/2.bmp', img)
"""


def imageQuantization(_img, quant=256):
    _shape = _img.shape
    sheet = []
    q = 256 // quant
    for i in _img:
        sheet.append([])
        for j in i:
            sheet[-1].append(int(j // q * q))

    return np.array(sheet, dtype='uint8')


gray256 = plt.imread('./data/2.bmp')
gray128 = imageQuantization(gray256, 128)
gray32 = imageQuantization(gray256, 32)
gray2 = imageQuantization(gray256, 2)
gray = [gray256, gray128, gray32, gray2]
label = ['256', '128', '32', '2']

fig, ax = plt.subplots(2, 2)

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(gray[i], cmap='gray')
    ax[i // 2][i % 2].set_title(label[i])
    plt.axis(False)

# fig.suptitle('不同采样量级下的灰度图片显示')
plt.show()
