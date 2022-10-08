import matplotlib.pyplot as plt
import cv2
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

img = cv2.resize(cv2.imread('./data/3.png', cv2.IMREAD_GRAYSCALE), (1920, 1080))

cv2.imshow('pic', img)
cv2.waitKey(0)

img_resize = cv2.resize(img, (400, 200))

img_liner = cv2.resize(img, (400, 200), interpolation=cv2.INTER_LINEAR)

img_lancz = cv2.resize(img, (400, 200), interpolation=cv2.INTER_LANCZOS4)

img_cubic = cv2.resize(img, (400, 200), interpolation=cv2.INTER_CUBIC)

gray = [img_resize, img_liner, img_lancz, img_cubic]
label = ['cv2默认的转换方式','线性插值方法','傅里叶变化','三次插值']

fig, ax = plt.subplots(2, 2)

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(gray[i], cmap='gray')
    ax[i // 2][i % 2].set_title(label[i])
    plt.axis(False)

plt.show()