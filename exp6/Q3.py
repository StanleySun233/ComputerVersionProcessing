import cv2
from matplotlib import pyplot as plt

img = cv2.imread('./data/fig2.png', cv2.IMREAD_GRAYSCALE)
img_origin = cv2.cvtColor(cv2.imread('./data/fig2.png'), cv2.COLOR_BGR2RGB)

img_canny = cv2.Canny(img, 256 * 0.1, 256 * 0.3)
img_canny_bin = cv2.threshold(img_canny, 128, 255, cv2.THRESH_BINARY)[1]

img_add_canny = []

for i in range(len(img)):
    img_add_canny.append([])
    for j in range(len(img[i])):
        num = img[i][j]
        if img_canny_bin[i][j] == 255:
            num = img_canny_bin[i][j]
        img_add_canny[-1].append(num)

plt.subplot(221)
plt.imshow(img_origin)
plt.title("Origin Color Picture")

plt.subplot(222)
plt.imshow(img, cmap='gray')
plt.title("Origin Picture")

plt.subplot(223)
plt.imshow(img, cmap='gray')
plt.title("Canny Picture")

plt.subplot(224)
plt.imshow(img_add_canny, cmap='gray')
plt.title("Edge Add to Origin")

plt.show()
