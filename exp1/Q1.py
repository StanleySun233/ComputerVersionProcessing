import cv2

fig = cv2.imread('./data/1.jpg')
# 图像读取
cv2.imwrite('./data/2.jpg', fig)
# 图像保存
cv2.imshow('Figure', fig)
cv2.waitKey(0)
# 图像显示