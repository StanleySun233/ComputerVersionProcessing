# 使用matplotlib完成
import matplotlib.pyplot as plt

img = plt.imread('./data/1.jpg')
# 读取
plt.imshow(img)
plt.axis(False)
plt.show()
# 显示
plt.imsave('./data/3.jpg', img)
# 保存
