import matplotlib.pyplot as plt
import numpy as np

f = np.array([100, 67, 34, 100,
              67, 67, 34, 100,
              67, 56, 211, 67,
              100, 100, 211, 100])
f_format = np.reshape(f, (4, 4))
plt.matshow(f_format, cmap=plt.cm.Greys)
plt.show()

x_label = [f'{i} - {i + 32}' for i in range(0, 255, 32)]
y_value = [0 for i in range(8)]

for i in f:
    y_value[i // 32] += 1

plt.hist(f)
plt.xlabel('Range')
plt.ylabel('Count')
plt.show()
