import matplotlib.pyplot as plt

fig = plt.figure()

for i in range(4):
    plt.subplot(2, 2, i+1)
    fig = plt.imread(f'./data/{i+5}.jpg')
    plt.imshow(fig)
    plt.axis(False)
    plt.title(f'Figure {i+1}')

plt.show()
