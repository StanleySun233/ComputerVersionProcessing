import numpy as np
import matplotlib.pyplot as plt


def _mean(_img, _round=2):
    _sum = 0
    _cnt = 0
    for i in _img:
        for j in i:
            _sum += j
            _cnt += 1
    return round(_sum / _cnt, _round)


def _std(_img, _round=2):
    __mean = _mean(_img, _round)
    _cnt = 0
    _sum = 0
    for i in _img:
        for j in i:
            _sum += (j - __mean) ** 2
            _cnt += 1
    return round(np.sqrt(_sum / _cnt), 2)


gray256 = plt.imread('./data/2.bmp')
np_mean = round(np.reshape(gray256, (-1,)).mean(), 2)
np_std = round(np.reshape(gray256, (-1,)).std(), 2)
print(f"自定义函数：\t平均数=\t{_mean(gray256)}\t标准差=\t{_std(gray256)}")
print(f"内置函数：\t平均数=\t{np_mean}\t标准差=\t{np_std}")
