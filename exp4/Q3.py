import cv2
import numpy as np
import math
import os
from fnmatch import fnmatch
from datetime import datetime
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 以二进制的方式读取文件，结果为字节
def fileload(filename):
    file_pth = os.path.dirname(__file__) + '/' + filename
    file_in = os.open(file_pth, os.O_BINARY | os.O_RDONLY)
    file_size = os.stat(file_in)[6]
    data = os.read(file_in, file_size)
    os.close(file_in)
    return data


# 计算文件中不同字节的频数和累积频数
def cal_pr(data):
    pro_dic = {}
    data_set = set(data)
    for i in data_set:
        pro_dic[i] = data.count(i)
    sym_pro = []
    accum_pro = []
    keys = []
    accum_p = 0
    data_size = len(data)
    for k in sorted(pro_dic, key=pro_dic.__getitem__, reverse=True):
        sym_pro.append(pro_dic[k])
        keys.append(k)
    for i in sym_pro:
        accum_pro.append(accum_p)
        accum_p += i
    accum_pro.append(data_size)
    tmp = 0
    for k in sorted(pro_dic, key=pro_dic.__getitem__, reverse=True):
        pro_dic[k] = [pro_dic[k], accum_pro[tmp]]
        tmp += 1
    return pro_dic, keys, accum_pro


# 编码
def encode(data, pro_dic, data_size):
    C_up = 0
    A_up = A_down = C_down = 1
    for i in range(len(data)):
        C_up = C_up * data_size + A_up * pro_dic[data[i]][1]
        C_down = C_down * data_size
        A_up *= pro_dic[data[i]][0]
        A_down *= data_size
    L = math.ceil(len(data) * math.log2(data_size) - math.log2(A_up))
    bin_C = dec2bin(C_up, C_down, L)
    amcode = bin_C[0:L]
    return C_up, C_down, amcode


# 译码
def decode(C_up, C_down, pro_dic, keys, accum_pro, byte_num, data_size):
    byte_list = []
    for i in range(byte_num):
        k = binarysearch(accum_pro, C_up * data_size / C_down)
        if k == len(accum_pro) - 1:
            k -= 1
        key = keys[k]
        byte_list.append(key)
        C_up = (C_up * data_size - C_down * pro_dic[key][1]) * data_size
        C_down = C_down * data_size * pro_dic[key][0]
    return byte_list


# 二分法搜索
def binarysearch(pro_list, target):
    low = 0
    high = len(pro_list) - 1
    if pro_list[0] <= target <= pro_list[-1]:
        while high >= low:
            middle = int((high + low) / 2)
            if (pro_list[middle] < target) & (pro_list[middle + 1] < target):
                low = middle + 1
            elif (pro_list[middle] > target) & (pro_list[middle - 1] > target):
                high = middle - 1
            elif (pro_list[middle] < target) & (pro_list[middle + 1] > target):
                return middle
            elif (pro_list[middle] > target) & (pro_list[middle - 1] < target):
                return middle - 1
            elif (pro_list[middle] < target) & (pro_list[middle + 1] == target):
                return middle + 1
            elif (pro_list[middle] > target) & (pro_list[middle - 1] == target):
                return middle - 1
            elif pro_list[middle] == target:
                return middle
        return middle
    else:
        return False


# 整数二进制转十进制
def int_bin2dec(bins):
    dec = 0
    for i in range(len(bins)):
        dec += int(bins[i]) * 2 ** (len(bins) - i - 1)
    return dec


# 小数十进制转二进制
def dec2bin(x_up, x_down, L):
    bins = ""
    while (x_up != x_down) & (len(bins) < L):
        x_up *= 2
        if x_up > x_down:
            bins += "1"
            x_up -= x_down
        elif x_up < x_down:
            bins += "0"
        else:
            bins += "1"
    return bins


# 保存文件
def filesave(data_after, filename):
    file_pth = os.path.dirname(__file__) + '/' + filename
    if fnmatch(filename, "*_am.*") == True:
        file_open = os.open(file_pth, os.O_WRONLY | os.O_CREAT | os.O_BINARY)
        os.write(file_open, data_after)
        os.close(file_open)
    else:
        byte_list = []
        byte_num = math.ceil(len(data_after) / 8)
        for i in range(byte_num):
            byte_list.append(int_bin2dec(data_after[8 * i:8 * (i + 1)]))
        file_open = os.open(file_pth, os.O_WRONLY | os.O_CREAT | os.O_BINARY)
        os.write(file_open, bytes(byte_list))
        os.close(file_open)
        return byte_num  # 返回字节数


# 计算编码效率
def code_efficiency(pro_dic, data_size, bit_num):
    entropy = 0
    for k in pro_dic.keys():
        entropy += (pro_dic[k][0] / data_size) * (math.log2(data_size) - math.log2(pro_dic[k][0]))


# 主函数
def amcode(filename, filetype):
    for i in range(len(filename)):
        print(60 * "-")
        print("加载文件:", filename[i] + filetype[i])

        data = fileload(filename[i] + filetype[i])
        data_size = len(data)
        print("计算字节的概率..")

        pro_dic, keys, accum_pro = cal_pr(data)
        amcode_ls = ""
        C_upls = []
        C_downls = []
        byte_num = 1000
        integra = math.ceil(data_size / byte_num)

        for k in range(integra):
            C_up, C_down, amcode = encode(data[byte_num * k: byte_num * (k + 1)], pro_dic, data_size)
            amcode_ls += amcode
            C_upls.append(C_up)
            C_downls.append(C_down)
        codebyte_num = filesave(amcode_ls, filename[i] + '.am')
        print("编码完成.")
        print("保存编码文件为: " + filename[i] + '.am')
        print("压缩比(原图大小除以压缩后大小)  %.3f%%" % ((data_size / codebyte_num) * 100))
        code_efficiency(pro_dic, data_size, len(amcode_ls))
        print()

        decodebyte_ls = []

        for k in range(integra):
            if (k == integra - 1) & (data_size % byte_num != 0):
                decodebyte_ls += decode(C_upls[k], C_downls[k], pro_dic, keys, accum_pro, data_size % byte_num,
                                        data_size)
            else:
                decodebyte_ls += decode(C_upls[k], C_downls[k], pro_dic, keys, accum_pro, byte_num, data_size)

        filesave(bytes(decodebyte_ls), filename[i] + '_am' + filetype[i])

        print("保存解压文件: " + filename[i] + '_am' + filetype[i])

        errornum = 0
        for j in range(data_size):
            if data[j] != decodebyte_ls[j]:
                errornum += 1
        print("误码率: %.3f%%" % (errornum / data_size * 100))
        import cv2
        org_img = cv2.imread(filename[i] + '_am' + filetype[i], 1)
        dep_img = cv2.imread(filename[i] + '_am' + filetype[i], 1)
        plt.figure()

        plt.suptitle('阈值编码')
        plt.subplot(1, 2, 1)
        plt.title('原图')
        plt.imshow(cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB))

        plt.subplot(1, 2, 2)
        plt.title('解压后')
        plt.imshow(cv2.cvtColor(dep_img, cv2.COLOR_BGR2RGB))
        plt.show()
        return dep_img


def matrix_conv(arr, kernel):
    n = len(kernel)
    ans = 0
    for i in range(n):
        for j in range(n):
            ans += arr[i, j] * float(kernel[i, j])
    return ans


def conv_2(image, kernel=np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])):
    n = len(kernel)
    image_1 = np.zeros((image.shape[0] + 2 * (n - 1), image.shape[1] + 2 * (n - 1)))
    image_1[(n - 1):(image.shape[0] + n - 1), (n - 1):(image.shape[1] + n - 1)] = image
    image_2 = np.zeros((image_1.shape[0] - n + 1, image_1.shape[1] - n + 1))
    for i in range(image_1.shape[0] - n + 1):
        for j in range(image_1.shape[1] - n + 1):
            temp = image_1[i:i + n, j:j + n]
            image_2[i, j] = matrix_conv(temp, kernel)
    new_image = image_2[(n - 1):(n + image.shape[0] - 1), (n - 1):(n + image.shape[1] - 1)]
    return new_image


img = np.array(cv2.imread('./data/1.jpg', cv2.IMREAD_GRAYSCALE), dtype='uint8')
img_conv = conv_2(img)
cv2.imwrite('./data/2.jpg', img_conv)

img_dft = np.fft.fft2(img)
img_shift = np.fft.fftshift(img_dft)
img_gray = 20 * np.log(np.abs(img_shift))
img_zip = amcode(filename=["./data/2"], filetype=[".jpg"])

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('original')
plt.axis('off')

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
plt.title('result')
plt.axis('off')

plt.subplot(223)
plt.imshow(img_conv, cmap='gray')
plt.title('conv')
plt.axis('off')

plt.subplot(224)
plt.imshow(img_zip, cmap='gray')
plt.title('zip')
plt.axis('off')

diff = img_conv - img_zip.mean(axis=2)
print(f"平均数 {np.abs(diff.mean())}")
print(f"标准差 {diff.std()}")

plt.show()
