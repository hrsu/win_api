import os
import numpy as np

path = os.path.abspath(".") + "\data"

# 数值文本文件直接转换为矩阵数组
def txt_to_matrix(filename):
    file = open(filename)
    lines = file.readlines()
    datamat = []  # 初始化矩阵
    for line in lines:
        line = line.strip().split(' ')  # strip()默认移除字符串首尾空格或换行符
        datamat.append(line[:])
    return datamat

#统计所有api到数组
def get_apis(data):
    api = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] not in api:
                api.append(data[i][j])
    return api

#通过api数组随机生成干扰数据放到另一个标签文件内
def write(apis, file):
    api_list = []
    for i in range(800):
        api = np.random.choice(apis, 700)
        api_list.append(api)

    with open(file, 'a') as w:  # 追加写入
        for line in api_list:
            str = ""
            for each in line:
                each = '{} '.format(each)
                str = str + each
            str = str[:-1] + '\n'
            w.write(str)

if __name__ == '__main__':
    data = txt_to_matrix(path + '\\api1.neg')   #读取txt到数组
    api = get_apis(data)     #生成api数组
    write(api, path + "\\api.pos")   #根据数组api生成随机数组写入指定文件


