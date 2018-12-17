import os
import numpy as np

path = os.path.abspath(".")

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
def get_apis(data, flag):
    api = []
    for i in range(len(data)):
        for j in range(1, len(data[i])):
            if data[i][j] not in api and flag in data[i][0]:
                api.append(data[i][j])
    return api

#通过api数组随机生成干扰数据放到另一个标签文件内
def write(apis, file, flag):
    api_list = []
    for i in range(1000):
        api = np.random.choice(apis, 800)
        api_list.append(api)

    with open(file, 'a') as w:  # 追加写入
        for line in api_list:
            str1 = str(flag)  + "	"
            for each in line:
                each = '{} '.format(each)
                str1 = str1 + each
            str1 = str1[:-1] + '\n'
            w.write(str1)

if __name__ == '__main__':
    from_api = '0'
    to_api = '1'
    data = txt_to_matrix(path + '\\lstm_data.txt')   #读取txt到数组
    api = get_apis(data, from_api)     #生成api数组,1表示从1到0的数据转换
    write(api, path + "\\lstm_data2.txt", to_api)   #根据数组api生成随机数组写入指定文件,写入时标签为0
