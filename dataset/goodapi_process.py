import os
import csv

goodpath = os.path.abspath(".") + "\\good\\"
goodfiles = os.listdir(goodpath)   #所有非恶意软件名

#将所有好的软件调用的api序列进行处理存储到good.csv
file = "good.csv"
out = open(file, 'a', newline='')   #追加写入
w = csv.writer(out, dialect='excel')

#一次处理每个良性软件
for i in range(len(goodfiles)):
    text = []  # 要写入文件的信息
    file1 = goodpath + goodfiles[i]   #每个文件的路径
    data = []
    with open(file1) as f:
        reader1 = csv.reader(f)
        reader = list(reader1)
    #跳过空行
    for j in range(len(reader)):
        if j % 2 == 0:
            data.append(reader[j])

    name = goodfiles[i].split('.')[0]
    text.append(name)    #第一列是文件名
    text.append("none")  # 第二列是md5码
    for k in range(len(data)):
        text.append(data[k][3])
    w.writerow(text)

