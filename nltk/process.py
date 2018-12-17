import os
import csv

file = os.path.abspath(".") + "\goodAPI1.csv"

#将csv读取成一个array
with open(file) as f:
    reader1 = csv.reader(f)
    reader = list(reader1)
f.close()

#将api编号写入新的文件
file = 'goodAPI.csv'
out = open(file, 'a', newline='')   #追加写入
w = csv.writer(out, dialect='excel')
for i in range(len(reader)):
    newdata = []
    lena = len(reader[i])
    #newdata.append(reader[i][0])
    #newdata.append(reader[i][1])   #前两列信息不变
    for j in range(2, lena):
        #将api加入到apiarray数组
        newdata.append(reader[i][j])   #查询字典api的编号
    # 将csv读取成一个array
    w.writerow(newdata)