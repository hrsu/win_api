import os
import csv

#print(goodfiles)
file1 = "good.csv"
datas = []
with open(file1) as f:
    reader1 = csv.reader(f)
    reader = list(reader1)
f.close()

for i in range(len(reader)):
    if len(reader[i])<500:
        datas.append(reader[i])
    else:
        data = reader[i][:500]
        datas.append(data)

file2 = "goodAPI.csv"
#将所有好的软件调用的api序列进行处理存储到good.csv
out = open(file2, 'a', newline='')   #追加写入
w = csv.writer(out, dialect='excel')

for k in datas:
    w.writerow(k)
