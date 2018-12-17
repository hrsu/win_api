import os
import csv
import operator

file = os.path.abspath(".") + "\goodAPI1.csv"

#将csv读取成一个array
with open(file) as f:
    reader1 = csv.reader(f)
    reader = list(reader1)
f.close()

apiarray = []
#计算api的数量，提取每行
for i in range(len(reader)):
    lena = len(reader[i])
    for j in range(2, lena):
        #将api加入到apiarray数组，生成api数组
        if reader[i][j] not in apiarray:
            apiarray.append(reader[i][j])

arr = []
#给api编号
for i in range(len(apiarray)):
    num = 0
    for j in range(len(reader)):
        if apiarray[i] in reader[j]:
            num += 1
    arr.append(num)

dict = dict(zip(apiarray,arr))   #生成api字典
dict1 = sorted(dict.items(), key=operator.itemgetter(0),reverse=True)

apiarr = []
for i in range(0,len(dict1)):
    apiarr.append(dict1[i][0])

#处理数据
file = 'frequent.txt'
with open(file, 'a', newline='') as w:   #追加写入
    for i in range(len(reader)):
        q = ""
        q = q + "0"
        for j in range(len(apiarr)):
            if apiarr[j] in reader[i]:
                q = q + ",1"
            else:
                q = q + ",0"
        w.write(q + '\n')
        print(i)
w.close()
