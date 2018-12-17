import os
import csv

file = os.path.abspath(".") + "\goodAPI1.csv"
file1 = os.path.abspath(".") + "\API_dataset1.csv"

#将csv读取成一个array
with open(file) as f:
    reader1 = csv.reader(f)
    reader = list(reader1)
f.close()

#将csv读取成一个array
with open(file1) as f:
    reader2 = csv.reader(f)
    reader3 = list(reader2)
f.close()

reader.extend(reader3)

apiarray = {}
#计算api的数量，提取每行
for i in range(len(reader)):
    lena = len(reader[i])
    for j in range(2, lena):
        #将api加入到apiarray数组，生成api数组
        if reader[i][j] not in apiarray and reader:
            apiarray[reader[i][j]] = 1
        else:
            apiarray[reader[i][j]] = apiarray[reader[i][j]] + 1

aaa = sorted(apiarray.items(),key = lambda x:x[1],reverse = True)

print(len(aaa))
lens = 0
for k,v in aaa:
    if k != '':
        lens += v

print(lens)

'''
arr = []
#给api编号
for i in range(len(apiarray)):
    arr.append(i+1)

dict1 = dict(zip(apiarray,arr))   #生成api字典

#apidict = 'good_apidict.txt'
#with open(apidict,'w') as f:
#    for k in dict1.items():
#        f.write(str(k[0]) + ': ')
#        f.write(str(k[1]) + '\n')
#f.close()

#将api编号写入新的文件

file = 'tgood.csv'
out = open(file, 'a', newline='')   #追加写入
w = csv.writer(out, dialect='excel')
for i in range(len(reader)):
    newdata = []
    lena = len(reader[i])
    #newdata.append(reader[i][0])
    #newdata.append(reader[i][1])   #前两列信息不变
    for j in range(2, lena):
        #将api加入到apiarray数组
        newdata.append(dict1[reader[i][j]])   #查询字典api的编号
    # 将csv读取成一个array
    w.writerow(newdata)

print(len(apiarray))
print(apiarray)
'''

