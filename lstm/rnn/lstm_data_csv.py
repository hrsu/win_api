import os
import csv

#将良性软件训练完成后将此处改为good.csv,追加到lstm_data末尾，删除19行的判断语句，改为good则为0
file = os.path.abspath(".") + "\API_dataset1.csv"

#将csv读取成一个array
with open(file) as f:
    reader1 = csv.reader(f)
    reader = list(reader1)
f.close()

'''
#处理数据
file = 'data.txt'
with open(file, 'a', newline='') as w:   #追加写入
    for i in range(len(reader)):
        newdata = ''
        lena = len(reader[i])
        #newdata = newdata + '1\t'   #1为标签
        for j in range(2, lena):
            newdata = newdata + str(reader[i][j]) + ' '
        w.write(newdata + '\n')
        print(i)
w.close()
'''


with open('data.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(reader)):
        print(i)
        newdata = []
        lena = len(reader[i])
        # newdata = newdata + '1\t'   #1为标签
        for j in range(2, lena):
            newdata.append(reader[i][j])
        writer.writerow(newdata)
