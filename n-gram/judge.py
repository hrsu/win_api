import os
import csv
import datetime

start = datetime.datetime.now()

#传入列表a获得n-gram矩阵
def gram(a,n):
    gram_list=[]
    for j in range(len(a)):
        for i in range(len(a[j])):
            if i+n<=len(a):
                gram_list.append([l for l in a[j][i:i+n] if l not in gram_list])
            else:
                break
        print(j, ': ' ,len(gram_list))
    #print("gram:",gram_list)   #提取出的gram
    print("gram_list:",len(gram_list))
    gram_time = datetime.datetime.now()
    print("cost time: ",gram_time-start)
    return gram_list

#判断a是否在b内
def judge(a, b):
    flag = 1
    for i in range(len(a)):
        if a[i] in b:
            b1 = b.index(a[i])
            b_in = b[b1:]
            b = b_in
        else:
            flag = 0
    return flag

#获得序列矩阵
def sequence(n,a):
    g = gram(a,n)
    ws = []
    for i in range(len(a)):
        print(i)
        #print(i)
        w = []
        if i < 1500:
            w.append(1)
        else:
            w.append(0)
        for j in range(len(g)):
            w.append(judge(g[j],a[i]))
        ws.append(w)
    list_time = datetime.datetime.now()
    print("list time: ",list_time-start)
    return ws

def read_data(file):
    # 将csv读取成一个array
    with open(file) as f:
        reader1 = csv.reader(f)
        reader = list(reader1)
    f.close()

    apiarray1 = []
    # 计算api的数量，提取每行
    for i in range(len(reader)):
        lena = len(reader[i])
        for j in range(2, lena):
            # 将api加入到apiarray数组，生成api数组
            if reader[i][j] not in apiarray1 and reader[i][j]!='':
                apiarray1.append(reader[i][j])
    apiarray = []
    for api in apiarray1:
        apiarray.append(int(api))

    all_datas = []
    for i in range(len(reader)):
        all_data = []
        if i < 1500:
            all_data.append(1)
        else:
            all_data.append(0)
        for j in range(2,len(reader[i])):
            if reader[i][j]!='':
                all_data.append(int(reader[i][j]))
        all_datas.append(all_data)
    return apiarray,all_datas

def main():
    file = os.path.abspath(".") + "\\all_data.csv"
    apiarray, all_data = read_data(file)   #获得api总的列表和整个序列矩阵
    ws = sequence(1,all_data)   #挖掘n-gram序列
    #print(all_data)
    #print(apiarray)
    #print(ws)
    #print(len(ws))

    # 处理数据
    file = 'sequence.txt'
    with open(file, 'a', newline='') as w:  # 追加写入
        for i in range(len(ws)):
            api_line = ''
            for j in range(len(ws[i])):
                api_line = api_line + str(ws[i][j]) + ','
            api_line = api_line[:-1]
            w.write(api_line + '\n')
            print(i)
    w.close()

    with open(os.path.abspath(".") + '\\sequence.csv', 'w', newline='') as csvfile:
        w = csv.writer(csvfile, dialect='excel')
        for api_list in ws:
            w.writerow(api_list)
    end = datetime.datetime.now()
    print("total cost: ",end-start)


main()
'''
a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
b = [[5,7,4],[1,6,9,4],[2,4,8,5],[3,4,1,6],[1,5,2,3],[1,7,4,6],[5,4,9,2]]
ws = sequence(a,1,b)   #挖掘n-gram序列

print(ws)
print(len(ws))
'''