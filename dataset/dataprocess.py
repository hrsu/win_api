import os
import csv

badpath = os.path.abspath(".") + "\\bad\\"
badfiles = os.listdir(badpath)   #所有恶意软件名

goodpath = os.path.abspath(".") + "\\good\\"
goodfiles = os.listdir(goodpath)   #所有非恶意软件名

#print(goodfiles)
file1 = goodpath + goodfiles[0]
data = []
with open(file1) as f:
    reader1 = csv.reader(f)
    reader = list(reader1)
f.close()

for i in range(len(reader)):
    if i % 2 ==0:
        data.append(reader[i])

print(data[0])