import os
import csv

file = os.path.abspath(".") + "\data.csv"

#将csv读取成一个array
with open(file) as f:
    reader1 = csv.reader(f)
    reader = list(reader1)
f.close()

data = []
for i in range(len(reader)):
    d = ''
    for j in range(len(reader[i])):
        d += reader[i][j] + ' '
    data.append(d)

print(data)
print(len(data))