import csv
import os
FilePath = os.path.abspath('.')
with open(FilePath+'\\frequent.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, dialect='excel')
    # 读要转换的txt文件，文件每行各词间以@@@字符分隔
    with open(FilePath+'\\frequent.txt', 'r') as filein:
        for line in filein:
            line_list = line.strip('\n').split(',')
            spamwriter.writerow(line_list)