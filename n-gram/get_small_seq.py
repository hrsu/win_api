import sys
import numpy as np
result=[]
lineq = 0
with open('sequence.txt','r') as f:
    for line in f:
        #result.append(list(line.strip('\n').split(',')))
        lineq += 1

print(lineq)

'''
small = []
for r in result:
    small.append(r[1000:2000])

a = np.array(small,dtype='int')
np.savetxt("small_data.txt", a, fmt='%s',newline='\n')
'''
