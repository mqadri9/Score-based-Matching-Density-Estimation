import sys 
import matplotlib.pyplot as plt
import os 
import numpy as np 

prefixed = [filename for filename in os.listdir('.') if filename.startswith("experiments")]

data = []
for file in prefixed:
    with open(file,'r') as data_file:
        for line in data_file:
            data.append(line.split())

data_formatted = []
for d in data:
    if d[-2] == "True":
        c = 1
    else:
        c = 0
    data_formatted.append([float(d[0]), c])

data_formatted = np.array(data_formatted)
#data_formatted = np.sort(data_formatted, axis=0)

ratio = data_formatted[:, 0]
L = np.arange(0, 1100, 150)

pc = []
for i in range(1, len(L)):
    ratio_less = np.where(np.logical_and(ratio<L[i] , ratio>L[i-1]))    
    d = data_formatted[ratio_less]
    dc = np.count_nonzero(d[:, 1])
    pc.append(dc/len(d))

ratio_less = np.where(ratio>L[-1])   
d = data_formatted[ratio_less]
dc = np.count_nonzero(d[:, 1])
pc.append(dc/len(d))


print(len(pc))
print(len(L))
#plt.xticks(L[1:])

plt.plot(L, pc)
plt.xlabel("Ratio of eigenvalues max(eigenval)/min(eigenval)")
plt.ylabel("probability of convergence")
plt.savefig("ratio_vs_convergence.png")