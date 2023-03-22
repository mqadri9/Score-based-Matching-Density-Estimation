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
    if d[1] == "True":
        c = 1
    else:
        c = 0
    data_formatted.append([float(d[0]), c])

data_formatted = np.array(data_formatted)
#data_formatted = np.sort(data_formatted, axis=0)

frechet = data_formatted[:, 0]
L = [1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425]

pc = []
for i in range(1, len(L)):
    frechet_less = np.where(np.logical_and(frechet<L[i] , frechet>L[i-1]))    
    d = data_formatted[frechet_less]
    dc = np.count_nonzero(d[:, 1])
    pc.append(dc/len(d))

print(len(pc))
print(len(L))
plt.xticks(L[1:])

plt.plot(L[1:], pc)
plt.xlabel("Frechet Distance")
plt.ylabel("probability of convergence")
plt.savefig("frechet_vs_convergence.png")