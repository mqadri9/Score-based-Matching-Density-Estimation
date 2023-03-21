import numpy as np
import sys
import matplotlib.pyplot as plt
#np.random.seed(1)
import torch
from torch.optim import Adam


n = 1000
learning_rate = 0.01
pi = torch.from_numpy(np.array([np.pi]))

# Generate data from a gaussian with mean = 0 and std = 3
x = torch.from_numpy(np.random.normal(loc=0, scale=3, size=n))

def NLLloss(data, std, mean):
    loss = torch.mean( (1/2) * ((x - mean)/std)**2 + torch.log(torch.sqrt(2*pi) * std)  )
    return loss 

std = torch.nn.Parameter(torch.ones(1,1)).requires_grad_()
mean = torch.nn.Parameter(torch.ones(1,1)).requires_grad_()
params_to_train = [std, mean]
optimizer = Adam(params_to_train, learning_rate)

for t in range(1000):
    optimizer.zero_grad()
    loss = NLLloss(x, std, mean)
    loss.backward()
    optimizer.step()

print(std, mean)