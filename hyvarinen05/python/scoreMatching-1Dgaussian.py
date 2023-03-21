import numpy as np
import sys
import matplotlib.pyplot as plt
#np.random.seed(1)
import torch
from torch.optim import Adam


n = 1000
learning_rate = 0.1
pi = torch.from_numpy(np.array([np.pi]))

def logq(x, mean, std):
    return (-1/2) * ((x - mean)/std)**2

def _grad_x(logq, mean, std, x):
    x.requires_grad_(True)
    y = logq(x, mean, std) 
    
    # Since logq is a function from R x R^2 to R. y will have the same shape as x (n , 1)
    
    d_output = torch.ones_like(y, requires_grad=False, device=y.device)
    gradients = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return gradients

def _hessian_x(logq, mean, std, x):
    x.requires_grad_(True)
    y = _grad_x(logq, mean, std, x) 
    d_output = torch.ones_like(y, requires_grad=False, device=y.device)
    gradients = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return gradients

# Generate data from a gaussian with mean = 0 and std = 3
x = torch.from_numpy(np.random.normal(loc=0, scale=3, size=n))

std = torch.nn.Parameter(torch.ones(1,1)).requires_grad_()
mean = torch.nn.Parameter(torch.ones(1,1)).requires_grad_()
params_to_train = [std, mean]
optimizer = Adam(params_to_train, learning_rate)

# Note that this loss requires the hessian which is expensive to compute
def scoreMatchingLoss(x, grad, hessian):
    loss = torch.mean( hessian + 1/2 * grad**2)
    return loss 

for t in range(1000):
    optimizer.zero_grad()
    grad = _grad_x(logq, mean, std, x)
    hessian =_hessian_x(logq, mean, std, grad)
    loss = scoreMatchingLoss(x, grad, hessian)
    loss.backward()
    optimizer.step()

print(std, mean)

