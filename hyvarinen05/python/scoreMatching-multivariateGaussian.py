import numpy as np
import sys
import matplotlib.pyplot as plt
#np.random.seed(1)
import torch
from torch.optim import Adam
import random 
from helpers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dim = 2
n = 2000
learning_rate = 0.1
pi = torch.from_numpy(np.array([np.pi]))


# Generate data from a gaussian with mean = 0 and std = 3

mean_true = np.array([random.randint(1, 10) for x in range(dim)])

A = np.random.uniform(low=0.1, high=3, size=(dim, dim))
cov_true = np.dot(A, A.transpose())

x = torch.from_numpy(np.random.multivariate_normal(mean_true, cov_true, size=n)).double().requires_grad_().to(device)

cov_true_tensor = torch.from_numpy(cov_true).to(device)
mean_true_tensor = torch.from_numpy(mean_true).to(device)


#L0 = torch.from_numpy(A.transpose())
#mean = torch.nn.Parameter(torch.from_numpy(mean_true).unsqueeze(dim=1).to(device).double()).requires_grad_()

L0 = torch.triu(torch.rand(dim, dim))
L = torch.nn.Parameter(L0.to(device).double()).requires_grad_()
mean = torch.nn.Parameter(torch.zeros(dim,1).to(device).double()).requires_grad_()

print(mean.shape)
params_to_train = [L, mean]
optimizer = Adam(params_to_train, learning_rate)

# Note that this loss requires the hessian which is expensive to compute
def scoreMatchingLoss(x, grad, hessian):
    trace = hessian.diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=1).to(device)
    grad = torch.linalg.norm(grad, dim=1)
    loss = torch.mean( trace + 1/2 * grad**2)
    return loss 

cov = torch.matmul(L.mT, L)
q = logq(x, mean, cov, n).squeeze() 

for t in range(400000):
    optimizer.zero_grad()
    cov = torch.matmul(L.mT, L)
    q = logq(x, mean, cov, n)   
    phi = gradlogq(x, mean, cov, n)
    gradPhi = hessianlogq(x, mean, cov, n)
    loss = scoreMatchingLoss(x, phi, gradPhi)
    loss.backward()
    optimizer.step()
    if (t%1000 == 0):
        print("t={} loss={}".format(t, loss))
    if (t%1000 == 0):
        print("cov_true {}".format(cov_true), "cov_estimated {}".format(cov))
        print("\n")
        print("mean_true {}".format(mean_true), "mean_estimated {}".format(mean))
        matrix_norm_err = torch.linalg.norm(cov_true_tensor - cov)
        mean_norm_err = torch.linalg.norm(mean_true_tensor - mean.squeeze())
        print("matrix norm error {}".format(matrix_norm_err))
        print("mean norm error {}".format(mean_norm_err))
        if matrix_norm_err < 1 and mean_norm_err < 0.1:
            break

print("cov_true {}".format(cov_true), "cov_estimated {}".format(cov))
print("\n")
print("mean_true {}".format(mean_true), "mean_estimated {}".format(mean))

