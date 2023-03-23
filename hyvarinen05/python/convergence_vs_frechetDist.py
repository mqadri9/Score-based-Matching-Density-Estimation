import numpy as np
import sys
import matplotlib.pyplot as plt
#np.random.seed(1)
import torch
from torch.optim import Adam
import random 
from helpers import *
import matplotlib.pyplot as plt
from IPython import display
from IPython.display import clear_output
import scipy.stats
from scipy.linalg import sqrtm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fo = open("experiments.txt", "a")

def frechet_distance(mean_true, cov_true, mean_estimated, cov_estimated):
    ssdiff = np.sum((mean_true - mean_estimated)**2.0)
    covmean = sqrtm(cov_true.dot(cov_estimated))
    return ssdiff + np.trace(cov_true + cov_estimated - 2.0 * covmean)

dim = 2
n = 1000
learning_rate = 0.01
pi = torch.from_numpy(np.array([np.pi]))
max_steps = 200000
max_mean = 17


# init our target distribution params (mean and covariance)

mean_true = np.array([random.randint(1, max_mean) for x in range(dim)])
A = np.random.rand(dim, dim)
cov_true = np.dot(A, A.transpose())


cov_true_tensor = torch.from_numpy(cov_true).to(device)
mean_true_tensor = torch.from_numpy(mean_true).to(device)

# sample from distribution. The objective is to estimate the parameters of the original gaussian
# from these samples 
samples_from_target = np.random.multivariate_normal(mean_true, cov_true, size=n)
x = torch.from_numpy(samples_from_target).double().requires_grad_().to(device)

# define training parameters and setup the optimizer
L0 = torch.triu(torch.rand(dim, dim))
L = torch.nn.Parameter(L0.to(device).double()).requires_grad_()

mean = torch.nn.Parameter(torch.zeros(dim,1).to(device).double()).requires_grad_()
params_to_train = [L, mean]

optimizer = Adam(params_to_train, learning_rate)

cov_estimated = torch.matmul(L.mT, L).detach().cpu().numpy()
mean_estimated = mean.detach().cpu().numpy().squeeze()

frechet_dist = frechet_distance(mean_true, cov_true, mean_estimated, cov_estimated)
eval, evec = np.linalg.eig(cov_true)

# Note that this loss requires the hessian which is expensive to compute
def scoreMatchingLoss(grad, hessian):
    trace = hessian.diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=1).to(device)
    grad = torch.linalg.norm(grad, dim=1)
    loss = torch.mean( trace + 1/2 * grad**2)
    return loss 

verbose = False
for t in range(max_steps):
    optimizer.zero_grad()
    cov = torch.matmul(L.mT, L)
    q = logq(x, mean, cov, n)
    phi = gradlogq(x, mean, cov, n)
    gradPhi = hessianlogq(x, mean, cov, n)
    loss = scoreMatchingLoss(phi, gradPhi)
    loss.backward()
    optimizer.step()        
    if (t%1000 == 0):
        matrix_norm_err = torch.linalg.norm(cov_true_tensor - cov)
        mean_norm_err = torch.linalg.norm(mean_true_tensor - mean.squeeze())
        #clear_output(wait=True)
        if verbose:
            print("t={} loss={}".format(t, loss))
            print("cov_true {}".format(cov_true), "cov_estimated {}".format(cov))
            print("\n")
            print("mean_true {}".format(mean_true), "mean_estimated {}".format(mean))
            print("matrix norm error {}".format(matrix_norm_err))
            print("mean norm error {}".format(mean_norm_err))
        cov_estimated = torch.matmul(L.mT, L).detach().cpu().numpy()
        mean_estimated = mean.detach().cpu().numpy().squeeze()
        if matrix_norm_err < 1 and mean_norm_err < 0.4:       
            break

if t < max_steps-10:
    converged = True
else:
    converged = False 

fo.write("{} {} {} {} {} {}\n".format(np.max(eval)/np.min(eval), eval[0], eval[1], frechet_dist, converged, t))
fo.close()

