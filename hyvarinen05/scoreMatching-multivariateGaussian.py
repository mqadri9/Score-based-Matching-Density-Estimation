import numpy as np
import sys
import matplotlib.pyplot as plt
#np.random.seed(1)
import torch
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dim = 2
n = 3000
learning_rate = 0.01
pi = torch.from_numpy(np.array([np.pi]))

def logq(x, mean, cov):
    mean = mean.mT
    mean = mean.repeat((n, 1))
    x = x.unsqueeze(dim=1)
    mean = mean.unsqueeze(dim=1)
    return (-1/2) * ((x - mean) @ torch.linalg.inv(cov) @ torch.transpose(x - mean, 2, 1))

def gradlogq(x, mean, cov):
    mean = mean.mT
    mean = mean.repeat((n, 1))
    x = x.unsqueeze(dim=1)
    mean = mean.unsqueeze(dim=1)
    return (-1) * ((x - mean) @ torch.linalg.inv(cov) ).squeeze()

def hessianlogq(x, mean, cov):
    mean = mean.mT
    mean = mean.repeat((n, 1))
    x = x.unsqueeze(dim=1)
    mean = mean.unsqueeze(dim=1)
    res = (-1) * torch.linalg.inv(cov)
    return res.repeat((n, 1, 1)) 

def _grad_x(y, x, create_graph=False):
    grad_y = torch.ones_like(y)
    grad_x, = torch.autograd.grad(y, x, grad_y, retain_graph=True, create_graph=create_graph)
    return grad_x

def _grad_x_origin(y, x, create_graph=False):      
    jac = []
    grad_y = torch.zeros_like(y)
    for i in range(len(grad_y)):                                                              
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(y, x, grad_y, retain_graph=True, create_graph=create_graph)
        print(grad_x)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape).requires_grad_()                                                

def _hessian_x(phi, x):
    flat_y = phi.reshape(-1)
    flat_x = x
    grad_y = torch.zeros_like(flat_y)
    num_f = x.shape[0]
    hess = torch.zeros((len(flat_y), dim))
    for i in range(len(flat_y)//num_f): 
        for j in range(i, len(flat_y), dim):
            grad_y[j] = 1.        
        grad_x, = torch.autograd.grad(flat_y, flat_x, grad_y, retain_graph=True, create_graph=False)
        c = 0 
        for j in range(i, len(flat_y), dim):
            hess[j] = grad_x[c]
            c+=1                                                           
        grad_y = torch.zeros_like(flat_y) 
    return hess.reshape(n, dim, -1)

def _hessian_x_origin(phi, x):                                                                                    
    return _grad_x_origin(phi, x)  

# Generate data from a gaussian with mean = 0 and std = 3

mean_true = np.array([1, 1])
cov_true = np.array([[1, 0.1], 
                    [0.1, 1]])

x = torch.from_numpy(np.random.multivariate_normal(mean_true, cov_true, size=n)).double().requires_grad_().to(device)
#x = torch.from_numpy(np.array([[0.2, 0.2], [0, 0], [0.3, 0.3], [0.5, 0.5]])).double().requires_grad_().to(device)
print(x.shape)
L0 = torch.triu(torch.rand(dim, dim))
#L0 = torch.triu(torch.from_numpy(cov_true))
L = torch.nn.Parameter(L0.to(device).double().requires_grad_()).requires_grad_()
mean = torch.nn.Parameter(torch.zeros(dim,1).to(device).double().requires_grad_()).requires_grad_()
params_to_train = [L, mean]
optimizer = Adam(params_to_train, learning_rate)

# Note that this loss requires the hessian which is expensive to compute
def scoreMatchingLoss(x, grad, hessian):
    trace = hessian.diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=1).to(device)
    #print(trace.shape)
    grad = torch.linalg.norm(grad, dim=1)
    #print(grad.shape)
    loss = torch.mean( trace + 1/2 * grad**2)
    return loss 

cov = torch.matmul(L.mT, L)
q = logq(x, mean, cov).squeeze() 
#print(q)
#print(gradlogq(x, mean, cov))
#print(hessianlogq(x, mean, cov))
#print(q.shape)
#print("===========================================")


gra = _grad_x(q, x, create_graph=True)
gradPhi = _hessian_x(gra, x)
#print(gradPhi)
#print(gradPhi.shape)
#sys.exit()
for t in range(1000):
    optimizer.zero_grad()
    cov = torch.matmul(L.mT, L)
    q = logq(x, mean, cov)   
    #phi = _grad_x(q, x, create_graph=True)
    #gradPhi =_hessian_x(phi, x)
    phi = gradlogq(x, mean, cov)
    #print(phi.shape)
    gradPhi = hessianlogq(x, mean, cov)
    #print(gradPhi.shape)
    loss = scoreMatchingLoss(x, phi, gradPhi)
    loss.backward()
    optimizer.step()
    if (t%100 == 0):
        print(loss)

print(cov, mean)

