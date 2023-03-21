import torch


def logq(x, mean, cov, n):
    mean = mean.mT
    mean = mean.repeat((n, 1))
    x = x.unsqueeze(dim=1)
    mean = mean.unsqueeze(dim=1)
    return (-1/2) * ((x - mean) @ torch.linalg.inv(cov) @ torch.transpose(x - mean, 2, 1))

def gradlogq(x, mean, cov, n):
    mean = mean.mT
    mean = mean.repeat((n, 1))
    x = x.unsqueeze(dim=1)
    mean = mean.unsqueeze(dim=1)
    return (-1) * ((x - mean) @ torch.linalg.inv(cov) ).squeeze()

def hessianlogq(x, mean, cov, n):
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

def _hessian_x(phi, x, dim, n):
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
