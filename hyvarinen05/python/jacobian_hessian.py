import torch


def jacobian(y, x, create_graph=False):      
    jac = []                                                                      
    flat_y = y.reshape(-1)                                                             
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):                                                              
        grad_y[i] = 1.                                                                 
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        print(grad_x)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                      
def hessian(y, x):                                                                                    
    return jacobian(jacobian(y, x, create_graph=True), x)                                             
                                                                                                      
def f(x): 
    coeffs = torch.Tensor([[1, 0, 0, 4], 
                           [1, 2, 0, 0], 
                           [0, 2, 3, 0], 
                           [0, 2, 0, 4]])
    return coeffs @ ( x * x)

x = torch.ones(4, requires_grad=True)
#jacobian(f(x), x)   

#print("========================================") 
hessian(f(x), x)                                     
#print(jacobian(f(x), x).shape)                                                                              
#print(hessian(f(x), x).shape)    