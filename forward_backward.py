import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd




# Anderson acceleration for fully connected architectures

def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, H = x0.shape
    X = torch.zeros(bsz, m, H, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, H, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.linalg.lstsq(H[:,:n+1,:n+1],y[:,:n+1])[0][:, 1:n+1, 0]
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:,k%m].view_as(x0), res

class DEQFixedPoint(nn.Module):
    def __init__(self, f,out_dim, solver,**kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs
        self.out_dim = out_dim


    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            bsz = x.shape[0]
            z, self.forward_res = self.solver(lambda z : self.f(z, x), torch.ones(bsz,self.out_dim,device=x.device), **self.kwargs)
        z = self.f(z,x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0,x)
        def backward_hook(grad):
            self.g, self.backward_res = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return self.g

        z.register_hook(backward_hook)
        return z
  
  
  
def anderson_conv(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.linalg.lstsq(H[:,:n+1,:n+1],y[:,:n+1])[0][:, 1:n+1, 0]
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:,k%m].view_as(x0), res
  
class DEQFixedPoint_conv(nn.Module):
    def __init__(self, f,out_chan, solver,**kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs
        self.out_chan = out_chan


    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            bsz = x.shape[0]
            h = x.shape[2]
            w = x.shape[3]
            z, self.forward_res = self.solver(lambda z : self.f(z, x), torch.ones(bsz,self.out_chan,h,w,device=x.device), **self.kwargs)
        z = self.f(z,x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0,x)
        def backward_hook(grad):
            self.g, self.backward_res = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return self.g

        z.register_hook(backward_hook)
        return z