
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import forward_backward



class Subhomlayerfc_shift(nn.Module):
    """
    In_dim = input dimension
    Out _dim = dimesion of the embedding
    """
    def __init__(self,out_dim,in_dim=28*28,norm=np.inf,shift=1.2):
        super().__init__()
        self.lin1 = nn.Linear(in_dim,out_dim,bias=True)
        self.lin0 = nn.Linear(out_dim,out_dim,bias=False)
        self.gamma = nn.Linear(out_dim, 1, bias=False).weight
        self.beta = nn.Linear(out_dim, 1, bias=False).weight
        self.norm = norm
        self.shift = shift
        self.lin1.weight.data.normal_(0, 0.01)
        self.lin0.weight.data.normal_(0, 0.01)


    def forward(self, z, x):
        y = F.relu(self.lin1(x))
        z = torch.tanh(self.lin0(z)) + y + torch.tensor(self.shift)
        if self.norm:
            z = torch.nn.functional.normalize(z,dim =1,p=self.norm)
        return z
      
class Subhomlayerfc_power_scal_tanh(nn.Module):
    """
    In_dim = input dimension
    Out _dim = dimesion of the embedding
    """
    def __init__(self,out_dim,in_dim=28*28,norm=None,pow=1.2):
        super().__init__()
        self.lin1 = nn.Linear(in_dim,out_dim,bias=True)
        self.lin0 = nn.Linear(out_dim,out_dim,bias=False)
        self.gamma = nn.Linear(out_dim, 1, bias=False).weight
        self.beta = nn.Linear(out_dim, 1, bias=False).weight
        self.norm = norm
        self.pow = pow
        self.lin1.weight.data.normal_(0, 0.01)
        self.lin0.weight.data = torch.abs(self.lin0.weight.data.normal_(0, 0.01))


    def forward(self, z, x):
        y = F.relu(self.lin1(x))
        if self.norm:
            self.lin0.weight.data = torch.abs(self.lin0.weight.data)
            z = torch.pow(torch.tanh(self.lin0(z))+ torch.tensor(1.01),self.pow)+y
            z = torch.nn.functional.normalize(z,dim =1,p=self.norm)
        else:
            z = torch.pow(torch.tanh(self.lin0(z))+ torch.tensor(1.01),self.pow)+y
        return z

class Subhomlayerfc_abs_shifttanh(nn.Module):
    """
    In_dim = input dimension
    Out _dim = dimesion of the embedding
    """
    def __init__(self,out_dim,in_dim=28*28,norm=np.inf):
        super().__init__()
        self.lin1 = nn.Linear(in_dim,out_dim,bias=True)
        self.lin0 = nn.Linear(out_dim,out_dim,bias=False)
        self.gamma = nn.Linear(out_dim, 1, bias=False).weight
        self.beta = nn.Linear(out_dim, 1, bias=False).weight
        self.norm = norm
        self.lin1.weight.data.normal_(0, 0.01)
        self.lin0.weight.data = torch.abs(self.lin0.weight.data.normal_(0, 0.01))


    def forward(self, z, x):
        y = F.relu(self.lin1(x))
        self.lin0.weight.data = torch.abs(self.lin0.weight.data)
        z = torch.tanh(self.lin0(z) + y) + torch.tensor(1.2)
        z = torch.nn.functional.normalize(z,dim =1,p=self.norm)
        return z
    
    
    
def Subdeq_shift(out = 87,norm=None,shift=1.2):
    f = Subhomlayerfc_shift(out_dim=out,norm=norm,shift=shift)
    model = nn.Sequential(nn.Flatten(),
        forward_backward.DEQFixedPoint(f,solver=forward_backward.anderson,out_dim=out, tol=1e-3, max_iter=30, m=5),
        nn.BatchNorm1d(out),
        nn.Linear(out,10))
    
    return model


def subdeq_abs_shifttanh(out = 87):
    f = Subhomlayerfc_abs_shifttanh(out_dim=out)
    model = nn.Sequential(nn.Flatten(),
        forward_backward.DEQFixedPoint(f,solver=forward_backward.anderson,out_dim=out, tol=1e-3, max_iter=30, m=5),
        nn.BatchNorm1d(out),
        nn.Linear(out,10))
    
    return model

def subdeq_power_scal_tanh(out = 87,norm=None,pow=0.999):
    f = Subhomlayerfc_power_scal_tanh(out_dim=out,norm=norm,pow=pow)
    model = nn.Sequential(nn.Flatten(),
        forward_backward.DEQFixedPoint(f,solver=forward_backward.anderson,out_dim=out, tol=1e-3, max_iter=30, m=5),
        nn.BatchNorm1d(out),
        nn.Linear(out,10))
    
    return model
