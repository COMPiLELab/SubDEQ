import forward_backward
import torch
import torch.fft
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

class Subhomlayer_shifttanh(nn.Module):
    """
    In_dim = input dimension
    Out_dim = dimesion of the embedding
    """
    def __init__(self, n_channels=3, n_inner_channels=3, kernel_size=3,norm=None,shift=1.2):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=True)
        self.conv2 = nn.Conv2d(n_inner_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.norm = norm
        self.shift = shift
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, z, x):
        y = F.relu(self.conv1(x))
        z = torch.tanh(self.conv2(z)) + y + torch.tensor(self.shift)
        if self.norm:
            z = F.normalize(z,dim = 2,p=self.norm)
        return z
    
    
class Subhomlayer_power_scal_tanh(nn.Module):
    """
    In_dim = input dimension
    Out_dim = dimesion of the embedding
    """
    def __init__(self, n_channels=3, n_inner_channels=3, kernel_size=3,norm=None,pow=0.9999):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=True)
        self.conv2 = nn.Conv2d(n_inner_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.norm = norm
        self.pow = pow
        self.conv1.weight.data.normal_(0, 0.01)

    def forward(self, z, x):
        y = F.relu(self.conv1(x))
        if self.norm:
            self.conv2.weight.data = torch.abs(self.conv2.weight.data)
            z = torch.pow(torch.tanh(self.conv2(z))+ torch.tensor(1.01),self.pow)+y
            z = torch.nn.functional.normalize(z,dim = 2,p=self.norm)
        else:
            z = torch.pow(torch.tanh(self.conv2(z))+ torch.tensor(1.01),self.pow)+y
        return z

class Subhomlayer_abs_shifttanh(nn.Module):
    """
    In_dim = input dimension
    Out_dim = dimesion of the embedding
    """
    def __init__(self, n_channels=3, n_inner_channels=3, kernel_size=3,norm=np.inf):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=True)
        self.conv2 = nn.Conv2d(n_inner_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.norm = norm
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data = torch.abs(self.conv2.weight.data.normal_(0, 0.01))

    def forward(self, z, x):
        y = F.relu(self.conv1(x))
        self.conv2.weight.data = torch.abs(self.conv2.weight.data)
        z = torch.tanh(self.conv2(z) + y) + torch.tensor(1.2)
        z = torch.nn.functional.normalize(z,dim = 2,p=self.norm)
        return z
    
  
def subdeq_shifttanh(chan,pool,n,input_chan,out_class=10,norm=None,shift=1.2):
    out = chan * (n // pool) ** 2
    f = Subhomlayer_shifttanh(n_channels=input_chan,n_inner_channels=chan,norm=norm,shift=shift)
    model = nn.Sequential(
        forward_backward.DEQFixedPoint_conv(f,solver=forward_backward.anderson_conv,out_chan=chan, tol=1e-3, max_iter=30, m=5),
        nn.BatchNorm2d(chan),
        nn.AvgPool2d(pool,pool),
        nn.Flatten(),
        nn.Linear(out,out_class))
    
    return model

def subdeq_abs_shifttanh(chan,pool,n,input_chan,out_class=10):
    out = chan * (n // pool) ** 2
    f = Subhomlayer_abs_shifttanh(n_channels=input_chan,n_inner_channels=chan)
    model = nn.Sequential(
        forward_backward.DEQFixedPoint_conv(f,solver=forward_backward.anderson_conv,out_chan=chan, tol=1e-3, max_iter=30, m=5),
        nn.BatchNorm2d(chan),
        nn.AvgPool2d(pool,pool),
        nn.Flatten(),
        nn.Linear(out,out_class))
    
    return model

def subdeq_power_scal_tanh(chan,pool,n,input_chan,out_class=10,norm=None,pow=0.9999):
    out = chan * (n // pool) ** 2
    f = Subhomlayer_power_scal_tanh(n_channels=input_chan,n_inner_channels=chan,norm=norm,pow=pow)
    model = nn.Sequential(
        forward_backward.DEQFixedPoint_conv(f,solver=forward_backward.anderson_conv,out_chan=chan, tol=1e-3, max_iter=30, m=5),
        nn.BatchNorm2d(chan),
        nn.AvgPool2d(pool,pool),
        nn.Flatten(),
        nn.Linear(out,out_class))
    
    return model