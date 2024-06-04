import torch
import numpy as np
import torch.nn as nn
from torch import Tensor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)

def epoch(loader, model, opt=None, lr_scheduler=None):
    total_loss, total_err = 0.,0.
    model.eval() if opt is None else model.train()
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]

    total_err = total_err / len(loader.dataset)
    total_loss = total_loss / len(loader.dataset)

    return 100.*total_err, 100.*total_loss

def epoch_eval(loader, model, opt=None, lr_scheduler=None):
    total_loss, total_err = 0.,0.
    with torch.no_grad():
      for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]

      total_err = total_err / len(loader.dataset)
      total_loss = total_loss / len(loader.dataset)

    return 100.*total_err, 100.*total_loss

def cuda(tensor,device=device):
    if torch.cuda.is_available():
        return tensor.to(device)
    else:
        return tensor
    
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
