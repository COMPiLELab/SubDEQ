import torch
import numpy as np
import torch.optim as optim
import time
import datetime
from copy import deepcopy
from torch.nn.modules.container import ModuleList, ModuleDict, Module
from torch.nn.parameter import Parameter
import random
import utils
import torch.nn as nn

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def train(model,train_loader,val_loeader,test_loader,runs = 5,max_epochs = 30):
    start = time.time()

    best_model = []
    err_ep = np.zeros((runs,max_epochs))
    err_test = []

    for run in range(runs):
      err_val_history = np.array([])

      opt = optim.Adam(model.parameters(),weight_decay=1e-5, lr=1e-3)

      lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs*len(train_loader), eta_min=1e-6)

      model.apply(weight_reset)
      model = utils.cuda(model)
      print("# Parmeters: ", sum(a.numel() for a in model.parameters()))
      print('==================================== Run {} ===================================='.format(run+1))
      for i in range(1,max_epochs+1):
        start_train = time.time()
        ep = utils.epoch(train_loader, model, opt, lr_scheduler)

        train_error = ep[0]
        train_loss = ep[1]

        temp = datetime.timedelta(seconds=(time.time() - start_train))
        print('==================================== Epoch {} ===================================='.format(i))
        print("Single epoch train time: {}".format(temp))
        print(f"\nTrain error: {train_error}, Train loss: {train_loss}")

        start_val = time.time()

        ep_val = utils.epoch(val_loeader, model)

        val_error = ep_val[0]
        val_loss = ep_val[1]

        temp = datetime.timedelta(seconds=(time.time() - start_val))

        print("\nTest evaluation time: {}".format(temp))
        print(f"\nValidation error: {val_error}, Validation loss: {val_loss}")
        print("\n")

        err_val_history = np.append(err_val_history,val_error)

        if err_val_history.min()==val_error:
          best_model = deepcopy(model)


      temp = datetime.timedelta(seconds=(time.time() - start))
      print(f"Total train time: {temp} of the run number {run+1}")
      err_test.append(utils.epoch(test_loader, best_model)[0])
      print(f"Test err:{err_test[run]}")
      err_ep[run] = err_val_history
      print("\n")

  

