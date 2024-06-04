import data_loader
import subdeq
import subdeq_conv
import train
import torch
import random
import numpy as np


worker = 0

#MNIST FC
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

print('=================================================================================')
print(f'MNIST (dense)')
print('=================================================================================')
print("\n")

train_loader,val_loeader, test_loader = data_loader.mnist_loaders(worker,test_batch_size=128*4)

print('=================================================================================')
print(f'Subdeq (PowerscaleTanh)')
print('=================================================================================')
print("\n")

subdeq_tanhshift= subdeq.subdeq_power_scal_tanh()
train.train(subdeq_tanhshift,train_loader,val_loeader, test_loader)

print('=================================================================================')
print(f'Subdeq (Normalized Tanh 1.2)')
print('=================================================================================')
print("\n")

subdeq_tanh = subdeq.subdeq_abs_shifttanh()
train.train(subdeq_tanh,train_loader,val_loeader, test_loader)

print('=================================================================================')
print(f'Subdeq (Powerscale Normalized Tanh)')
print('=================================================================================')
print("\n")

subdeq_tanh = subdeq.subdeq_power_scal_tanh(norm=np.inf)
train.train(subdeq_tanh,train_loader,val_loeader, test_loader)

#MNIST conv

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

print('=================================================================================')
print(f'MNIST (Convolutional)')
print('=================================================================================')
print("\n")

print('=================================================================================')
print(f'Subdeq (PowerscaleTanh)')
print('=================================================================================')
print("\n")

subdeq1 = subdeq_conv.subdeq_power_scal_tanh(chan=32,pool=6,n=28,input_chan=1)
train.train(subdeq1,train_loader,val_loeader, test_loader,max_epochs=40)

print('=================================================================================')
print(f'Subdeq (Normalized Tanh 1.2)')
print('=================================================================================')
print("\n")

subdeq1 = subdeq_conv.subdeq_abs_shifttanh(chan=32,pool=6,n=28,input_chan=1)
train.train(subdeq1,train_loader,val_loeader, test_loader,max_epochs=40)

print('=================================================================================')
print(f'Subdeq (Powerscale Normalized Tanh)')
print('=================================================================================')
print("\n")

subdeq1 = subdeq_conv.subdeq_power_scal_tanh(chan=32,pool=6,n=28,input_chan=1,norm=np.inf)
train.train(subdeq1,train_loader,val_loeader, test_loader,max_epochs=40)

#CIFAR-10 conv

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

train_loader,val_loeader, test_loader = data_loader.cifar_loaders(worker,test_batch_size=128*4)

print('=================================================================================')
print(f'CIFAR-10 (Convolutional)')
print('=================================================================================')
print("\n")

print('=================================================================================')
print(f'Subdeq (PowerscaleTanh)')
print('=================================================================================')
print("\n")

subdeq1 = subdeq_conv.subdeq_power_scal_tanh(chan=48,pool=8,n=32,input_chan=3)
train.train(subdeq1,train_loader,val_loeader, test_loader,max_epochs=40)

print('=================================================================================')
print(f'Subdeq (Normalized Tanh 1.2)')
print('=================================================================================')
print("\n")

subdeq1 = subdeq_conv.subdeq_abs_shifttanh(chan=48,pool=8,n=32,input_chan=3)
train.train(subdeq1,train_loader,val_loeader, test_loader,max_epochs=40)

print('=================================================================================')
print(f'Subdeq (Powerscale Normalized Tanh)')
print('=================================================================================')
print("\n")

subdeq1 = subdeq_conv.subdeq_power_scal_tanh(chan=48,pool=8,n=32,input_chan=3)
train.train(subdeq1,train_loader,val_loeader, test_loader,max_epochs=40,norm=np.inf)

#SVHN conv

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

train_loader,val_loeader, test_loader = data_loader.svhn_loaders(worker, test_batch_size=128*4)

print('=================================================================================')
print(f'SVHN (Convolutional)')
print('=================================================================================')
print("\n")

print('=================================================================================')
print(f'Subdeq (PowerscaleTanh)')
print('=================================================================================')
print("\n")

subdeq1 = subdeq_conv.subdeq_power_scal_tanh(chan=48,pool=8,n=32,input_chan=3)
train.train(subdeq1,train_loader,val_loeader, test_loader,max_epochs=40)

print('=================================================================================')
print(f'Subdeq (Normalized Tanh 1.2)')
print('=================================================================================')
print("\n")

subdeq1 = subdeq_conv.subdeq_abs_shifttanh(chan=48,pool=8,n=32,input_chan=3)
train.train(subdeq1,train_loader,val_loeader, test_loader,max_epochs=40)

print('=================================================================================')
print(f'Subdeq (Powerscale Normalized Tanh)')
print('=================================================================================')
print("\n")

subdeq1 = subdeq_conv.subdeq_power_scal_tanh(chan=48,pool=8,n=32,input_chan=3,norm=np.inf)
train.train(subdeq1,train_loader,val_loeader, test_loader,max_epochs=40)





