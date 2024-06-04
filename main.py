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
print(f'Subdeq (Tanh)')
print('=================================================================================')
print("\n")

subdeq_tanhshift= subdeq.Subdeq_shift()
train.train(subdeq_tanhshift,train_loader,val_loeader, test_loader)

print('=================================================================================')
print(f'Subdeq (Normalized Tanh)')
print('=================================================================================')
print("\n")

subdeq_tanh = subdeq.Subdeq_shift(norm=np.inf,shift=1.603)
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
print(f'Subdeq (Tanh)')
print('=================================================================================')
print("\n")

subdeq1 = subdeq_conv.subdeq_shifttanh(chan=32,pool=6,n=28,input_chan=1)
train.train(subdeq1,train_loader,val_loeader, test_loader,max_epochs=40)

print('=================================================================================')
print(f'Subdeq (Normalized Tanh)')
print('=================================================================================')
print("\n")

subdeq1 = subdeq_conv.subdeq_shifttanh(chan=32,pool=6,n=28,input_chan=1,norm=np.inf,shift=1.603)
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
print(f'Subdeq (Tanh)')
print('=================================================================================')
print("\n")

subdeq1 = subdeq_conv.subdeq_shifttanh(chan=48,pool=8,n=32,input_chan=3)
train.train(subdeq1,train_loader,val_loeader, test_loader,max_epochs=40)

print('=================================================================================')
print(f'Subdeq (Normalized Tanh)')
print('=================================================================================')
print("\n")

subdeq1 = subdeq_conv.subdeq_shifttanh(chan=48,pool=8,n=32,input_chan=3)
train.train(subdeq1,train_loader,val_loeader, test_loader,max_epochs=40,norm=np.inf,shift=1.603)

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
print(f'Subdeq (Tanh)')
print('=================================================================================')
print("\n")

subdeq1 = subdeq_conv.subdeq_shifttanh(chan=48,pool=8,n=32,input_chan=3)
train.train(subdeq1,train_loader,val_loeader, test_loader,max_epochs=40)

print('=================================================================================')
print(f'Subdeq (Normalized Tanh)')
print('=================================================================================')
print("\n")

subdeq1 = subdeq_conv.subdeq_shifttanh(chan=48,pool=8,n=32,input_chan=3,norm=np.inf,shift=1.603)
train.train(subdeq1,train_loader,val_loeader, test_loader,max_epochs=40)

#Tiny ImageNet conv

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

train_loader,val_loader = data_loader.imagnet_loader()

print('=================================================================================')
print(f'Tiny ImageNet (Convolutional)')
print('=================================================================================')
print("\n")

print('=================================================================================')
print(f'Subdeq (Tanh)')
print('=================================================================================')
print("\n")

subdeq1 = subdeq_conv.subdeq_shifttanh(chan=64,pool=8,n=64,input_chan=3,out_class=200)
train.train(subdeq1,train_loader,val_loader, val_loader,max_epochs=50,runs = 3)

print('=================================================================================')
print(f'Subdeq (Normalized Tanh)')
print('=================================================================================')
print("\n")

subdeq1 = subdeq_conv.subdeq_shifttanh(chan=64,pool=8,n=64,input_chan=3,out_class=200,norm=np.inf,shift=1.603)
train.train(subdeq1,train_loader,val_loader, val_loader,max_epochs=50,runs = 3)


