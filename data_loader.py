import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import ssl
import pandas as pd
import os

# The following line of code is for disable the ssl certeficate  
# We need it to download cifar-10
ssl._create_default_https_context = ssl._create_unverified_context

#imagnet
use_cuda = torch.cuda.is_available()

def generate_dataloader(data, name, transform):
    if data is None:
        return None

    # Read image files to pytorch dataset using ImageFolder, a generic data
    # loader where images are in format root/label/filename
    # See https://pytorch.org/vision/stable/datasets.html
    if transform is None:
        dataset = dset.ImageFolder(data, transform=transforms.ToTensor())
    else:
        dataset = dset.ImageFolder(data, transform=transform)

    # Set options for device
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}

    # Wrap image dataset (defined above) in dataloader
    batch_size = 64
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                        shuffle=(name=="train"),
                        **kwargs)

    return dataloader

def imagnet_loader():

    #val_data = pd.read_csv(f'{VALID_DIR}/val_annotations.txt',
    #                   sep='\t',#
    #                   header=None,
    #                   names=['File', 'Class', 'X', 'Y', 'H', 'W'])
    
    # Create separate validation subfolders for the validation images based on
    # their labels indicated in the val_annotations txt file



    DATA_DIR = 'tiny-imagenet-200/'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VALID_DIR = os.path.join(DATA_DIR, 'val')

    val_data = pd.read_csv(f'{VALID_DIR}/val_annotations.txt',
                       sep='\t',
                       header=None,
                       names=['File', 'Class', 'X', 'Y', 'H', 'W'])

    val_img_dir = os.path.join(VALID_DIR, 'images')

    # Open and read val annotations text file
    fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
    data = fp.readlines()

    # Create dictionary to store img filename (word 0) and corresponding
    # label (word 1) for every line in the txt file (as key value pair)
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create subfolders (if not present) for validation images based on label ,
    # and move images into the respective folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(val_img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(val_img_dir, img)):
            os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))

    # Save class names (for corresponding labels) as dict from words.txt file
    class_to_name_dict = dict()
    fp = open(os.path.join(DATA_DIR, 'words.txt'), 'r')
    data = fp.readlines()
    for line in data:
        words = line.strip('\n').split('\t')
        class_to_name_dict[words[0]] = words[1].split(',')[0]
    fp.close()

    preprocess_transform = transforms.Compose([
                #transforms.Resize(128), # Resize images to 256 x 256
                #transforms.CenterCrop(120), # Center crop image
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # Converting cropped images to tensors
                transforms.Normalize(mean=[0.4820, 0.4479, 0.3965],std=[0.2623, 0.2539, 0.2671]) #
    ])
    train_loader = generate_dataloader(TRAIN_DIR, "train",
                                  transform=preprocess_transform)
    val_loader = generate_dataloader(val_img_dir, "val",
                                 transform=preprocess_transform)
    return train_loader, val_loader



def mnist_loaders(worker,test_batch_size=None,train_batch_size=128):
    if test_batch_size is None:
        test_batch_size = train_batch_size

    train_loader = dset.ImageNet('data',
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    train_loader, val_loeader = torch.utils.data.random_split(train_loader, [50000, 10000])


    train_loader = torch.utils.data.DataLoader(
       train_loader,
        batch_size=train_batch_size,
        shuffle=True)

    val_loeader = torch.utils.data.DataLoader(
       val_loeader,
        batch_size=train_batch_size,
        shuffle=True)


    test_loader = torch.utils.data.DataLoader(
        dset.MNIST('data',
                   train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
        batch_size=test_batch_size,
        shuffle=False)
    return train_loader,val_loeader, test_loader




# Minst
def mnist_loaders(worker,test_batch_size=None,train_batch_size=128):
    if test_batch_size is None:
        test_batch_size = train_batch_size

    train_loader = dset.MNIST('data',
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    train_loader, val_loeader = torch.utils.data.random_split(train_loader, [50000, 10000])


    train_loader = torch.utils.data.DataLoader(
       train_loader,
        batch_size=train_batch_size,
        shuffle=True)

    val_loeader = torch.utils.data.DataLoader(
       val_loeader,
        batch_size=train_batch_size,
        shuffle=True)


    test_loader = torch.utils.data.DataLoader(
        dset.MNIST('data',
                   train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
        batch_size=test_batch_size,
        shuffle=False)
    return train_loader,val_loeader, test_loader

# svhn
def svhn_loaders(worker, test_batch_size=None,train_batch_size=128):
  if test_batch_size is None:
        test_batch_size = train_batch_size

  normalize = transforms.Normalize(mean=[0.4371, 0.4433, 0.4726],
                                      std=[0.1971, 0.2001, 0.1962])


  train_loader = dset.SVHN(root='data', split='train', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize
                    ]))


  train_loader, val_loeader = torch.utils.data.random_split(train_loader, [50000, 23257])

  train_loader = torch.utils.data.DataLoader(train_loader,
            batch_size=train_batch_size, shuffle=True,num_workers=worker)

  val_loeader = torch.utils.data.DataLoader(val_loeader,
            batch_size=train_batch_size, shuffle=False,num_workers=worker)

  test_loader = dset.SVHN(root='data', split='test', download=True,transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
                  ]))
  test_loader = torch.utils.data.DataLoader(test_loader,batch_size=test_batch_size, shuffle=False,num_workers=worker)

  return train_loader,val_loeader, test_loader

# Cifar
def cifar_loaders(worker,test_batch_size=None,train_batch_size=128):
    
    if test_batch_size is None:
        test_batch_size = train_batch_size

    normalize = transforms.Normalize(mean=[0.4250, 0.4152, 0.3842],
                                         std=[0.2827, 0.2777, 0.2843])



    transforms_list = [transforms.ToTensor(),
                           normalize]




    train_dset = dset.CIFAR10('data',
                              train=True,
                              download=True,
                              transform=transforms.Compose(transforms_list))

    train_loader, val_loeader = torch.utils.data.random_split(train_dset, [42000, 8000])

    test_dset = dset.CIFAR10('data',
                             train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 normalize
                             ]))


    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=train_batch_size,
                                              shuffle=True, pin_memory=True,num_workers=worker)

    val_loeader = torch.utils.data.DataLoader(val_loeader, batch_size=train_batch_size,
                                              shuffle=False, pin_memory=True,num_workers=worker)

    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=test_batch_size,
                                             shuffle=False, pin_memory=True,num_workers=worker)

    return train_loader,val_loeader,test_loader


# Cifar
def cifar100_loaders(train_batch_size, test_batch_size=None):
    if test_batch_size is None:
        test_batch_size = train_batch_size

    normalize = transforms.Normalize(mean = [0.5071, 0.4865, 0.4409],
                                         std= [0.2668, 0.2560, 0.2756])



    transforms_list = [transforms.ToTensor(),
                           normalize
                       ]




    train_dset = dset.CIFAR100('./',
                              train=True,
                              download=True,
                              transform=transforms.Compose(transforms_list)
                              )


    test_dset = dset.CIFAR100('./',
                             train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 normalize])
    )


    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=train_batch_size,
                                              shuffle=True, pin_memory=True,num_workers=2)

   # test_loader = torch.utils.data.DataLoader(test_loader, batch_size=train_batch_size,
    #                                          shuffle=False, pin_memory=True,num_workers=2)

    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=test_batch_size,
                                             shuffle=False, pin_memory=True,num_workers=2)

    #return train_loader,test_loader,test_loader
    return train_loader,test_loader
