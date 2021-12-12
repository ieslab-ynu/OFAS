"""prepare CIFAR and SVHN
"""

from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


crop_size = 32
padding = 4


def prepare_train_data(dataset='cifar10', batch_size=64,
                       shuffle=True, num_workers=4):

    if 'cifar' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(crop_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        '''
         __dict__用法:https://blog.csdn.net/Z609834342/article/details/107202830/
        '''
        trainset = torchvision.datasets.__dict__[dataset.upper()](
            root='/tmp/data',
            train=True,
            download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=0)
    else:
        train_loader = None
    return train_loader


def prepare_test_data(dataset='cifar10', batch_size=64,
                      shuffle=False, num_workers=4):

    if 'cifar' in dataset:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.__dict__[dataset.upper()](root='/tmp/data',
                                               train=False,
                                               download=True,
                                               transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=0)

    else:
        test_loader = None
    return test_loader

# train_loader = prepare_train_data(dataset='cifar10',
#                                   batch_size=256,
#                                   shuffle=True,
#                                   num_workers=0)
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# for step, data in enumerate(train_loader):
#     input, target = data
#     print(input.size(0))