import torch
import math
import os
import sys
import torchvision
from torch.utils.data import random_split
from torchvision import datasets, transforms


def load_dataset(dataset):

    if not os.path.exists('./data'):
        os.mkdir('./data')
        
    if dataset == 'mnist':
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'cifar10':
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'svhn':
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_set = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    else:
        raise ValueError('Undefined Dataset.')

    return train_set, test_set


def load_data(dataset, batch_size, seed=42):

    train_set, test_set = load_dataset(dataset)

    torch.manual_seed(seed)
    
    val_size = int(len(train_set) * 0.2)
    train_set, val_set = random_split(train_set, [len(train_set) - val_size, val_size])

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, valloader, testloader


def load_unlearn_data(dataset, num_unlearn):
    train_set, _ = load_dataset(dataset)

    torch.manual_seed(0)
    res_set, unl_set = random_split(train_set, [len(train_set) - num_unlearn, num_unlearn])

    # unlearnloader = torch.utils.data.DataLoader(unl_set, batch_size=len(unl_set), shuffle=False, num_workers=2)
    # residualloader = torch.utils.data.DataLoader(res_set, batch_size=len(res_set), shuffle=False, num_workers=2)

    return unl_set, res_set


def load_train_data(dataset, batch_size, seed=42):
    torch.manual_seed(seed)
    
    val_size = int(len(dataset) * 0.2)
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader, valloader


def params_to_vec(parameters, grad=False):
    vec = []
    for param in parameters:
        if grad:
            vec.append(param.grad.view(1, -1))
        else:
            vec.append(param.data.view(1, -1))
    return torch.cat(vec, dim=1).squeeze()


def vec_to_params(vec, parameters):
    param = []
    for p in parameters:
        size = p.view(1, -1).size(1)
        param.append(vec[:size].view(p.size()))
        vec = vec[size:]
    return param


def batch_grads_to_vec(parameters):
    vec = []
    for param in parameters:
        # vec.append(param.view(1, -1))
        vec.append(param.reshape(1, -1))
    return torch.cat(vec, dim=1).squeeze()


def batch_vec_to_grads(vec, parameters):
    grads = []
    for param in parameters:
        size = param.view(1, -1).size(1)
        grads.append(vec[:size].view(param.size()))
        vec = vec[size:]
    return grads