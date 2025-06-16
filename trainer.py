import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import time
import torch.nn.functional as F
import torch.autograd as autograd
import sys
import os
import numpy as np
# from tqdm import tqdm

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from utils_ensemble import AverageMeter, accuracy, test, requires_grad_
from utils_ensemble import Cosine, Magnitude


def PGD(models, inputs, labels, eps):
    steps = 6
    alpha = eps / 3.

    adv = inputs.detach() + torch.FloatTensor(inputs.shape).uniform_(-eps, eps).cuda()
    adv = torch.clamp(adv, 0, 1)
    criterion = nn.CrossEntropyLoss()

    adv.requires_grad = True
    for _ in range(steps):
        #adv.requires_grad_()
        grad_loss = 0
        for i, m in enumerate(models):
            loss = criterion(m(adv), labels)
            grad = autograd.grad(loss, adv, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            grad_loss += Magnitude(grad)

        grad_loss /= 3
        grad_loss.backward()
        sign_grad = adv.grad.data.sign()
        with torch.no_grad():
            adv.data = adv.data + alpha * sign_grad
            adv.data = torch.max(torch.min(adv.data, inputs + eps), inputs - eps)
            adv.data = torch.clamp(adv.data, 0., 1.)

    adv.grad = None
    return adv.detach()


def Naive_Trainer(args, loader: DataLoader, model, criterion, optimizer: Optimizer, epoch: int, device: torch.device, writer=None, scheduler=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    model.train()
    requires_grad_(model, True)

    for i, (inputs, targets) in enumerate(loader):
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        inputs.requires_grad = True

        logits = model(inputs)
        loss = criterion(logits, targets)
        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if args.arch == 'vit':
        #     scheduler.step()

    print('Epoch: ', epoch, 'Loss: ', losses.avg)

    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)

    return losses.avg
