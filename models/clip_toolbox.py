import torch
import sys
import matplotlib.pyplot as plt
import torchvision
import numpy
import copy
import torch.optim as optim
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"



def power_qr(m, x0, device='cpu', n_iters = 20, record = False, x0_out = False, quiet = False, A=None, sort=True): 
    k = x0.shape[0]
    x0.requires_grad_(True)
    if record:
        out = torch.empty(n_iters, k, device = device)
    else:
        out = torch.empty(1, k, device = device)
        r = None
    for nit in range(n_iters):
        loss = (m(x0) ** 2).sum() / 2
        loss.backward()
        with torch.no_grad():
            x = 1*x0 + x0.grad
            shape = x.shape
            x = x.view(k, -1).T
            (q, r) = torch.linalg.qr(x) 
            x0 = q.T.view(shape) # - 7*torch.eye(q.T.shape[0], device=device)
            x0.requires_grad_(True)
            if record:
                sv = (r.diagonal() - 1).abs().sqrt()
                out[nit, :] = sv 
    if not record:
        sv = (r.diagonal() - 1).abs().sqrt()
        # sv,sort_idx = torch.sort(sv, descending=True)
        # x0 = x0[sort_idx]
        out[0, :] = sv 
    else:
        if sort:
            out,_ = torch.sort(out, descending=True)

    if x0_out:
        return (x0.detach(), out)
    else:
        return out


def deflate_model(conv_model_trained, conv_model_clipped, data, SV, clip=1., rand_data=False, epochs=1, early_stop=False, step_size=0.01, quiet=True, device='cuda'):
    loss_list = []
    min_loss = 100000000 # torch.Tensor(float("Inf")) # 100000000

    data_new = data
    for epoch in range(epochs):
        if rand_data:
            data_new = torch.rand_like(data)

        conv_model_trained.zero_grad()

        if isinstance(conv_model_trained, torch.nn.Linear):
            loss = ((conv_model_clipped((clip/SV) * data_new) - conv_model_clipped((clip/SV) * torch.zeros_like(data_new)) - conv_model_trained(data_new) + conv_model_trained(torch.zeros_like(data_new)))**2).sum((1)).mean()
        elif isinstance(conv_model_trained, torch.nn.Conv1d):
            loss = ((conv_model_clipped((clip/SV) * data_new) - conv_model_clipped((clip/SV) * torch.zeros_like(data_new)) - conv_model_trained(data_new) + conv_model_trained(torch.zeros_like(data_new)))**2).sum((1,2)).mean()
        elif isinstance(conv_model_trained, torch.nn.Conv2d):
            loss = ((conv_model_clipped((clip/SV) * data_new) - conv_model_clipped((clip/SV) * torch.zeros_like(data_new)) - conv_model_trained(data_new) + conv_model_trained(torch.zeros_like(data_new)))**2).sum((1,2)).mean()
        else:
            loss = ((conv_model_clipped((clip/SV) * data_new) - conv_model_trained(data_new))**2).sum((1,2)).mean()

        loss.backward()

        loss_list.append(loss.item())
        if not quiet:
            if epoch % 50 == 0:
                print(loss.item())

        if loss.item() < min_loss: 
            min_loss = loss.item()
        else:
            if early_stop:
                print('not decreasing anymore!')
                break

        with torch.no_grad():
            for P, tup in zip(conv_model_trained.parameters(), conv_model_trained.named_parameters()):
                # if tup[0] == 'weight':
                # if tup[0] in ['weight', 'conv.weight', 'bn.weight']:
                if tup[0].split('.')[-1] == 'weight':
                    # print(tup[0])
                    P -= step_size * P.grad

        # del loss, dec_part, inc_part

    conv_model_trained.zero_grad()

    return loss_list


def deflate_model_multi(trained_model, x0, clip=1., rand_data=True, epochs=1, deflate_iter=5, early_stop=False, pqr_iters=10, step_size=0.35, quiet=True, device='cuda'):
    # trained_model = copy.deepcopy(orig_model)
    loss_list = []
    VT = x0
    for i in range(deflate_iter):
        # print('--------> ', i)
        # x0 = torch.randn(1, num_inp_channels, n).to(device)
        # trained_model.eval()
        (VT, qr) = power_qr(lambda x: trained_model(x) - trained_model(torch.zeros_like(x)), VT.clone().detach(), n_iters=pqr_iters, x0_out=True, quiet=True, device=device)
        trained_model.zero_grad()
        # print('largest SV: ', qr[-1])
        D_powerqr = qr[-1]
        # trained_model.train()

        # if D_powerqr[0] < clip:
        if D_powerqr[0] <= clip and clip - D_powerqr[0] <= 0.05:
            break

        trained_conv_model_copy = copy.deepcopy(trained_model)
        loss_list_cur = deflate_model(trained_model, trained_conv_model_copy, VT, D_powerqr[0], clip=clip, rand_data=False, epochs=epochs, early_stop=early_stop, step_size=step_size, quiet=quiet)
        loss_list.append(loss_list_cur[-1])

        if rand_data:
            VT = torch.rand_like(VT).to(device)

    return loss_list 

    
