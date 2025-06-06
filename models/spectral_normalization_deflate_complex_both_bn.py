import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from models.clip_toolbox import *
import copy
import numpy as np
import pandas as pd

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', k=1, clip_flag=True, deflate_iter=1, init_delay=500, clip=1., init_pqr_iter=100, pqr_iter=1, clip_steps=50, clip_opt_iter=5, clip_opt_stepsize=0.35, clip_rsv_size=1, summary=False, device='cuda', bn_hard=True, writer=None, identifier=None, save_info=False, scale_only=False, exact_flag=True):
        super(SpectralNorm, self).__init__()
        self.name = name
        self.pqr_iter = pqr_iter
        self.init_pqr_iter = init_pqr_iter
        self.D_powerqr = None
        self.clip_steps = clip_steps
        self.clip = clip
        self.clip_opt_iter = clip_opt_iter
        self.clip_opt_stepsize = clip_opt_stepsize 
        self.device=device
        self.module = module.to(self.device)
        self.counter = 0
        self.k = k
        self.clip_data_size = clip_rsv_size
        self.summary=summary
        self.clip_flag = clip_flag
        self.init_delay = init_delay
        self.deflate_iter = deflate_iter
        self.save_info = save_info
        self.scale_only = scale_only
        self.bn_hard = bn_hard
        self.warn_flag = True
        self.exact_flag = exact_flag

        if self.save_info:
            self.lsv_list = []
        if self.summary:
            self.lsv = None
            self.identifier = identifier
            if self.identifier == None:
                self.identifier = self.module.__class__.__name__
                print(self.identifier)
        self.writer = writer

        if self.clip_flag:
            print('!!!!!!! Clipping is active !!!!!!!! clip val: ', self.clip)


    def _update_VT(self, *args, n_iters=None, rand_init=False):
        if self.save_info and self.counter > 0 and self.counter % 200 == 0:
            df = pd.DataFrame(self.lsv_list)

        if isinstance(self.module, torch.nn.BatchNorm2d):
            if self.summary:
                running_var = torch.ones_like(self.module.weight)
                if vars(self.module)['_buffers']['running_var'] is not None:
                    input_ = args[0]
                    cur_var = torch.var(input_, unbiased=False, dim=[0,2,3])
                    running_var_prev = vars(self.module)['_buffers']['running_var'] 
                    running_var = running_var_prev 
                    running_var = running_var + vars(self.module)['eps']  # ----------------> the one we use
                else:
                    input_ = args[0]
                    cur_var = torch.var(input_, unbiased=False, dim=[0,2,3])
                    running_var = cur_var + vars(self.module)['eps']

                self.lsv = torch.max(torch.abs(self.module.weight/torch.sqrt(running_var)))
                if self.counter % 50 == 0 and self.writer is not None:
                    self.writer.add_scalar('train/lsv_bn_' + str(self.identifier), self.lsv, self.counter)

            if self.save_info and self.counter % 100 == 0:
                self.lsv_list.append(self.lsv.item()) 
            return 

        if n_iters is None:
            n_iters = self.pqr_iter
        if not self._made_params():
            self._make_params(*args)
        VT = getattr(self.module, self.name + "_VT")
        if rand_init:
            VT = torch.rand_like(VT)

        self.module.eval()
        VT, out = power_qr(lambda x: self.module(x) - self.module(torch.zeros_like(x)), VT, n_iters=n_iters, record=False, quiet=True, x0_out=True, device=self.device)
        self.module.zero_grad()
        self.D_powerqr = out[-1].detach()
        self.module.train()

        if self.summary:
            self.lsv = self.D_powerqr[0]
            if self.counter % 50 == 0 and self.writer is not None:
                self.writer.add_scalar('train/lsv_' + str(self.identifier), self.lsv, self.counter)

        if self.save_info and self.counter % 100 == 0 and self.counter > 0:
            self.lsv_list.append(self.lsv.item()) 

        VT.requires_grad_(False)
        self.D_powerqr.requires_grad_(False)
        
        with torch.no_grad():
            setattr(self.module, self.name + "_VT", VT.detach())
        del VT


    def _clip_module(self, *args):
        if isinstance(self.module, torch.nn.Linear):
            VT = getattr(self.module, self.name + "_VT")
            w = getattr(self.module, self.name)
            data_size = self.clip_data_size
            trained_module = copy.deepcopy(self.module)

            if self.D_powerqr[0] > self.clip:
                deflate_model_multi(trained_module, VT[0:data_size].detach(), clip=self.clip, step_size=self.clip_opt_stepsize, epochs=self.clip_opt_iter, device=self.device, deflate_iter=self.deflate_iter)

            w.data = copy.deepcopy(trained_module.weight.detach())

        elif isinstance(self.module, (torch.nn.Conv1d, torch.nn.Conv2d)):
            if self.scale_only:
                w = getattr(self.module, self.name)
                if self.D_powerqr[0] > self.clip:
                    w.data = copy.deepcopy(self.module.weight.detach()/self.D_powerqr[0])

            else:
                VT = getattr(self.module, self.name + "_VT")
                w = getattr(self.module, self.name)
                with torch.no_grad():
                    data_size = self.clip_data_size
                    trained_module = copy.deepcopy(self.module)

                if self.exact_flag:
                    clip_condition = self.D_powerqr[0] > self.clip or self.clip - self.D_powerqr[0] >= 0.05
                else:
                    clip_condition = self.D_powerqr[0] > self.clip
                if clip_condition:
                    deflate_model_multi(trained_module, VT[0:data_size].detach(), clip=self.clip, step_size=self.clip_opt_stepsize, epochs=self.clip_opt_iter, device=self.device, deflate_iter=self.deflate_iter)

                w.data = copy.deepcopy(trained_module.weight.detach())
                del trained_module

        elif isinstance(self.module, torch.nn.BatchNorm2d):
            w = getattr(self.module, self.name)
            with torch.no_grad():
                running_var = torch.ones_like(w)
                if vars(self.module)['_buffers']['running_var'] is not None:
                    input_ = args[0]
                    cur_var = torch.var(input_, unbiased=False, dim=[0,2,3])
                    running_var_prev = vars(self.module)['_buffers']['running_var'] 
                    momentum = vars(self.module)['momentum'] 
                    new_running_var = momentum*cur_var + (1.-momentum)*running_var_prev 
                    running_var = running_var_prev + vars(self.module)['eps'] 

                    if self.counter % 50 == 0 and self.writer is not None:
                        self.writer.add_scalar('train/prev_run_var_' + str(self.identifier), running_var_prev.sum().item()/running_var_prev.shape[0], self.counter)
                        self.writer.add_scalar('train/cur_run_var_' + str(self.identifier), cur_var.sum().item()/cur_var.shape[0], self.counter)
                        self.writer.add_scalar('train/new_run_var_' + str(self.identifier), new_running_var.sum().item()/new_running_var.shape[0], self.counter)

                else:
                    input_ = args[0]
                    cur_var = torch.var(input_, unbiased=False, dim=[0,2,3])
                    if self.counter % 50 == 0 and self.writer is not None:
                        self.writer.add_scalar('train/cur_run_var_' + str(self.identifier), cur_var.sum().item()/cur_var.shape[0], self.counter)

            
            w_clamped = torch.clamp(copy.deepcopy(w), min=-torch.sqrt(running_var)*self.clip, max=torch.sqrt(running_var)*self.clip)

            if self.bn_hard:
                w.data = w_clamped
            else:
                w.data = w_clamped

        else:
            VT = getattr(self.module, self.name + "_VT")
            layers = []
            for (m_name, m) in self.module.named_modules():
                if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                    for (m_name_, m_) in m.named_modules():
                        if isinstance(m_, (torch.nn.Linear, torch.nn.Conv2d)):
                            layers.append(copy.deepcopy(m))

                elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    layers.append(copy.deepcopy(m))

            trained_module = nn.Sequential(*layers)

            self.clip_opt_stepsize = 0.01
            with torch.no_grad():
                data_size = self.clip_data_size

            if self.D_powerqr[0] > self.clip:
                deflate_model_multi(trained_module, VT[0:data_size].detach(), clip=self.clip, step_size=self.clip_opt_stepsize, epochs=self.clip_opt_iter, device=self.device, deflate_iter=self.deflate_iter)

            trained_module.zero_grad()

            stateDict = {}
            for key, value in trained_module.state_dict().items():
                if key.split('_')[-1] == 'VT':
                    continue
                ln = key.split('.')[0] 
                new_key = key[1:]
                if ln == '0':
                    new_key = 'sub_conv1.module' + new_key
                elif ln == '1':
                    new_key = 'bn1.module' + new_key

                stateDict[new_key] = value

            with torch.no_grad():
                self.module.load_state_dict(stateDict, strict=False)
            del trained_module


    def _made_params(self):
        try:
            VT = getattr(self.module, self.name + "_VT")
            return True
        except AttributeError:
            return False


    def _make_params(self, *args):
        input_shape = args[0].shape
        self.n = input_shape[-1]
        self.VT_shape = input_shape[1:]

        VT_shape = tuple([self.k] + list(self.VT_shape))
        x_batch = torch.randn(VT_shape, device=self.device)
        self.module.eval()
        VT, out = power_qr(lambda x: self.module(x) - self.module(torch.zeros_like(x)), x_batch, n_iters=self.init_pqr_iter, record=False, quiet=True, x0_out=True, device=self.device)
        self.module.zero_grad()
        self.D_powerqr = out[-1].detach()
        self.module.train()
        if self.summary:
            self.lsv = self.D_powerqr[0]
            if self.save_info:
                self.lsv_list.append(self.lsv.item()) 
            if self.counter % 50 == 0 and self.writer is not None:
                self.writer.add_scalar('train/lsv_' + str(self.identifier), self.lsv, self.counter)

        self.module.register_buffer(self.name + "_VT", VT)
        del VT
        del x_batch


    def forward(self, *args):
        if self.training:
            self._update_VT(*args)
            self.counter += 1
            if self.counter % self.clip_steps == 0 and self.counter > self.init_delay and self.clip_flag:
                if self.warn_flag:
                    self.warn_flag = False

                self._clip_module(*args)
                self._update_VT(*args, n_iters=10, rand_init=True)

        return self.module.forward(*args)


