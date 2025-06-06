import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm_miyato(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1, writer=None, k=1, clip_flag=True, deflate_iter=1, init_delay=500, clip=1., init_pqr_iter=100, pqr_iter=1, clip_steps=50, clip_opt_iter=1, clip_opt_stepsize=0.35, clip_rsv_size=1, summary=False, device='cuda', identifier=None, eval_mode=False):
        super(SpectralNorm_miyato, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self.writer = writer
        self.counter = 0
        self.eval_mode=eval_mode
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        if self.eval_mode:
            setattr(self.module, self.name, torch.nn.Parameter(w / sigma.expand_as(w)))
        else:
            setattr(self.module, self.name, w / sigma.expand_as(w))

        # if self.counter % 50 == 0:
        #     print(u.shape)
        #     print(v.shape)
        #     self.writer.add_scalar('train/miyato_', torch.sqrt(torch.sum(self.module(u.reshape(1,-1))**2)), self.counter)

        # w_orig = getattr(self.module, self.name)
        # w_orig.data = copy.deepcopy(w.detach())

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        if not self.eval_mode:
            del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        if self.training:
            self.counter += 1
        input_shape = args[0].shape
        # print(input_shape)
        self._update_u_v()
        return self.module.forward(*args)


