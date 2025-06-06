import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spectral_normalization_deflate_complex_both_bn import SpectralNorm


class CNNBN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='zeros', bias=False, device='cpu', clip_flag=True, clip=1., clip_steps=50, clip_opt_iter=1, init_delay=0, summary=False, identifier=0, writer=None, bn=True, save_info=False):
        super(CNNBN, self).__init__()
        self.sub_conv1 = SpectralNorm(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode= padding_mode, bias=bias), device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier='_conv', save_info=save_info)

        self.bn1 = SpectralNorm(nn.BatchNorm2d(out_planes, momentum=0.1, track_running_stats=True), device=device, clip_flag=False, clip=1., clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier='_bn', save_info=save_info)

        self.bn_flag = bn

    def forward(self, x):
        x = self.sub_conv1(x)
        if self.bn_flag:
            x = self.bn1(x)
        return x



class SimpleConv(nn.Module):
    def __init__(self, concat_sv=False, in_chan=3, k=1, num_classes=10, leaky_relu=False, device='cpu', clip_flag=True, clip_outer=False, clip=1., clip_concat=1., clip_steps=50, clip_opt_iter=5, init_delay=0, summary=False, identifier='', writer=None, bias=True, bn=True, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='zeros', lin_layer=None):
        super(SimpleConv, self).__init__()
        if concat_sv:
            print('cocant sv is being recorded!')
            self.conv1_ = CNNBN(in_chan, 64, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode=padding_mode, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier='', bn=bn)
            self.conv1 = SpectralNorm(self.conv1_, device=device, clip_flag=clip_outer, clip=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, summary=summary, writer=writer, identifier=identifier+'_concat', init_delay=init_delay)
        else:
            self.conv1_ = CNNBN(in_chan, 64, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode=padding_mode, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier, bn=bn)
            self.conv1 = self.conv1_

        W = 32
        if in_chan == 1:
            W = 28

        outdim = (W - kernel_size+2*padding)//stride + 1
        size = outdim*outdim*64

        if lin_layer is not None:
            size = lin_layer

        self.fc1_ = nn.Linear(size, 10)
        self.fc1 = SpectralNorm(self.fc1_, device=device, clip_flag=False, clip=clip, clip_steps=clip_steps, clip_opt_iter=1, summary=summary, writer=writer, identifier='_dense', clip_opt_stepsize=0.5, init_delay=init_delay)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def simpleConv(**kwargs):
    return SimpleConv(**kwargs)



