'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.spectral_normalization_nondiff_final_deflate import SpectralNorm
# from models.spectral_normalization import SpectralNorm #as SpectralNorm_miyato
# from models.spectral_normalization_deflate_complex_both import SpectralNorm
from models.spectral_normalization_deflate_complex_both_bn import SpectralNorm


class CNNBN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=False, device='cpu', clip_flag=True, clip=1., clip_steps=50, clip_opt_iter=1, init_delay=0, summary=False, identifier=0, writer=None, bn=True, bn_clip=False, bn_count=100, bn_hard=False, save_info=False):
        super(CNNBN, self).__init__()
        # self.sub_conv1_ = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode='circular')#, padding_mode='circular') ############## mnist
        self.sub_conv1 = SpectralNorm(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias), device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=str(identifier) + '_1', save_info=save_info)
        # self.sub_conv1 = self.sub_conv1_
        self.bn1 = SpectralNorm(nn.BatchNorm2d(out_planes, momentum=0.1, track_running_stats=True), device=device, clip_flag=bn_clip, clip=clip, clip_steps=bn_count, bn_hard=bn_hard, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=str(identifier) + '_bn', save_info=save_info)
        # self.bn1 = nn.BatchNorm2d(out_planes, momentum=0.1, track_running_stats=True)
        self.bn_flag = bn

    def forward(self, x):
        x = self.sub_conv1(x)
        if self.bn_flag:
            x = self.bn1(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, device='cpu', clip_flag=True, clip_outer=False, clip=1., clip_concat=1., clip_steps=50, clip_opt_iter=1, init_delay=0, summary=False, identifier=0, writer=None, bn=True, bn_clip=False, bn_count=100, bn_hard=False, concat_sv=False, save_info=False, outer_steps=200, outer_iters=1, elu_flag=False):
        super(BasicBlock, self).__init__()
        self.bn = bn
        self.elu_flag = elu_flag
        if concat_sv:
            self.conv1 = SpectralNorm(CNNBN(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+1, bn=bn, bn_clip=bn_clip, save_info=save_info and True, bn_count=bn_count, bn_hard=bn_hard), device=device, clip_flag=clip_outer, clip=clip_concat, clip_steps=outer_steps, clip_opt_iter=outer_iters, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+1, save_info=save_info and True)
        else:
            self.conv1 = CNNBN(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+1, bn=bn, bn_clip=bn_clip, save_info=save_info and True, bn_count=bn_count, bn_hard=bn_hard)
        # self.bn1 = nn.BatchNorm2d(planes)

        if concat_sv:
            self.conv2 = SpectralNorm(CNNBN(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+2, bn=bn, bn_clip=bn_clip, save_info=save_info and True, bn_count=bn_count, bn_hard=bn_hard), device=device, clip_flag=clip_outer, clip=clip_concat, clip_steps=outer_steps, clip_opt_iter=outer_iters, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+2, save_info=save_info and True)
        else:
            self.conv2 = CNNBN(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+2, bn=bn, bn_clip=bn_clip, save_info=save_info and True, bn_count=bn_count, bn_hard=bn_hard)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.shortcut_flag = stride != 1 or in_planes != self.expansion*planes
        if self.shortcut_flag:
            if concat_sv:
                self.shortcut = nn.Sequential(
                    SpectralNorm(CNNBN(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+3, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard), device=device, clip_flag=clip_outer, clip=clip_concat, clip_steps=outer_steps, clip_opt_iter=outer_iters, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+3),
                )
            else:
                self.shortcut = nn.Sequential(
                    CNNBN(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+3, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard),
                )

    def forward(self, x):
        out = self.conv1(x)
        if self.elu_flag:
            out = F.elu(out)
        else:
            out = F.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)

        if self.shortcut_flag:
            out = out / 2

        if self.elu_flag:
            out = F.elu(out)
        else:
            out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, device='cpu', clip_flag=True, clip_outer=False, clip=1., clip_steps=50, clip_opt_iter=1, init_delay=0, summary=False, identifier=0, writer=None, bn=True, save_info=False, outer_steps=200, outer_iters=1, clip_concat=1.):
        super(Bottleneck, self).__init__()
        # self.bn = bn
        self.conv1 = SpectralNorm(CNNBN(in_planes, planes, kernel_size=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+1, bn=bn, save_info=save_info and True), device=device, clip_flag=clip_outer, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+1, save_info=save_info and True)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SpectralNorm(CNNBN(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+2, bn=bn, save_info=save_info and True), device=device, clip_flag=clip_outer, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+2, save_info=save_info and True)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SpectralNorm(CNNBN(planes, self.expansion *
                               planes, kernel_size=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+3, bn=bn), device=device, clip_flag=clip_outer, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+3)
        # self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SpectralNorm(CNNBN(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+4, bn=bn), device=device, clip_flag=clip_outer, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+4),
                # nn.BatchNorm2d(self.expansion*planes)
            )


    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_chan=3, num_classes=10, device='cpu', clip_flag=True, clip_outer=False, clip=1., clip_concat=1., clip_steps=50, clip_opt_iter=1, init_delay=0, summary=False, identifier=0, writer=None, bn=True, bn_clip=False, bn_count=100, bn_hard=False, concat_sv=False, save_info=False, outer_steps=200, outer_iters=1, elu_flag=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.elu_flag = elu_flag

        if concat_sv:
            self.conv1 = SpectralNorm(CNNBN(in_chan, 64, kernel_size=3, stride=1, padding=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+100, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard), device=device, clip_flag=clip_outer, clip=clip_concat, clip_steps=outer_steps, clip_opt_iter=outer_iters, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+100)
        else:
            self.conv1 = CNNBN(in_chan, 64, kernel_size=3, stride=1, padding=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+100, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard)
        # self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, device=device, clip_flag=clip_flag, clip_outer=clip_outer, clip=clip, clip_concat=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+200, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard, concat_sv=concat_sv, outer_steps=outer_steps, outer_iters=outer_iters, elu_flag=elu_flag)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, device=device, clip_flag=clip_flag, clip_outer=clip_outer, clip=clip, clip_concat=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+300, bn=bn, bn_clip=bn_clip, save_info=save_info, bn_count=bn_count, bn_hard=bn_hard, concat_sv=concat_sv, outer_steps=outer_steps, outer_iters=outer_iters, elu_flag=elu_flag)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, device=device, clip_flag=clip_flag, clip_outer=clip_outer, clip=clip, clip_concat=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+400, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard, concat_sv=concat_sv, elu_flag=elu_flag)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, device=device, clip_flag=clip_flag, clip_outer=clip_outer, clip=clip, clip_concat=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+500, bn=bn, bn_clip=bn_clip, save_info=save_info, bn_count=bn_count, bn_hard=bn_hard, concat_sv=concat_sv, outer_steps=outer_steps, outer_iters=outer_iters, elu_flag=elu_flag)
        self.linear = SpectralNorm(nn.Linear(512*block.expansion, num_classes), device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+600)

    def _make_layer(self, block, planes, num_blocks, stride, device='cpu', clip_flag=True, clip_outer=False, clip=1., clip_concat=1., clip_steps=50, clip_opt_iter=1, init_delay=0, summary=False, identifier=0, writer=None, bn=True, bn_clip=False, save_info=False, bn_count=100, bn_hard=False, concat_sv=False, outer_steps=200, outer_iters=1, elu_flag=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            if save_info and i <= 1:
                save_info = True
            else:
                save_info = False

            layers.append(block(self.in_planes, planes, stride, device=device, clip_flag=clip_flag, clip_outer=clip_outer, clip=clip,clip_concat=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+10*(i+2), bn=bn, bn_clip=bn_clip, save_info=save_info, bn_count=bn_count, bn_hard=bn_hard, concat_sv=concat_sv, outer_steps=outer_steps, outer_iters=outer_iters, elu_flag=elu_flag))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.elu_flag:
            out = F.elu(out)
        else:
            out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(device='cpu', in_chan=3, clip_flag=True, clip_outer=False, clip=1., clip_concat=1., clip_steps=50, clip_opt_iter=1, init_delay=0, summary=False, identifier=0, writer=None, bn=True, bn_clip=False, bn_hard=False, bn_count=100, save_info=False, concat_sv=False, outer_steps=200, outer_iters=1, elu_flag=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_chan=in_chan, device=device, clip_flag=clip_flag, clip_outer=clip_outer, clip=clip, clip_concat=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, concat_sv=concat_sv, save_info=save_info, outer_steps=outer_steps, outer_iters=outer_iters, elu_flag=elu_flag)


def ResNet34(device='cpu', clip_flag=True, clip=1., clip_steps=50, clip_opt_iter=1, init_delay=0, summary=False, identifier=0, writer=None, bn=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier, bn=bn)


def ResNet50(device='cpu', clip_flag=True, clip=1., clip_steps=50, clip_opt_iter=1, init_delay=0, summary=False, identifier=0, writer=None, bn=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier, bn=bn)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
