'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.spectral_normalization import SpectralNorm_miyato as miyato
from models.spectral_normalization_deflate_complex_both_bn import SpectralNorm



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, bn_clip=False, bn_count=100, bn_hard=False, device='cuda', elu_flag=False):
        super(BasicBlock, self).__init__()
        self.input_size = [6]
        self.conv1 = nn.Conv2d( in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.kernels = [(self.conv1.weight, self.input_size)]
        self.elu_flag = elu_flag

        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = SpectralNorm(nn.BatchNorm2d(planes), device=device, clip_flag=bn_clip, clip=1., clip_steps=bn_count, bn_hard=bn_hard)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.kernels.append((self.conv2.weight, self.input_size))

        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = SpectralNorm(nn.BatchNorm2d(planes), device=device, clip_flag=bn_clip, clip=1., clip_steps=bn_count, bn_hard=bn_hard)


        self.shortcut_flag = False

        self.shortcut = nn.Sequential()

        self.bn_flag = bn


        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_flag = True
            if self.bn_flag:
                # self.shortcut = nn.Sequential(
                conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
                # nn.BatchNorm2d(self.expansion*planes)
                bn_l = SpectralNorm(nn.BatchNorm2d(self.expansion*planes), device=device, clip_flag=bn_clip, clip=1., clip_steps=bn_count, bn_hard=bn_hard)
                # )
                self.shortcut = nn.Sequential(conv, bn_l)
                self.kernels.append((conv.weight, self.input_size))
            else:
                conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
                self.shortcut = nn.Sequential(conv)
                self.kernels.append((conv.weight, self.input_size))


    def get_all_kernels(self):
        return self.kernels


    def forward(self, x):
        self.input_size[0] = x.shape[-1]
        if self.bn_flag:
            if self.elu_flag:
                out = F.elu(self.bn1(self.conv1(x)))
            else:
                out = F.relu(self.bn1(self.conv1(x)))
        else:
            if self.elu_flag:
                out = F.elu(self.conv1(x))
            else:
                out = F.relu(self.conv1(x))

        if self.bn_flag:
            out = self.bn2(self.conv2(out))
        else:
            out = self.conv2(out)

        out += self.shortcut(x)

        # if self.shortcut_flag:
            # print('4: ', out.shape)

        if self.shortcut_flag:
            # to be one 1 lip
            out = out / 2

        if self.elu_flag:
            out = F.elu(out)
        else:
            out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def get_all_kernels(self):
        return self.kernels

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        print(out.shape)
        out = F.relu(self.bn2(self.conv2(out)))
        print(out.shape)
        out = self.bn3(self.conv3(out))
        print(out.shape)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_orig(nn.Module):
    def __init__(self, in_chan, block, num_blocks, num_classes=10, bn=True, bn_clip=False, bn_count=100, bn_hard=False, clip_linear=False, device='cuda', elu_flag=False):
        super(ResNet_orig, self).__init__()
        self.in_planes = 64
        self.elu_flag = elu_flag

        self.conv1 = nn.Conv2d(in_chan, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.kernels = [(self.conv1.weight, [32])]
        if in_chan == 1:
            self.kernels = [(self.conv1.weight, [28])]



        # self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = SpectralNorm(nn.BatchNorm2d(64), device=device, clip_flag=bn_clip, clip=1., bn_hard=bn_hard, clip_steps=bn_count)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, elu_flag=elu_flag)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, elu_flag=elu_flag)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, elu_flag=elu_flag)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, elu_flag=elu_flag)

        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.bn_flag = bn

    def _make_layer(self, block, planes, num_blocks, stride, bn=True, bn_clip=False, bn_count=100, bn_hard=False, device='cuda', elu_flag=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            one_block = block(self.in_planes, planes, stride, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard, device=device, elu_flag=elu_flag)
            self.kernels.extend(one_block.get_all_kernels())
            layers.append(one_block)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def get_all_kernels(self):
        return self.kernels


    def forward(self, x):
        # print('x: ', x.shape)
        if self.bn_flag:
            if self.elu_flag:
                out = F.elu(self.bn1(self.conv1(x)))
            else:
                out = F.relu(self.bn1(self.conv1(x)))
        else:
            if self.elu_flag:
                out = F.elu(self.conv1(x))
            else:
                out = F.relu(self.conv1(x))

        # print('1: ', out.shape)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_orig(in_chan=3, bn=True, bn_clip=False, bn_count=100, bn_hard=False, clip_linear=False, device='cuda', elu_flag=False):
    return ResNet_orig(in_chan, BasicBlock, [2, 2, 2, 2], bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard, clip_linear=clip_linear, device=device, elu_flag=elu_flag)


def ResNet34_orig():
    return ResNet_orig(BasicBlock, [3, 4, 6, 3])


def ResNet50_orig():
    return ResNet_orig(Bottleneck, [3, 4, 6, 3])


def ResNet101_orig():
    return ResNet_orig(Bottleneck, [3, 4, 23, 3])


def ResNet152_orig():
    return ResNet_orig(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18_orig()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
