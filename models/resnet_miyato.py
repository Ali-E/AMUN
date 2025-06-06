'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spectral_normalization import SpectralNorm_miyato as miyato
from models.spectral_normalization_deflate_complex_both_bn import SpectralNorm



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, bn_clip=False, bn_count=100, bn_hard=False, device='cuda', eval_mode=False):
        super(BasicBlock, self).__init__()
        self.conv1 = miyato(nn.Conv2d( in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False), eval_mode=eval_mode)

        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = SpectralNorm(nn.BatchNorm2d(planes), device=device, clip_flag=bn_clip, clip=1., clip_steps=bn_count, bn_hard=bn_hard)

        self.conv2 = miyato(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False), eval_mode=eval_mode)

        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = SpectralNorm(nn.BatchNorm2d(planes), device=device, clip_flag=bn_clip, clip=1., clip_steps=bn_count, bn_hard=bn_hard)

        self.shortcut_flag = False

        self.bn = bn

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_flag = True
            if self.bn:
                self.shortcut = nn.Sequential(
                    miyato(nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False), eval_mode=eval_mode),
                    # nn.BatchNorm2d(self.expansion*planes)
                    SpectralNorm(nn.BatchNorm2d(self.expansion*planes), device=device, clip_flag=bn_clip, clip=1., clip_steps=bn_count, bn_hard=bn_hard)
                )
            else:
                self.shortcut = nn.Sequential(
                    miyato(nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False), eval_mode=eval_mode),
                )


    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        # print('2: ', out.shape)
        if self.bn:
            out = self.bn2(self.conv2(out))
        else:
            out = self.conv2(out)

        # print('3: ', out.shape)
        out += self.shortcut(x)

        # if self.shortcut_flag:
            # print('4: ', out.shape)

        if self.shortcut_flag:
            # to be one 1 lip
            out = out / 2

        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = miyato(nn.Conv2d(in_planes, planes, kernel_size=1, bias=False), eval_mode=eval_mode)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = miyato(nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False), eval_mode=eval_mode)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = miyato(nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False), eval_mode=eval_mode)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                miyato(nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False), eval_mode=eval_mode),
                nn.BatchNorm2d(self.expansion*planes)
            )

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


class ResNet_miyato(nn.Module):
    def __init__(self, in_chan, block, num_blocks, num_classes=10, bn=True, bn_clip=False, bn_count=100, bn_hard=False, device='cuda', eval_mode=False):
        super(ResNet_miyato, self).__init__()
        self.in_planes = 64

        self.conv1 = miyato(nn.Conv2d(in_chan, 64, kernel_size=3, stride=1, padding=1, bias=False), eval_mode=eval_mode)
        # self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = SpectralNorm(nn.BatchNorm2d(64), device=device, clip_flag=bn_clip, clip=1., bn_hard=bn_hard, clip_steps=bn_count)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, eval_mode=eval_mode)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, eval_mode=eval_mode)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, eval_mode=eval_mode)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, eval_mode=eval_mode)
        self.linear = miyato(nn.Linear(512*block.expansion, num_classes), eval_mode=eval_mode)

        self.bn = bn

    def _make_layer(self, block, planes, num_blocks, stride, bn=True, bn_clip=False, bn_count=100, bn_hard=False, device='cuda', eval_mode=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard, device=device, eval_mode=eval_mode))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print('x: ', x.shape)
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
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


def ResNet18_miyato(in_chan=3, bn=True, bn_clip=False, bn_count=100, bn_hard=False, device='cuda', eval_mode=False):
    return ResNet_miyato(in_chan, BasicBlock, [2, 2, 2, 2], bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard, device=device, eval_mode=eval_mode)


def ResNet34_miyato(in_chan=3, bn=True, bn_clip=False, device='cuda'):
    return ResNet_miyato(in_chan, BasicBlock, [3, 4, 6, 3], bn=bn, bn_clip=bn_clip, device=device)


def ResNet50_miyato():
    return ResNet_miyato(Bottleneck, [3, 4, 6, 3])


def ResNet101_miyato():
    return ResNet_miyato(Bottleneck, [3, 4, 23, 3])


def ResNet152_miyato():
    return ResNet_miyato(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18_miyato()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
