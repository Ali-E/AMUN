'''DLA in PyTorch.

Reference:
    Deep Layer Aggregation. https://arxiv.org/abs/1707.06484
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spectral_normalization import SpectralNorm_miyato as miyato
from models.spectral_normalization_deflate_complex_both_bn import SpectralNorm


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,  bn=True, bn_clip=False, bn_count=100, bn_hard=False, device='cuda', eval_mode=False):
        super(BasicBlock, self).__init__()
        self.conv1 = miyato(nn.Conv2d( in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False), eval_mode=eval_mode)

        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = SpectralNorm(nn.BatchNorm2d(planes), device=device, clip_flag=bn_clip, clip=1., clip_steps=bn_count, bn_hard=bn_hard)

        self.conv2 = miyato(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False), eval_mode=eval_mode)

        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = SpectralNorm(nn.BatchNorm2d(planes), device=device, clip_flag=bn_clip, clip=1., clip_steps=bn_count, bn_hard=bn_hard)

        self.bn_flag = bn

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if self.bn_flag:
                self.shortcut = nn.Sequential(
                    miyato(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False), eval_mode=eval_mode),
                    SpectralNorm(nn.BatchNorm2d(self.expansion*planes), device=device, clip_flag=bn_clip, clip=1., bn_hard=bn_hard, clip_steps=bn_count),
                    # nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    miyato(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False), eval_mode=eval_mode),
                )


    def forward(self, x):
        if self.bn_flag:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))

        if self.bn_flag:
            out = self.bn2(self.conv2(out))
        else:
            out = self.conv2(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bn=True, bn_clip=False, bn_count=100, bn_hard=False, device='cuda', eval_mode=False):
        super(Root, self).__init__()
        self.conv = miyato(nn.Conv2d( in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False), eval_mode=eval_mode)

        # self.bn = nn.BatchNorm2d(out_channels)
        self.bn = SpectralNorm(nn.BatchNorm2d(out_channels), device=device, clip_flag=bn_clip, clip=1., clip_steps=bn_count, bn_hard=bn_hard)

        self.bn_flag = bn

    def forward(self, xs):
        x = torch.cat(xs, 1)
        if self.bn_flag:
            out = F.relu(self.bn(self.conv(x)))
        else:
            out = F.relu(self.conv(x))
        return out


class Tree(nn.Module):
    def __init__(self, block, in_channels, out_channels, level=1, stride=1, bn=True, bn_clip=False, bn_count=100, bn_hard=False, device='cuda', eval_mode=False):
        super(Tree, self).__init__()
        self.level = level
        if level == 1:
            self.root = Root(2*out_channels, out_channels, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, eval_mode=eval_mode)
            self.left_node = block(in_channels, out_channels, stride=stride, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, eval_mode=eval_mode)
            self.right_node = block(out_channels, out_channels, stride=1, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, eval_mode=eval_mode)
        else:
            self.root = Root((level+2)*out_channels, out_channels, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, eval_mode=eval_mode)
            for i in reversed(range(1, level)):
                subtree = Tree(block, in_channels, out_channels, level=i, stride=stride, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, eval_mode=eval_mode)
                self.__setattr__('level_%d' % i, subtree)
            self.prev_root = block(in_channels, out_channels, stride=stride, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, eval_mode=eval_mode)
            self.left_node = block(out_channels, out_channels, stride=1, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, eval_mode=eval_mode)
            self.right_node = block(out_channels, out_channels, stride=1, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, eval_mode=eval_mode)

    def forward(self, x):
        xs = [self.prev_root(x)] if self.level > 1 else []
        for i in reversed(range(1, self.level)):
            level_i = self.__getattr__('level_%d' % i)
            x = level_i(x)
            xs.append(x)
        x = self.left_node(x)
        xs.append(x)
        x = self.right_node(x)
        xs.append(x)
        out = self.root(xs)
        return out


class DLA_miyato(nn.Module):
    def __init__(self, in_chan=3, block=BasicBlock, num_classes=10, bn=True, bn_clip=False, bn_count=100, bn_hard=False, device='cuda', eval_mode=False):
        super(DLA_miyato, self).__init__()

        self.bn_flag = bn

        if self.bn_flag:
            # self.base = nn.Sequential(
            conv = miyato(nn.Conv2d(in_chan, 16, kernel_size=3, stride=1, padding=1, bias=False), eval_mode=eval_mode)
            bn_l = SpectralNorm(nn.BatchNorm2d(16), device=device, clip_flag=bn_clip, clip=1., bn_hard=bn_hard, clip_steps=bn_count)
            # nn.BatchNorm2d(16),
            rel_act = nn.ReLU(True)
            # )
            self.base = nn.Sequential(conv, bn_l, rel_act)
        else:
            # self.base = nn.Sequential(
            conv = miyato(nn.Conv2d(in_chan, 16, kernel_size=3, stride=1, padding=1, bias=False), eval_mode=eval_mode)
            rel_act = nn.ReLU(True)
            # )
            self.base = nn.Sequential(conv, rel_act)

        if self.bn_flag:
            # self.layer1 = nn.Sequential(
            conv = miyato(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False), eval_mode=eval_mode)
            bn_l = SpectralNorm(nn.BatchNorm2d(16), device=device, clip_flag=bn_clip, clip=1., bn_hard=bn_hard, clip_steps=bn_count)
            # nn.BatchNorm2d(16),
            rel_act = nn.ReLU(True)
            # )
            self.layer1 = nn.Sequential(conv, bn_l, rel_act)
        else:
            # self.layer1 = nn.Sequential(
            conv = miyato(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False), eval_mode=eval_mode)
            rel_act = nn.ReLU(True)
            # )
            self.layer1 = nn.Sequential(conv,  rel_act)

        if self.bn_flag:
            # self.layer2 = nn.Sequential(
            conv = miyato(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False), eval_mode=eval_mode)
            bn_l = SpectralNorm(nn.BatchNorm2d(32), device=device, clip_flag=bn_clip, clip=1., bn_hard=bn_hard, clip_steps=bn_count)
            # nn.BatchNorm2d(32),
            rel_act = nn.ReLU(True)
            # )
            self.layer2 = nn.Sequential(conv, bn_l, rel_act)
        else:
            # self.layer2 = nn.Sequential(
            conv = miyato(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False), eval_mode=eval_mode)
            rel_act = nn.ReLU(True)
            # )
            self.layer2 = nn.Sequential(conv,  rel_act)

        self.layer3 = Tree(block,  32,  64, level=1, stride=1, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, eval_mode=eval_mode)
        self.layer4 = Tree(block,  64, 128, level=2, stride=2, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, eval_mode=eval_mode)
        self.layer5 = Tree(block, 128, 256, level=2, stride=2, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, eval_mode=eval_mode)
        self.layer6 = Tree(block, 256, 512, level=1, stride=2, bn=bn, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=bn_count, device=device, eval_mode=eval_mode)
        self.linear = miyato(nn.Linear(512, num_classes), eval_mode=eval_mode)

    def forward(self, x):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = DLA_miyato()
    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    test()
