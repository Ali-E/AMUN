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

    def __init__(self, in_planes, planes, stride=1, bn=True, bn_clip=False, bn_count=100, bn_hard=False, device='cuda'):
        super(BasicBlock, self).__init__()
        self.input_size = [6]
        self.conv1 = nn.Conv2d( in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.kernels = [(self.conv1.weight, self.input_size)]

        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = SpectralNorm(nn.BatchNorm2d(planes), device=device, clip_flag=bn_clip, clip=1., clip_steps=bn_count, bn_hard=bn_hard)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.kernels.append((self.conv2.weight, self.input_size))

        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = SpectralNorm(nn.BatchNorm2d(planes), device=device, clip_flag=bn_clip, clip=1., clip_steps=bn_count, bn_hard=bn_hard)

        self.bn_flag = bn

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if self.bn_flag:
                # self.shortcut = nn.Sequential(
                conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
                bn_l = SpectralNorm(nn.BatchNorm2d(self.expansion*planes), device=device, clip_flag=bn_clip, clip=1., bn_hard=bn_hard, clip_steps=bn_count)
                    # nn.BatchNorm2d(self.expansion*planes)
                # )
                self.shortcut = nn.Sequential(conv, bn_l)
                self.kernels.append((conv.weight, self.input_size))
            else:
                # self.shortcut = nn.Sequential(
                conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
                # )
                self.shortcut = nn.Sequential(conv)
                self.kernels.append((conv.weight, self.input_size))


    def get_all_kernels(self):
        return self.kernels


    def forward(self, x):
        self.input_size[0] = x.shape[-1]
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
    def __init__(self, in_channels, out_channels, kernel_size=1, bn=True, bn_clip=False, bn_count=100, bn_hard=False, device='cuda'):
        super(Root, self).__init__()
        self.input_size = [6]
        self.kernels = []

        self.conv = nn.Conv2d( in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False)
        self.kernels = [(self.conv.weight, self.input_size)]

        # self.bn = nn.BatchNorm2d(out_channels)
        self.bn = SpectralNorm(nn.BatchNorm2d(out_channels), device=device, clip_flag=bn_clip, clip=1., clip_steps=bn_count, bn_hard=bn_hard)

        self.bn_flag = bn

    def get_all_kernels(self):
        return self.kernels

    def forward(self, xs):
        # self.input_size[0] = xs.shape[-1]
        x = torch.cat(xs, 1)
        self.input_size[0] = x.shape[-1]
        if self.bn_flag:
            out = F.relu(self.bn(self.conv(x)))
        else:
            out = F.relu(self.conv(x))
        return out


class Tree(nn.Module):
    def __init__(self, block, in_channels, out_channels, level=1, stride=1, bn=True, bn_clip=False, bn_count=100, bn_hard=False, device='cuda'):
        super(Tree, self).__init__()
        self.kernels = []
        self.level = level
        if level == 1:
            self.root = Root(2*out_channels, out_channels, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard, device='cuda')
            self.kernels.extend(self.root.get_all_kernels())
            self.left_node = block(in_channels, out_channels, stride=stride, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard, device='cuda')
            self.kernels.extend(self.left_node.get_all_kernels())
            self.right_node = block(out_channels, out_channels, stride=1, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard, device='cuda')
            self.kernels.extend(self.right_node.get_all_kernels())
        else:
            self.root = Root((level+2)*out_channels, out_channels, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard, device='cuda')
            self.kernels.extend(self.root.get_all_kernels())
            for i in reversed(range(1, level)):
                subtree = Tree(block, in_channels, out_channels, level=i, stride=stride, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard, device='cuda')
                self.kernels.extend(subtree.get_all_kernels())
                self.__setattr__('level_%d' % i, subtree)
            self.prev_root = block(in_channels, out_channels, stride=stride, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard, device='cuda')
            self.kernels.extend(self.prev_root.get_all_kernels())
            self.left_node = block(out_channels, out_channels, stride=1, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard, device='cuda')
            self.kernels.extend(self.left_node.get_all_kernels())
            self.right_node = block(out_channels, out_channels, stride=1, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard, device='cuda')
            self.kernels.extend(self.right_node.get_all_kernels())

    def get_all_kernels(self):
        return self.kernels

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


class DLA_orig(nn.Module):
    def __init__(self, in_chan=3, block=BasicBlock, num_classes=10, bn=True, bn_clip=False, bn_count=100, bn_hard=False, clip_linear=False, device='cuda'):
        super(DLA_orig, self).__init__()

        self.input_size = [32]
        self.bn_flag = bn
        self.kernels = []

        if self.bn_flag:
            # self.base = nn.Sequential(
            conv = nn.Conv2d(in_chan, 16, kernel_size=3, stride=1, padding=1, bias=False)
            bn_l = SpectralNorm(nn.BatchNorm2d(16), device=device, clip_flag=bn_clip, clip=1., bn_hard=bn_hard, clip_steps=bn_count)
            # nn.BatchNorm2d(16),
            rel_act = nn.ReLU(True)
            # )
            self.base = nn.Sequential(conv, bn_l, rel_act)
            self.kernels.append((conv.weight, self.input_size))
        else:
            # self.base = nn.Sequential(
            conv = nn.Conv2d(in_chan, 16, kernel_size=3, stride=1, padding=1, bias=False)
            rel_act = nn.ReLU(True)
            # )
            self.base = nn.Sequential(conv, rel_act)
            self.kernels.append((conv.weight, self.input_size))

        if self.bn_flag:
            # self.layer1 = nn.Sequential(
            conv = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
            bn_l = SpectralNorm(nn.BatchNorm2d(16), device=device, clip_flag=bn_clip, clip=1., bn_hard=bn_hard, clip_steps=bn_count)
            # nn.BatchNorm2d(16),
            rel_act = nn.ReLU(True)
            # )
            self.layer1 = nn.Sequential(conv, bn_l, rel_act)
            self.kernels.append((conv.weight, self.input_size))
        else:
            # self.layer1 = nn.Sequential(
            conv = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
            rel_act = nn.ReLU(True)
            # )
            self.layer1 = nn.Sequential(conv,  rel_act)
            self.kernels.append((conv.weight, self.input_size))

        if self.bn_flag:
            # self.layer2 = nn.Sequential(
            conv = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
            bn_l = SpectralNorm(nn.BatchNorm2d(32), device=device, clip_flag=bn_clip, clip=1., bn_hard=bn_hard, clip_steps=bn_count)
            # nn.BatchNorm2d(32),
            rel_act = nn.ReLU(True)
            # )
            self.layer2 = nn.Sequential(conv, bn_l, rel_act)
            self.kernels.append((conv.weight, self.input_size))
        else:
            # self.layer2 = nn.Sequential(
            conv = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
            rel_act = nn.ReLU(True)
            # )
            self.layer2 = nn.Sequential(conv,  rel_act)
            self.kernels.append((conv.weight, self.input_size))


        self.layer3 = Tree(block,  32,  64, level=1, stride=1, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard, device='cuda')
        self.kernels.extend(self.layer3.get_all_kernels())
        self.layer4 = Tree(block,  64, 128, level=2, stride=2, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard, device='cuda')
        self.kernels.extend(self.layer4.get_all_kernels())
        self.layer5 = Tree(block, 128, 256, level=2, stride=2, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard, device='cuda')
        self.kernels.extend(self.layer5.get_all_kernels())
        self.layer6 = Tree(block, 256, 512, level=1, stride=2, bn=bn, bn_clip=bn_clip, bn_count=bn_count, bn_hard=bn_hard, device='cuda')
        self.kernels.extend(self.layer6.get_all_kernels())

        if clip_linear:
            self.linear = miyato(nn.Linear(512, num_classes))
        else:
            self.linear = nn.Linear(512, num_classes)

    def get_all_kernels(self):
        return self.kernels

    def forward(self, x):
        self.input_size[0] = x.shape[-1]
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
    net = DLA_orig()
    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    test()
