'''DLA in PyTorch.

Reference:
    Deep Layer Aggregation. https://arxiv.org/abs/1707.06484
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spectral_normalization_deflate_complex_both_bn import SpectralNorm


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"



class CNNBN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=False, device='cpu', clip_flag=True, clip=1., clip_steps=50, clip_opt_iter=1, init_delay=0, summary=False, identifier=0, writer=None, bn=True, clipBN=False, bn_count=100, bn_hard=False):
        super(CNNBN, self).__init__()
        self.sub_conv1 = SpectralNorm(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias), device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=str(identifier) + '_1')
        self.bn1 = SpectralNorm(nn.BatchNorm2d(out_planes, momentum=0.1, track_running_stats=True), device=device, clip_flag=clipBN, bn_hard=bn_hard, clip=1., clip_steps=bn_count, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=str(identifier) + '_bn')
        self.bn_flag = bn

    def forward(self, x):
        x = self.sub_conv1(x)
        if self.bn_flag:
            x = self.bn1(x)
        return x



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, device='cpu', clip_flag=True, clip_outer=False, clip=1., clip_concat=1., clip_steps=50, clip_opt_iter=1, summary=False, init_delay=0, writer=None, identifier=0, bn=True, clipBN=False, bn_count=100, bn_hard=False, concat_sv=False, outer_steps=200, outer_iters=1):
        super(BasicBlock, self).__init__()

        if concat_sv:
            self.cnnbn1 = SpectralNorm(CNNBN(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+10, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard), device=device, clip_flag=clip_outer, clip=clip_concat, clip_steps=outer_steps, clip_opt_iter=outer_iters, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+10)
        else:
            self.cnnbn1 = CNNBN(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+10, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard)

        if concat_sv:
            self.cnnbn2 = SpectralNorm(CNNBN(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+20, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard), device=device, clip_flag=clip_outer, clip=clip_concat, clip_steps=outer_steps, clip_opt_iter=outer_iters, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+20)
        else:
            self.cnnbn2 = CNNBN(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+20, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if concat_sv:
                self.shortcut = nn.Sequential(
                    SpectralNorm(CNNBN(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+30, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard), device=device, clip_flag=clip_outer, clip=clip_concat, clip_steps=outer_steps, clip_opt_iter=outer_iters, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+30),
                )
            else:
                self.shortcut = nn.Sequential(
                    CNNBN(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+30, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard),
                )

    def forward(self, x):
        out = F.relu(self.cnnbn1(x))
        out = self.cnnbn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, device='cpu', clip_flag=True, clip_outer=False, clip=1., clip_concat=1., clip_steps=50, clip_opt_iter=1, summary=False, init_delay=0, writer=None, identifier=0, bn=True, clipBN=False, bn_count=100, bn_hard=False, concat_sv=False, outer_steps=200, outer_iters=1):
        super(Root, self).__init__()
        if concat_sv:
            self.cnnbn = SpectralNorm(CNNBN( in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+1, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard), device=device, clip_flag=clip_outer, clip=clip_concat, clip_steps=outer_steps, clip_opt_iter=outer_iters, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+10)
        else:
            self.cnnbn = CNNBN(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+1, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard)

    def forward(self, xs):
        x = torch.cat(xs, 1)
        # out = F.relu(self.bn(self.conv(x)))
        out = F.relu(self.cnnbn(x))
        return out


class Tree(nn.Module):
    def __init__(self, block, in_channels, out_channels, level=1, stride=1, device='cpu', clip_flag=True, clip_outer=False, clip=1., clip_concat=1., clip_steps=50, clip_opt_iter=1, summary=False, init_delay=0, writer=None, identifier=0, bn=True, clipBN=False, bn_count=100, bn_hard=False, concat_sv=False, outer_steps=200, outer_iters=1):
        super(Tree, self).__init__()
        self.level = level
        if level == 1:
            self.root = Root(2*out_channels, out_channels, device=device, clip_flag=clip_flag, clip_outer=clip_outer, clip=clip, clip_concat=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+100, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard, concat_sv=concat_sv, outer_steps=outer_steps, outer_iters=outer_iters)
            self.left_node = block(in_channels, out_channels, stride=stride, device=device, clip_flag=clip_flag, clip_outer=clip_outer, clip=clip, clip_concat=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+200, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard, concat_sv=concat_sv, outer_steps=outer_steps, outer_iters=outer_iters)
            self.right_node = block(out_channels, out_channels, stride=1, device=device, clip_flag=clip_flag, clip_outer=clip_outer, clip=clip, clip_concat=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+300, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard, concat_sv=concat_sv, outer_steps=outer_steps, outer_iters=outer_iters)
        else:
            self.root = Root((level+2)*out_channels, out_channels, device=device, clip_flag=clip_flag, clip_outer=clip_outer, clip=clip, clip_concat=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+400, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard, concat_sv=concat_sv)
            for i in reversed(range(1, level)):
                subtree = Tree(block, in_channels, out_channels,
                               level=i, stride=stride, device=device, clip_flag=clip_flag, clip_outer=clip_outer, clip=clip, clip_concat=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+500, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard, concat_sv=concat_sv, outer_steps=outer_steps, outer_iters=outer_iters)
                self.__setattr__('level_%d' % i, subtree)
            self.prev_root = block(in_channels, out_channels, stride=stride, device=device, clip_flag=clip_flag, clip_outer=clip_outer, clip=clip, clip_concat=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+600, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard, concat_sv=concat_sv, outer_steps=outer_steps, outer_iters=outer_iters)
            self.left_node = block(out_channels, out_channels, stride=1, device=device, clip_flag=clip_flag, clip_outer=clip_outer, clip=clip, clip_concat=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+700, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard, concat_sv=concat_sv, outer_steps=outer_steps, outer_iters=outer_iters)
            self.right_node = block(out_channels, out_channels, stride=1, device=device, clip_flag=clip_flag, clip_outer=clip_outer, clip=clip, clip_concat=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+800, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard, concat_sv=concat_sv, outer_steps=outer_steps, outer_iters=outer_iters)

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


class DLA(nn.Module):
    def __init__(self, in_chan=3, block=BasicBlock, num_classes=10, device='cuda', clip_flag=True, clip_outer=False, clip=1., clip_concat=1., clip_steps=50, outer_steps=200, outer_iters=1, clip_opt_iter=1, summary=False, init_delay=0, writer=None, identifier=0, bn=True, bn_clip=False, bn_hard=False, bn_count=100, concat_sv=False):
        super(DLA, self).__init__()
        clipBN = bn_clip
        if writer is None:
            print("!!!!!!! Writer is not set!")

        if concat_sv:
            self.base = nn.Sequential(
                SpectralNorm(CNNBN(in_chan, 16, kernel_size=3, stride=1, padding=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+1, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard), device=device, clip_flag=clip_outer, clip=clip_concat, clip_steps=outer_steps, clip_opt_iter=outer_iters, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+1),
                nn.ReLU(True)
            )
        else:
            self.base = nn.Sequential(
                CNNBN(in_chan, 16, kernel_size=3, stride=1, padding=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+1, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard),
                nn.ReLU(True)
            )

        if concat_sv:
            self.layer1 = nn.Sequential(
                SpectralNorm(CNNBN(16, 16, kernel_size=3, stride=1, padding=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+2, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard), device=device, clip_flag=clip_outer, clip=clip_concat, clip_steps=outer_steps, clip_opt_iter=outer_iters, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+2),
                nn.ReLU(True)
            )
        else:
            self.layer1 = nn.Sequential(
                CNNBN(16, 16, kernel_size=3, stride=1, padding=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+2, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard),
                nn.ReLU(True)
            )


        if concat_sv:
            self.layer2 = nn.Sequential(
                SpectralNorm(CNNBN(16, 32, kernel_size=3, stride=1, padding=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+3, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard), device=device, clip_flag=clip_outer, clip=clip_concat, clip_steps=outer_steps, clip_opt_iter=outer_iters, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+3),
                nn.ReLU(True)
            )
        else:
            self.layer2 = nn.Sequential(
                CNNBN(16, 32, kernel_size=3, stride=1, padding=1, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+3, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard),
                nn.ReLU(True)
            )


        self.layer3 = Tree(block,  32,  64, level=1, stride=1, device=device, clip_flag=clip_flag, clip_outer=clip_outer, clip=clip, clip_concat=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+1000, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard, concat_sv=concat_sv, outer_steps=outer_steps, outer_iters=outer_iters)
        self.layer4 = Tree(block,  64, 128, level=2, stride=2, device=device, clip_flag=clip_flag, clip_outer=clip_outer, clip=clip, clip_concat=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+2000, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard, concat_sv=concat_sv, outer_steps=outer_steps, outer_iters=outer_iters)
        self.layer5 = Tree(block, 128, 256, level=2, stride=2, device=device, clip_flag=clip_flag, clip_outer=clip_outer, clip=clip, clip_concat=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+3000, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard, concat_sv=concat_sv, outer_steps=outer_steps, outer_iters=outer_iters)
        self.layer6 = Tree(block, 256, 512, level=1, stride=2, device=device, clip_flag=clip_flag, clip_outer=clip_outer, clip=clip, clip_concat=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+4000, bn=bn, clipBN=clipBN, bn_count=bn_count, bn_hard=bn_hard, concat_sv=concat_sv, outer_steps=outer_steps, outer_iters=outer_iters)
        self.linear = SpectralNorm(nn.Linear(512, num_classes), device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, clip_opt_stepsize=0.5, summary=summary, init_delay=init_delay, writer=writer, identifier=identifier+5)
        # self.linear = SpectralNorm_miyato(nn.Linear(512, num_classes), writer=writer)
        # self.linear = nn.Linear(512, num_classes)

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
    net = DLA()
    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    test()
