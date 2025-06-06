import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision
from models.resnet import ResNet18
from models.resnet_orig import ResNet18_orig
from models.simple_conv import simpleConv
from models.simple_conv_orig import simpleConv_orig

from pgdl2_modified import PGDL2

import random
import numpy as np

import sys
import os
import pandas as pd

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)


from utils_ensemble import AverageMeter, accuracy, test
from trainer import Naive_Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--dataset', default='cifar', type=str, choices=DATASETS)
parser.add_argument('--dataset', default='cifar')
# parser.add_argument('--arch', default='ResNet18', type=str, choices=ARCHITECTURES)
parser.add_argument('--arch', default='wideResnet', type=str)
parser.add_argument('--model_path', default=None, type=str)
parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epoch', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch', default=128, type=int, metavar='N', help='batchsize (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=40, help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--num-models', default=3, type=int)
parser.add_argument('--resume', action='store_true', help='if true, tries to resume training from existing checkpoint')
parser.add_argument('--adv-training', action='store_true')
parser.add_argument('--epsilon', default=512, type=float)
parser.add_argument('--num-steps', default=4, type=int)
parser.add_argument('--adv-eps', default=0.2, type=float)

parser.add_argument('--model', default='ResNet18', help='clipping method (use orig for no clipping)')
parser.add_argument('--method', default='orig', help='clipping method (use orig for no clipping)')
parser.add_argument('--mode', default='wBN', help='what to do with BN layers (leave empty for keeping it as it is)')
parser.add_argument('--seed', default=1, type=int, help='seed value')
parser.add_argument('--convsn', default=1., type=float, help='clip value for conv and dense layers')

parser.add_argument('--unlearn_indices', default=None, type=str)

parser.add_argument('--widen_factor', default=1, type=int, help='widen factor for WideResNet')

parser.add_argument('--coeff', default=2.0, type=float)
parser.add_argument('--lamda', default=2.0, type=float)
parser.add_argument('--scale', default=5.0, type=float)
parser.add_argument('--plus-adv', action='store_false')
# parser.add_argument('--init-eps', default=0.1, type=float)
parser.add_argument('--init-eps', default=0.01, type=float)

args = parser.parse_args()

if args.adv_training:
    mode = f"adv_{args.epsilon}_{args.num_steps}"
else:
    if args.method == 'orig':
        mode = f"vanilla_orig_{args.mode}"
    else:
        mode = f"vanilla_clip{args.convsn}_{args.mode}"


args.epsilon /= 256.0

print('learning rate: ', args.lr)
print('dataset: ', args.dataset)


def main():

    elu_flag     = False #### for elu activation ----------------------------------
    clip_flag    = False
    orig_flag    = False

    seed_val = args.seed
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)

    mode = args.mode
    bn_flag = True
    opt_iter = 5
    clip_steps = 50
    if mode == 'wBN':
        mode = ''
        bn_flag = True
    elif mode == 'noBN':
        bn_flag = False
        opt_iter = 1
        clip_steps = 100

    if args.method == 'orig':
        orig_flag    = True
    else:
        clip_flag    = True


    if args.dataset == 'cifar':
        print('cifar!')
        in_chan = 3
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        transform_adv = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10( root='./data', train=False, download=True, transform=transform_test)


        indices_seed = args.unlearn_indices.split('/')[-1][:-4]
        indices_count = args.unlearn_indices.split('/')[-2]

        args.outdir = f"/{args.dataset}_unnorm/conf/{indices_count}/unl_idx_{indices_seed}/"
        args.outdir = "~/amun_exps/logs/correct/" + args.outdir

        print(args.outdir)
        print('learning rate: ', args.lr)
        print('dataset: ', args.dataset)

        unlearn_idx = pd.read_csv(args.unlearn_indices)['unlearn_idx'].values
        if len(unlearn_idx) != int(indices_count):
            print('unlearn_idx count is not correct!')
            exit(0)

        model_seed = args.model_path.split('/')[-2].split('_')[-1]
        # outdir = args.outdir + '/' + args.model + "_" + args.method + "_" + mode + "_" + model_seed + "/"
        outdir = args.outdir + args.model_path.split('/')[-2]
        print('outdir: ', outdir)

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        writer = SummaryWriter(outdir)

        print('==> Building model..')
        print('------------> outdir: ', outdir)
        print('-----------------------------------------------------------------')
        print('initial len of trainset: ', len(trainset))  

        removed_classes = [trainset[i][1] for i in unlearn_idx]

        ### remove the unlearned images from the trainset
        trainset_filtered = torch.utils.data.Subset(trainset, list(set(range(len(trainset))) - set(unlearn_idx)))
        print('len of filtered trainset: ', len(trainset_filtered))  

        forgetset = torch.utils.data.Subset(trainset, unlearn_idx)
        print('len of forget set: ', len(forgetset))  

        # advset_filtered = torch.utils.data.Subset(advset, unlearn_idx)
        # print('len of advset set: ', len(advset_filtered))  

        trainset = trainset_filtered

        print('final len of trainset: ', len(trainset))  
        print('-----------------------------------------------------------------')

        trainloader = torch.utils.data.DataLoader(trainset, shuffle=False, batch_size=1, num_workers=1)
        testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=1, num_workers=1)
        forgetloader = torch.utils.data.DataLoader(forgetset, shuffle=False, batch_size=1, num_workers=1)

    # writer = SummaryWriter(args.outdir)
    writer = None

    # submodel = get_architecture(args.arch, args.dataset)
    print('elu flag: ', elu_flag)
    print('arch: ', args.arch)
    if clip_flag:
        if args.arch == 'ResNet18':
            submodel = ResNet18(concat_sv=False, in_chan=in_chan, device=device, clip=args.convsn, clip_flag=True, bn=bn_flag, clip_steps=clip_steps,  clip_outer=False, clip_opt_iter=opt_iter, summary=True, writer=writer, save_info=False, elu_flag=elu_flag, identifier=0)
    elif orig_flag:
        if args.arch == 'ResNet18':
            submodel = ResNet18_orig(in_chan=in_chan, bn=bn_flag, device=device, elu_flag=elu_flag)
        elif args.arch == 'simpleConv':
            submodel = simpleConv_orig(in_chan=in_chan, bn=bn_flag, device=device)

    submodel = nn.DataParallel(submodel).cuda()
    model = submodel

    criterion = nn.CrossEntropyLoss().cuda()
    # for epoch in range(0,args.epochs,10):

    print('epoch: ', args.epoch)
    # model_path = os.path.join(args.outdir, 'checkpoint.pth.tar.', str(i), '_', str(epoch))
    model_path = args.model_path + '/checkpoint.pth.tar_' + str(args.epoch)
    if not os.path.exists(model_path):
        model_path = args.model_path + '/_ckpt.' + str(args.epoch)
    checkpoint = torch.load(model_path)
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    except:
        model.load_state_dict(checkpoint['net'], strict=False)

    model.eval()
    print('model loaded')

    test_acc, test_loss = test(testloader, model, criterion, args.epoch, device, writer)
    # train_acc, train_loss = test(train_loader, model, criterion, epoch, device, writer)
    print('test acc: ', test_acc)
    # print('train acc: ', train_acc)

    idx_list = []
    prob_list = []
    label_list = []
    for idx, (inputs, targets) in enumerate(trainloader):
        # if idx > 10:
        #     break
        # if idx % 50 == 0:
        # print(idx)

        # measure data loading time
        inputs, targets = inputs.to(device), targets.to(device)

        # print('max: ', inputs.max(), 'min: ', inputs.min())
        output = model(inputs)
        probs = nn.Softmax(dim=1)(output).reshape(-1)
        target_prob = probs[targets.reshape(-1)].item()

        idx_list.append(idx)
        prob_list.append(target_prob)
        label_list.append(targets.reshape(-1).item())

    df = pd.DataFrame({'idx': idx_list, 'label': label_list, 'prob': prob_list})
    df.to_csv(outdir + '_remain_probs.csv', index=False)


    idx_list = []
    prob_list = []
    label_list = []
    for idx, (inputs, targets) in enumerate(testloader):
        # if idx > 10:
        #     break
        # if idx % 50 == 0:
        # print(idx)

        # measure data loading time
        inputs, targets = inputs.to(device), targets.to(device)

        # print('max: ', inputs.max(), 'min: ', inputs.min())
        output = model(inputs)
        probs = nn.Softmax(dim=1)(output).reshape(-1)
        target_prob = probs[targets.reshape(-1)].item()

        idx_list.append(idx)
        prob_list.append(target_prob)
        label_list.append(targets.reshape(-1).item())

    df = pd.DataFrame({'idx': idx_list, 'label': label_list, 'prob': prob_list})
    df.to_csv(outdir + '_test_probs.csv', index=False)


    idx_list = []
    prob_list = []
    label_list = []
    for idx, (inputs, targets) in enumerate(forgetloader):
        # if idx > 10:
        #     break
        # if idx % 50 == 0:
        # print(idx)

        # measure data loading time
        inputs, targets = inputs.to(device), targets.to(device)

        # print('max: ', inputs.max(), 'min: ', inputs.min())
        output = model(inputs)
        probs = nn.Softmax(dim=1)(output).reshape(-1)
        target_prob = probs[targets.reshape(-1)].item()

        idx_list.append(idx)
        prob_list.append(target_prob)
        label_list.append(targets.reshape(-1).item())

    df = pd.DataFrame({'idx': idx_list, 'label': label_list, 'prob': prob_list})
    df.to_csv(outdir + '_forget_probs.csv', index=False)

if __name__ == "__main__":
    main()


