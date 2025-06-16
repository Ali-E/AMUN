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
from models.vgg import VGG
from models.simple_conv import simpleConv
from models.simple_conv_orig import simpleConv_orig
from datasetslocal import get_dataset, unnormalize
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import copy

from vit_pytorch import ViT

import random
import numpy as np
import time
import torch.nn.functional as F
from torch.autograd import Variable

from datasets import load_dataset
import sys
import os
import pandas as pd

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)



from utils_ensemble import AverageMeter, accuracy, test
from trainer import Naive_Trainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)


class basicDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_in = self.data[idx]['image']
        image = copy.deepcopy(np.asarray(image_in))
        # print(image.shape)
        if len(image.shape) == 2:
            image = copy.deepcopy(np.stack((image, image, image), axis=2))
        # image = image.transpose(2, 0, 1)
        label = self.data[idx]['label']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='cifar')
parser.add_argument('--arch', default='ResNet18', type=str)
parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=201, type=int, metavar='N', help='number of total epochs to run')
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
parser.add_argument('--adv-eps', default=0.04, type=float)

parser.add_argument('--method', default='orig', help='clipping method (use orig for no clipping)')
parser.add_argument('--mode', default='wBN', help='what to do with BN layers (leave empty for keeping it as it is)')
parser.add_argument('--seed', default=1, type=int, help='seed value')
parser.add_argument('--convsn', default=1., type=float, help='clip value for conv and dense layers')

parser.add_argument('--widen_factor', default=1, type=int, help='widen factor for WideResNet')

parser.add_argument('--coeff', default=2.0, type=float)
parser.add_argument('--lamda', default=2.0, type=float)
parser.add_argument('--scale', default=5.0, type=float)
parser.add_argument('--plus-adv', action='store_false')
parser.add_argument('--init-eps', default=0.01, type=float)

args = parser.parse_args()

if args.adv_training:
    mode = f"adv_{args.epsilon}_{args.num_steps}"
else:
    if args.method == 'orig':
        mode = f"vanilla_orig_{args.mode}"
    else:
        mode = f"vanilla_clip{args.convsn}_{args.mode}"

# args.outdir = f"/{args.dataset}_unnorm/{mode}_{args.seed}/"
args.outdir = f"/{args.dataset}_{args.arch}_unnorm/{mode}_{args.seed}/"
if args.arch == 'vit':
    args.outdir = f"/{args.dataset}_vit_unnorm/{mode}_{args.seed}/"

args.epsilon /= 256.0

if (args.resume):
    args.outdir = "resume" + args.outdir
else:
    args.outdir = "scratch" + args.outdir

args.outdir = "logs/" + args.outdir
# args.outdir = "/scratch/bcwm/aebrahimpour/amun/logs/correct/" + args.outdir

print(args.outdir)
print('learning rate: ', args.lr)
print('dataset: ', args.dataset)

tinynet_flag = False


def main():

    elu_flag     = False #### for elu activation ----------------------------------
    clip_flag    = False
    orig_flag    = False
    tinynet_flag = False

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
        args.epochs = 121

    if args.method == 'orig':
        orig_flag    = True
    else:
        clip_flag    = True

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # copy_code(args.outdir)

    if args.dataset == 'cifar':
        print('cifar!')
        in_chan = 3
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=1)

        testset = torchvision.datasets.CIFAR10( root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=1)

    elif args.dataset == 'tinynet':
        print('Tine ImageNet!')
        args.batch = 256
        in_chan = 3
        tinynet_flag = True
        num_classes = 200
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset_all = load_dataset('Maysee/tiny-imagenet', split='train')
        trainset = basicDataset(trainset_all, transform=transform_train, target_transform=None)
        print('trainset: ', len(trainset))
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=1)

        testset_all = load_dataset('Maysee/tiny-imagenet', split='valid')
        testset = basicDataset(testset_all, transform=transform_test, target_transform=None)
        print('testset: ', len(testset))
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=1)

    else:

        print('args.dataset: ', args.dataset)
        # args.lr_step_size = 30
        train_dataset = get_dataset(args.dataset, 'train')
        test_dataset = get_dataset(args.dataset, 'test')
        pin_memory = (args.dataset == "imagenet")
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch, num_workers=args.workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch, num_workers=args.workers, pin_memory=pin_memory)
    

    if args.dataset == 'cifar':
        in_chan = 3
    elif args.dataset == 'mnist':
        in_chan = 1
        args.epochs = 121
        args.adv_eps = 0.1

    model_path = os.path.join(args.outdir, 'checkpoint.pth.tar')
    writer = SummaryWriter(args.outdir)

    model = None
    # submodel = get_architecture(args.arch, args.dataset)
    print('elu flag: ', elu_flag)
    print('arch: ', args.arch)
    if clip_flag:
        if args.arch == 'ResNet18':
            submodel = ResNet18(concat_sv=False, in_chan=in_chan, device=device, clip=args.convsn, clip_flag=True, bn=bn_flag, clip_steps=clip_steps, clip_outer=False, clip_opt_iter=opt_iter, summary=True, writer=writer, save_info=False, elu_flag=elu_flag, identifier=1000)
    elif orig_flag:
        if args.arch == 'ResNet18':
            submodel = ResNet18_orig(in_chan=in_chan, bn=bn_flag, device=device, elu_flag=elu_flag, num_classes=num_classes, tinynet=tinynet_flag)
        elif args.arch == 'VGG':
            submodel = VGG('VGG19', in_chan=in_chan, num_classes=num_classes, tinynet=tinynet_flag)
        elif args.arch == 'vit':
            submodel = ViT(
                image_size = 64,
                patch_size = 4,
                num_classes = num_classes,
                dim = 128,
                depth = 6,
                heads = 8,
                mlp_dim = 128
            )
        
    submodel = nn.DataParallel(submodel)
    model = submodel
    print("Model loaded")

    criterion = nn.CrossEntropyLoss().cuda()

    # param = list(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(param, lr=args.lr, weight_decay=args.weight_decay, eps=1e-7)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    if args.arch == 'vit':# or args.arch == 'VGG':
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
        # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
        # scheduler = CosineAnnealingLR(optimizer, T_max=80000, eta_min=0)


    if (args.resume):
        base_classifier = "logs/Empirical/scratch/" + args.dataset + "/vanilla/checkpoint.pth.tar"
        print(base_classifier)
        for i in range(3):
            checkpoint = torch.load(base_classifier + ".%d" % (i))
            print("Load " + base_classifier + ".%d" % (i))
            model[i].load_state_dict(checkpoint['state_dict'])
            model[i].train()
        print("Loaded...")

    trans_list = []
    loss_acc_list = []
    best_acc = 0.0
    for epoch in range(args.epochs):
        start = time.time()
        train_loss = Naive_Trainer(args, train_loader, model, criterion, optimizer, epoch, device, writer)
        tot_time = time.time() - start
        print('time: ', tot_time)

        # if epoch % 5 == 0:
        if True:
            test_acc, test_loss = test(test_loader, model, criterion, epoch, device, writer)
            loss_acc_list.append((epoch, train_loss, test_loss, test_acc))

            if test_acc >= best_acc:
                model_path_i = model_path + '_best'
                torch.save({
                        'epoch': epoch,
                        'arch': args.arch,
                        'scheduler': scheduler.state_dict(),
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, model_path_i)

                best_acc = test_acc
                # better_acc = True

            writer.add_scalar('test/best_acc', best_acc, epoch)


        if epoch % 10 == 0:
            # trans_list_new = evaltrans(args, test_loader, model, criterion, epoch, device, writer)
            # trans_list_new = evaltrans_correct(args, test_loader, model, criterion, epoch, device, writer, perturb_robust=False)
            # trans_list += trans_list_new

            model_path_i = model_path + "_%d" % (epoch)
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_path_i)

        # if not args.arch == 'vit':
        scheduler.step()

    loss_acc_res_path = os.path.join(args.outdir, 'loss_acc_e' + str(args.adv_eps) + '_s' + str(args.seed) + '.csv')
    df = pd.DataFrame(loss_acc_list, columns=['epoch', 'train_loss', 'test_loss', 'test_acc'])
    df.to_csv(loss_acc_res_path)

if __name__ == "__main__":
    main()


