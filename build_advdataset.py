import torch
import torch.nn as nn
import argparse
import torchvision.transforms as transforms
import torchvision
from models.resnet import ResNet18
from models.resnet_orig import ResNet18_orig
from models.simple_conv import simpleConv
from models.simple_conv_orig import simpleConv_orig
from pgdl2_modified import PGDL2#, FGSM
# from pgdl2_modified_better import PGDL2#, FGSM
import random
import numpy as np
import sys
import os
import pandas as pd

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from utils_ensemble import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='cifar')
parser.add_argument('--arch', default='ResNet18', type=str)
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
parser.add_argument('--method', default='orig', help='clipping method (use orig for no clipping)')
parser.add_argument('--mode', default='wBN', help='what to do with BN layers (leave empty for keeping it as it is)')
parser.add_argument('--seed', default=1, type=int, help='seed value')
parser.add_argument('--convsn', default=1., type=float, help='clip value for conv and dense layers')
parser.add_argument('--bottom_clip', default=0.5, type=float, help='lower bound for smallest singular value')
parser.add_argument('--widen_factor', default=1, type=int, help='widen factor for WideResNet')
parser.add_argument('--unnormalize', default=True, type=bool)
parser.add_argument('--norm_cond', default='unnorm', help='unnorm or norm for transform')
parser.add_argument('--attack', default='pgdl2', help='unnorm or norm for transform')
parser.add_argument('--coeff', default=2.0, type=float)
parser.add_argument('--lamda', default=2.0, type=float)
parser.add_argument('--scale', default=5.0, type=float)
parser.add_argument('--plus-adv', action='store_false')
parser.add_argument('--init-eps', default=0.01, type=float)

args = parser.parse_args()

if args.norm_cond == 'norm':
    args.unnormalize = False
print('!!!!!!!!! unnormalized: ', args.unnormalize)

if args.adv_training:
    mode = f"adv_{args.epsilon}_{args.num_steps}"
else:
    if args.method == 'orig':
        mode = f"vanilla_orig_{args.mode}"
    else:
        mode = f"vanilla_clip{args.convsn}_{args.mode}"

args.outdir = args.model_path + f"/adv_data/seed_{args.seed}/"
args.epsilon /= 256.0


print(args.outdir)
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

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    if args.dataset == 'cifar':
        print('cifar!')
        in_chan = 3

        if args.unnormalize:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        trainset = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, transform=transform_train)
        # train_loader = torch.utils.data.DataLoader( trainset, batch_size=1, shuffle=False, num_workers=1)
        train_loader = torch.utils.data.DataLoader( trainset, batch_size=args.batch, shuffle=False, num_workers=1)

        testset = torchvision.datasets.CIFAR10( root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader( testset, batch_size=1, shuffle=False, num_workers=1)

    writer = None

    print('elu flag: ', elu_flag)
    print('arch: ', args.arch)
    if clip_flag:
        if args.arch == 'ResNet18':
            submodel = ResNet18(concat_sv=False, in_chan=in_chan, device=device, clip=args.convsn, clip_flag=True, bn=bn_flag, clip_steps=clip_steps,  clip_outer=False, clip_opt_iter=opt_iter, summary=True, writer=writer, save_info=False, elu_flag=elu_flag, identifier=0)
        elif args.arch == 'simpleConv':
            submodel = simpleConv(concat_sv=False, in_chan=in_chan, device=device, clip=args.convsn, clip_bottom=args.bottom_clip, clip_flag=True, bn=bn_flag, clip_steps=clip_steps//2,  clip_outer=False, clip_opt_iter=opt_iter, summary=True, writer=writer, save_info=False, identifier=0)
    elif orig_flag:
        if args.arch == 'ResNet18':
            submodel = ResNet18_orig(in_chan=in_chan, bn=bn_flag, device=device, elu_flag=elu_flag)
        elif args.arch == 'simpleConv':
            submodel = simpleConv_orig(in_chan=in_chan, bn=bn_flag, device=device)

    submodel = nn.DataParallel(submodel).cuda()
    model = submodel
    criterion = nn.CrossEntropyLoss().cuda()

    print('epoch: ', args.epoch)
    model_path = args.model_path + '/checkpoint.pth.tar_' + str(args.epoch)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    print('model loaded')

    test_acc, test_loss = test(test_loader, model, criterion, args.epoch, device, writer)
    print('test acc: ', test_acc)

    adv_img_list = None
    s_eps_list = []
    idx_list = []
    adv_pred_list = []
    orig_pred_list = []
    label_list = []
    delta_norm_list = []
    min_eps = 1000

    global_idx = 0
    for idx, (inputs, targets) in enumerate(train_loader):
        if idx > 50:
            break
        if idx % 50 == 0:
            print(idx)
        inputs, targets = inputs.to(device), targets.to(device)

        if args.attack == 'pgdl2':
            pgd_finder = PGDL2(model)

        adv_img, orig_pred, adv_pred, eps_vec, dnorm_vec = pgd_finder.forward(inputs, targets)

        # concatenate adversarial images
        adv_img_list = adv_img if adv_img_list is None \
                    else torch.cat((adv_img_list, adv_img), dim=0)

        # book-keeping
        batch_size = inputs.size(0)
        idx_list.extend(range(global_idx, global_idx + batch_size))
        label_list.extend(targets.cpu().tolist())
        orig_pred_list.extend(orig_pred.cpu().tolist())
        adv_pred_list.extend(adv_pred.cpu().tolist())
        s_eps_list.extend(eps_vec.cpu().tolist())
        delta_norm_list.extend(dnorm_vec.cpu().tolist())

        global_idx += batch_size

        if True: #idx > 0 and idx % 5000 == 0:
            if args.attack == 'pgdl2':
                adv_tensor_path = os.path.join(args.outdir, f'paral_cor_adv_tensor.pt')
                smallest_eps = os.path.join(args.outdir, f'paral_cor_smallest_eps.csv')
            else:
                adv_tensor_path = os.path.join(args.outdir, f'paral_cor_adv_tensor_{args.attack}.pt')
                smallest_eps = os.path.join(args.outdir, f'paral_cor_smallest_eps_{args.attack}.csv')

            torch.save(adv_img_list, adv_tensor_path)
            df = pd.DataFrame({'idx': idx_list, 'label': label_list, 'orig_pred': orig_pred_list, 'adv_pred': adv_pred_list, 'smallest_eps': s_eps_list, 'delta_norm': delta_norm_list})
            df.to_csv(smallest_eps, index=False)

    if args.attack == 'pgdl2':
        adv_tensor_path = os.path.join(args.outdir, f'paral_cor_adv_tensor.pt')
        smallest_eps = os.path.join(args.outdir, f'Paral_cor_smallest_eps.csv')
    else:
        adv_tensor_path = os.path.join(args.outdir, f'paral_cor_adv_tensor_{args.attack}.pt')
        smallest_eps = os.path.join(args.outdir, f'paral_cor_smallest_eps_{args.attack}.csv')

    torch.save(adv_img_list, adv_tensor_path)

    df = pd.DataFrame({'idx': idx_list, 'label': label_list, 'orig_pred': orig_pred_list, 'adv_pred': adv_pred_list, 'smallest_eps': s_eps_list, 'delta_norm': delta_norm_list})
    df.to_csv(smallest_eps, index=False)


if __name__ == "__main__":
    main()

