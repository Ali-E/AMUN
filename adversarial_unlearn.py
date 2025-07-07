import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from numpy.lib.format import open_memmap
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from datasetslocal import get_dataset
from generate_mask import save_gradient_ratio
from datasets import load_dataset
from pgdl2_modified import PGDL2
import os
import argparse
from models import *
from models.resnet_orig import ResNet18_orig
from models.resnet_orig_orig import ResNet18_orig as ResNet18_orig_orig
from models.vgg import VGG
import pandas as pd
import random
import time
import copy
from torch.utils.data import Dataset
from torchvision.io import read_image

from helper import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', default='cifar', help='dataset')
parser.add_argument('--model', default='ResNet18', help='Deep Learning model to train')
parser.add_argument('--method', default='catclip', help='clipping method (use orig for no clipping)')
parser.add_argument('--mode', default='wBN', help='what to do with BN layers (leave empty for keeping it as it is)')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--LRsteps', default=40, type=int, help='LR scheduler step')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--seed', default=1, type=int, help='seed value')
parser.add_argument('--steps', default=50, type=int, help='setp count for clipping BN')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes in the dataset')
parser.add_argument('--batch_size', default=128, type=int, help='number of classes in the dataset')

parser.add_argument('--unlearn_method', default='adv', type=str)
parser.add_argument('--unlearn_indices', default=None, type=str)

parser.add_argument('--unlearn_count', default=1000, type=int)
parser.add_argument('--start_idx', default=0, type=int)

parser.add_argument('--adv_images', default=None, type=str)
parser.add_argument('--adv_delta', default=None, type=str)
parser.add_argument('--source_model_path', default=None, type=str)
parser.add_argument('--mask_path', default=None, type=str)
parser.add_argument('--save_checkpoints', default=0, type=int)

parser.add_argument('--use_all_ref', default=True, type=bool)
parser.add_argument('--use_remain', default=True, type=bool)
parser.add_argument('--remain', default='use', type=str)
parser.add_argument('--use_remain_sample', default=False, type=bool)

parser.add_argument('--ablation_test', default=-1, type=int)
parser.add_argument('--ablation_RL', default=False, type=bool)
parser.add_argument('--ablation_RS', default=False, type=bool)
parser.add_argument('--ablation_forgetset', default=False, type=bool)
parser.add_argument('--ablation_forgetset_RL', default=False, type=bool)
parser.add_argument('--ablation_forgetset_AdvL', default=False, type=bool)

parser.add_argument('--ablation_advset', default=False, type=bool)

parser.add_argument('--req_mode', default='single', type=str)
parser.add_argument('--real_adaptive', default=False, type=bool)
parser.add_argument('--adaptive_lr', default=False, type=bool)
parser.add_argument('--adaptive_lr_factor', default=0.9, type=float)

parser.add_argument('--alpha_l1', default=0., type=float)
parser.add_argument('--salun_ratio', default='0.5', type=str, help='ratio of masking in salun')
parser.add_argument('--attack', default='pgdl2', type=str)
parser.add_argument('--amun_randadvset', default=False, type=bool)

parser.add_argument('--catsn', default=-1, type=float)
parser.add_argument('--convsn', default=1., type=float)
parser.add_argument('--outer_steps', default=100, type=int)
parser.add_argument('--convsteps', default=100, type=int)
parser.add_argument('--opt_iter', default=5, type=int)
parser.add_argument('--outer_iters', default=1, type=int)

args = parser.parse_args()


if args.ablation_test > 0:
    if args.ablation_test == 1:
        args.ablation_RL = True
        print('ablation RL')
    if args.ablation_test == 2:
        args.ablation_RS = True
        print('ablation RS')
    if args.ablation_test == 3:
        args.ablation_forgetset = True
        print('ablation forgetset')
    if args.ablation_test == 4: 
        args.ablation_forgetset_RL = True
        print('ablation forgetset RL')
    if args.ablation_test == 5:
        args.ablation_forgetset_AdvL = True
        print('ablation forgetset AdvL')

    if args.ablation_test == 6:
        args.ablation_advset = True
        print('ablation advset')

if args.unlearn_method != 'reference':
    unlearn_indices_check = pd.read_csv(args.unlearn_indices)['unlearn_idx'].values
    count_unlearn = len(unlearn_indices_check)
    print('count_unlearn: ', count_unlearn)

print('requested mode: ', args.req_mode)
print('!!!!!!!!! salun ratio: ', args.salun_ratio)
print('model: ', args.model)

dataset_name = args.dataset + '_' + args.model
if args.dataset == 'tinynet':
    args.num_classes = 200
    args.batch_size = 256
print('dataset', dataset_name)

if args.remain != 'use':
    args.use_remain = False

if args.remain == 'use' or args.unlearn_method == 'reference' or args.unlearn_method == 'retrain':
    args.use_remain = True

if args.ablation_advset:
    args.use_remain = False

if args.unlearn_method == 'amun_rand':
    args.amun_randadvset = True

print('use remain flag: ', args.use_remain)

if args.unlearn_method == 'amun' or args.unlearn_method == 'amun_others' or args.unlearn_method == 'amun_rand':
    print('using attack: ', args.attack)

indices_seed = args.unlearn_indices.split('/')[-1][:-4]
use_noise_adv = False

if args.unlearn_method in ['advonly_sa', 'amun_sa', 'salun']:
    # args.mask_path = f'/scratch/bcwm/aebrahimpour/amun/logs/correct/scratch/{dataset_name}/unlearn/genmask/{count_unlearn}/unl_idx_seed_1/vanilla_orig_wBN_1/salun_mask/with_{args.salun_ratio}.pt'
    model_name = args.source_model_path.split('/')[-1]
    args.mask_path = f'/scratch/bcwm/aebrahimpour/amun/logs/correct/scratch/{dataset_name}/unlearn/genmask/{count_unlearn}/unl_idx_{indices_seed}/{model_name}/salun_mask/with_{args.salun_ratio}.pt'
    print('mask path: ', args.mask_path)

if args.unlearn_method == 'advonly_sa':
    args.unlearn_method = 'advonly'
elif args.unlearn_method == 'amun_sa':
    args.unlearn_method = 'amun'
elif args.unlearn_method == 'salun':
    args.unlearn_method = 'RL'
elif args.unlearn_method == 'amun_l1':
    args.unlearn_method = 'amun'
    args.alpha_l1 = 0.0005

save_checkpoints = args.save_checkpoints
if save_checkpoints == 1:
    save_checkpoints = True
else:
    save_checkpoints = False

print('save_checkpoints: ', save_checkpoints)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('==========', device)

if device == 'cuda':
    print('chosen: ', device)
    cudnn.benchmark = True

base_path_df = pd.read_csv('path_file.csv')
print(base_path_df)
tuples = zip(base_path_df['info'], base_path_df['path'])
base_path_dict = dict(tuples)
base_path = base_path_dict['base_path']
print('base_path: ', base_path)

# Training
def train(epoch, optimizer, scheduler, criterion, unlearn_method='adv', writer=None, model_path="./checkpoints/", mask=None, advset=None):
    if unlearn_method == 'amun' or args.unlearn_method == 'amun_others' or args.unlearn_method == 'amun_rand':
        unlearn_method = 'adv'
    elif unlearn_method == 'advonly_others':
        unlearn_method = 'advonly'

    print('\nEpoch: %d' % epoch)
    print('l1 regularization: ', args.alpha_l1)
    print('unlearn method: ', unlearn_method)
    global count_setp
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = -1

    transform_adv = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.dataset == 'cifar':
        transform_advrand = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    elif args.dataset == 'tinynet':
        transform_advrand = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

    print('\ninside train function :')
    print('trainset :', len(trainset) )
    if args.unlearn_method != 'reference':
        print('unl idx :', len(unlearn_idx) )

    if unlearn_method == 'adv' or unlearn_method == 'advonly':
        if args.ablation_RS or args.ablation_forgetset_AdvL or args.ablation_forgetset_RL:
            ## image_orig contains the original images for the unlearn indices (forgetset) without random cropping and flipping
            advset = CustomImageDataset(labels_file=args.adv_delta, imgs_path=args.adv_images,
                                     unlearn_indices=unlearn_idx, transform=transform_adv, target_transform=None, ablation_RL=args.ablation_RL, ablation_RS=args.ablation_RS, image_orig=trainset_filtered_RS_images, ablation_forgetset_AdvL=args.ablation_forgetset_AdvL, ablation_forgetset_RL=args.ablation_forgetset_RL, num_classes=args.num_classes)

        elif args.ablation_forgetset: ## using forgetset only as the train set (no advset)
            advset = copy.deepcopy(forgetset)
        elif args.ablation_advset: ## using all the adversarial examples (for all the samples) as the train set 
            if args.dataset == 'cifar':
                unlearn_indices_ = np.array(list(range(50000)))
            elif args.dataset == 'tinynet':
                unlearn_indices_ = np.array(list(range(100000)))

            advset = CustomImageDataset(labels_file=args.adv_delta, imgs_path=args.adv_images,
                                         unlearn_indices=unlearn_indices_, transform=transform_adv, target_transform=None, ablation_RL=args.ablation_RL, ablation_RS=args.ablation_RS, num_classes=args.num_classes)
                                        # unlearn_indices=unlearn_indices_, transform=transform_advrand, target_transform=None, ablation_RL=args.ablation_RL, ablation_RS=args.ablation_RS, num_classes=args.num_classes)

        elif args.req_mode == 'adaptive' and advset is None and args.real_adaptive:
            ############# new adv:

            trainset_new = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, transform=transform_adv) ### transofrm=transform_train
            forgetset_new = torch.utils.data.Subset(trainset_new, unlearn_idx)
            forget_loader_1 = torch.utils.data.DataLoader(forgetset_new, batch_size=1, shuffle=False, num_workers=1)

            net.zero_grad()
            net.eval()

            adv_img_list = None
            adv_pred_list = []
            orig_pred_list = []
            label_list = []
            s_eps_list = []
            for idx, (inputs, targets) in enumerate(forget_loader_1):
                # net.zero_grad()
                if idx % 50 == 0:
                    print(idx)

                inputs, targets = inputs.to(device), targets.to(device)
                pgd_finder = PGDL2(net)

                adv_img, orig_pred, adv_pred, s_eps, delta_norm = pgd_finder.forward(inputs, targets)
                if adv_img_list is None:
                    adv_img_list = adv_img
                else:
                    adv_img_list = torch.cat((adv_img_list, adv_img), dim=0)

                s_eps_list.append(s_eps)
                label_list.extend(list(targets.detach().cpu().numpy()))
                orig_pred_list.extend(list(orig_pred.detach().cpu().numpy()))
                adv_pred_list.extend(list(adv_pred.detach().cpu().numpy()))

            advset = simpleDataset(adv_img_list, adv_pred_list, transform=transform_adv, target_transform=None)
            print('len of new advset: ', len(advset))
            print(s_eps_list[:20])

            net.train()

            #########################

        else:
            if not args.amun_randadvset:

                advset = CustomImageDataset(labels_file=args.adv_delta, imgs_path=args.adv_images,
                                         unlearn_indices=unlearn_idx, transform=transform_adv, target_transform=None, ablation_RL=args.ablation_RL, ablation_RS=args.ablation_RS, num_classes=args.num_classes)
            else:
                print('using random crop and flip for adv images')
                advset = CustomImageDataset(labels_file=args.adv_delta, imgs_path=args.adv_images,
                                        unlearn_indices=unlearn_idx, transform=transform_advrand, target_transform=None, ablation_RL=args.ablation_RL, ablation_RS=args.ablation_RS, num_classes=args.num_classes)

        if args.use_remain:
            trainset_combined = torch.utils.data.ConcatDataset([trainset, advset])
        else:
            if unlearn_method == 'advonly':
                print('only advset is being used!')
                trainset_combined = advset
            else:
                trainset_combined = torch.utils.data.ConcatDataset([forgetset, advset])

    elif unlearn_method == 'retrain' or unlearn_method == 'FT' or unlearn_method == 'l1':
        if not args.use_remain:
            sample_indices = np.random.choice(len(trainset), len(forgetset), replace=False)
            trainset_combined = torch.utils.data.Subset(trainset, sample_indices)
        else:
            trainset_combined = trainset
        
        if unlearn_method == 'l1' and args.alpha_l1 == 0.:
            args.alpha_l1 = 0.0005

    elif unlearn_method == 'RL':
        # RLset = RLDataset(forgetset, num_classes=args.num_classes, new_classes=new_classes)
        RLset = RLDataset(forgetset, num_classes=args.num_classes, new_classes=None)
        if args.use_remain:
            trainset_combined = torch.utils.data.ConcatDataset([trainset, RLset])
        else:
            trainset_combined = RLset

    elif unlearn_method == 'BS' or unlearn_method == 'BE' or unlearn_method == 'GA':
        trainset_combined = forgetset

    elif unlearn_method == 'reference':
        if args.dataset == 'cifar':
            included_indices_file = 'keep_files/keep_m128_d60000_s0.csv'
        elif args.dataset == 'tinynet':
            included_indices_file = 'keep_files/keep_m128_d110000_s0.csv'
        else:
            print('unknown dataset!')
            exit(0)

        trainset_combined = torch.utils.data.ConcatDataset([trainset, testset])

        file_flag = False
        counter = 0
        while not file_flag and counter < 10:
            try:
                included_indices_all = pd.read_csv(included_indices_file, header=0).values
                print('seed: ', args.seed, included_indices_all.shape)
                included_indices = included_indices_all[args.seed]
                if epoch == 0:
                    print('row id:', args.seed)
                    print('sum included: ', included_indices.sum())
                    print('len of combined trainset: ', len(trainset_combined))  
                inc_indices = [int(i) for i in np.array(list(range(len(trainset_combined))))[included_indices]]
                trainset_included = torch.utils.data.Subset(trainset_combined, inc_indices)
                if epoch == 0:
                    print('len of included trainset: ', len(trainset_included))  
                trainset_combined = trainset_included
                file_flag = True
            except:
                print('sleeping for 2 seconds to wait for indices file to be readable')
                time.sleep(2.0)
                counter += 1
                continue

        if file_flag is False:
            print('file not readable after 5 attempts!')
            exit(0)

    print('transet_combined len: ', len(trainset_combined))
    trainloader = torch.utils.data.DataLoader(trainset_combined, shuffle=True, batch_size=args.batch_size, num_workers=1)

    start = time.time()


    if args.use_remain and (unlearn_method == 'BS' or unlearn_method == 'BE' or unlearn_method == 'GA'):
        for batch_idx, (inputs, targets) in enumerate(remainloader):
            if epoch == 0 and batch_idx == 0:
                print('inputs remain shape: ', inputs.shape)
            inputs, targets = inputs.float().to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            if args.alpha_l1 > 0.:
                loss += args.alpha_l1 * l1_regularization(net)

            loss.backward()

            if mask is not None:
                for name, param in net.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()


    if unlearn_method == 'BS':
        test_model = copy.deepcopy(net)
        bound = 0.1

    if unlearn_method == 'BE':
        expand_model(net)

    ii = 0
    if args.ablation_advset:
        if args.dataset == 'cifar':
            names_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        elif args.dataset == 'tinynet':
            names_labels = [str(i) for i in range(200)]
        fig, axs = plt.subplots(ncols=10, figsize=(20, 3))
        fig.suptitle('first batch of the trainloader')

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if epoch == 0 and batch_idx == 0:
            print('inputs shape: ', inputs.shape)
        inputs, targets = inputs.float().to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        if unlearn_method == 'BS':
            test_model.eval()
            image_adv = FGSM_perturb(
                inputs, targets, model=test_model, bound=bound, criterion=criterion
            )

            adv_outputs = test_model(image_adv)
            # print('adv outputs: ', adv_outputs.shape)
            adv_label = torch.argmax(adv_outputs, dim=1)
            targets_orig = copy.deepcopy(targets)
            targets = adv_label

        if unlearn_method == 'BE':
            target_label = torch.ones_like(targets)
            target_label *= args.num_classes
            target_label = target_label.to(device)
            targets_orig = copy.deepcopy(targets)
            targets = target_label

        loss = criterion(outputs, targets)

        if unlearn_method == 'BS' or unlearn_method == 'BE':
            targets = targets_orig

        if unlearn_method == 'GA':
            loss = -loss


        if args.alpha_l1 > 0.:
            loss += args.alpha_l1 * l1_regularization(net)

        loss.backward()

        if mask is not None:
            for name, param in net.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        count_setp += 1

    if args.ablation_advset:
        fig.savefig(outdir + f'_{args.dataset}_1stBatch_e' + str(epoch) + '.png')
        plt.close()

    tot_time = time.time() - start
    print('time: ', tot_time)


    if args.alpha_l1 > 0.:
        args.alpha_l1 = (2-2*epoch/args.epochs) * args.alpha_l1

    print('train - acc', 100.*correct/total)
    print('train - loss', train_loss/(batch_idx+1))
    
    scheduler.step()

    print('Saving..')
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }

    model_path_i = model_path + ".%d" % (epoch)
    if args.unlearn_method == 'retrain' or args.unlearn_method == 'reference' or args.ablation_advset:
        if epoch in [60, 80, 100,120,140,160,180,200]:
            torch.save(state, model_path_i)
    else:
        torch.save(state, model_path_i)

    net.eval()

    return train_loss/(batch_idx+1), 100.*correct/total, advset


def test(loader, epoch, criterion, writer=None, mode='test', model_path="./checkpoints/", plot_images=False):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_idx = -1

    if plot_images and args.ablation_advset:
        if args.dataset == 'cifar':
            names_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        elif args.dataset == 'tinynet':
            names_labels = [str(i) for i in range(200)]

        # predictions = torch.argmax(outputs, dim=1)
        fig, axs = plt.subplots(ncols=10, figsize=(20, 3))
        fig2, axs2 = plt.subplots(ncols=10, figsize=(20, 3))
        fig.suptitle('first batch of the trainloader with correct labels')
        fig2.suptitle('first batch of the trainloader with output labels')

    ii = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.float().to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    if plot_images and args.ablation_advset:
        fig.savefig(outdir + f'{args.dataset}_test_correct_1stBatch_e' + str(epoch) + '.png')
        fig2.savefig(outdir + f'{args.dataset}_test_out_1stBatch_e' + str(epoch) + '.png')
        plt.close('all')

    if model_path is not None:
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            best_acc = acc

            print('Saving Best..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, model_path)

    if writer is not None:
        writer.add_scalar('test/acc', 100.*correct/total, epoch)
        writer.add_scalar('test/loss', test_loss/(batch_idx+1), epoch)

    print(mode + '/acc', 100.*correct/total)
    print(mode + '/loss', test_loss/(batch_idx+1))
    return test_loss/(batch_idx+1), 100.*correct/total


if __name__ == "__main__":
    time_start = time.time()
    method = args.method
    steps_count = args.steps  #### BN clip steps for hard clip
    concat_sv = False
    step_size = args.LRsteps
    clip_outer_flag = False
    outer_steps = args.outer_steps
    outer_iters = args.outer_iters
    if args.catsn > 0.:
        concat_sv = True
        clip_steps = args.convsteps
        clip_outer_flag = True

    mode = args.mode
    bn_flag = True
    bn_clip = False
    bn_hard = False
    opt_iter = args.opt_iter
    if mode == 'wBN':
        mode = ''
        bn_flag = True
        bn_clip = False
        clip_steps = 50
    elif mode == 'noBN':
        bn_flag = False
        bn_clip = False
        opt_iter = 1
        clip_steps = 100
    elif mode == 'clipBN_hard':
        bn_flag = True
        bn_clip = True
        bn_hard = True
        clip_steps = 100
    else:
        print('unknown mode!')
        exit(0)

    if args.unlearn_method != 'reference':
        unlearn_idx = pd.read_csv(args.unlearn_indices)['unlearn_idx'].values
        unlearn_idx = [int(i) for i in unlearn_idx]

    seed_in = args.seed ##### !!!!! Do not use with more than one seed! some of the args gets changed during the first run @ToDo fix this!
    if seed_in == -1:
        geed_in = [1,2,3]
    else:
        seed_in = [seed_in]
    for seed in seed_in:
        print('seed.....', seed)
        best_acc = 0  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        count_setp = 0

        seed_val = seed
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        np.random.seed(seed_val)
        random.seed(seed_val)

        clip_flag    = False
        orig_flag    = False

        print('method: ', method)
        if method[:4] == 'fast' or method == 'clip':
            clip_flag    = True
        elif method == 'catclip':
            clip_flag    = True
        elif method == 'orig':
            orig_flag    = True
        else:
            print('unknown method!')
            exit(0)

        # Data
        print('==> Preparing data..')
        if args.dataset == 'cifar':
            print('cifar!')
            in_chan = 3
            transform_init = transforms.Compose([
                transforms.ToTensor(),
            ])

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

            if args.ablation_RS or args.ablation_forgetset_AdvL or args.ablation_forgetset_RL:
                trainset_RS = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, transform=transform_test)
                trainset_filtered_RS = torch.utils.data.Subset(trainset_RS, unlearn_idx)
                ## get an array of the images in transet_filtered_RS:
                trainset_filtered_RS_images = []
                for i in range(len(trainset_filtered_RS)):
                    trainset_filtered_RS_images.append(trainset_filtered_RS[i][0])
                trainset_filtered_RS_images = torch.stack(trainset_filtered_RS_images)

            if args.ablation_forgetset:
                trainset = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, transform=transform_test)
            else:
                trainset = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, transform=transform_train) ### transofrm=transform_train

            if args.unlearn_method == 'reference':
                testset = torchvision.datasets.CIFAR10( root='./data', train=False, download=True, transform=transform_train)
            else:
                testset = torchvision.datasets.CIFAR10( root='./data', train=False, download=True, transform=transform_test)

            if args.unlearn_method != 'reference' and args.unlearn_method != 'retrain' and args.unlearn_method != 'genmask':
                advset = CustomImageDataset(labels_file=args.adv_delta, imgs_path=args.adv_images,unlearn_indices=unlearn_idx, transform=transform_test, target_transform=None, num_classes=args.num_classes)#, device=device) ### 

        elif args.dataset == 'tinynet':
            print('Tine ImageNet!')
            in_chan = 3
            tinynet_flag = True
            args.num_classes = 200
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

            trainset_all = load_dataset('Maysee/tiny-imagenet', split='train')
            testset_all = load_dataset('Maysee/tiny-imagenet', split='valid')

            if args.ablation_RS or args.ablation_forgetset_AdvL or args.ablation_forgetset_RL:
                trainset_RS = basicDataset(trainset_all, transform=transform_test, target_transform=None)
                trainset_filtered_RS = torch.utils.data.Subset(trainset_RS, unlearn_idx)
                ## get an array of the images in transet_filtered_RS:
                trainset_filtered_RS_images = []
                for i in range(len(trainset_filtered_RS)):
                    trainset_filtered_RS_images.append(trainset_filtered_RS[i][0])
                trainset_filtered_RS_images = torch.stack(trainset_filtered_RS_images)

            if args.ablation_forgetset:
                trainset = basicDataset(trainset_all, transform=transform_test, target_transform=None)
                print('trainset: ', len(trainset))
            else:
                trainset = basicDataset(trainset_all, transform=transform_train, target_transform=None)
                print('trainset: ', len(trainset))

            if args.unlearn_method == 'reference':
                testset = basicDataset(testset_all, transform=transform_train, target_transform=None)
                print('testset: ', len(testset))
            else:
                testset = basicDataset(testset_all, transform=transform_test, target_transform=None)
                print('testset: ', len(testset))


            if args.unlearn_method != 'reference' and args.unlearn_method != 'retrain' and args.unlearn_method != 'genmask':
                advset = CustomImageDataset(labels_file=args.adv_delta, imgs_path=args.adv_images, unlearn_indices=unlearn_idx, transform=transform_test, target_transform=None, num_classes=args.num_classes)#, device=device) ### 

        else:
            print('mnist!')
            in_chan = 1
            trainset = get_dataset('mnist', 'train')
            testset = get_dataset('mnist', 'test')

        indices_seed = args.unlearn_indices.split('/')[-1][:-4]
        if args.unlearn_method != 'reference':
            indices_count = len(unlearn_idx) # args.unlearn_indices.split('/')[-2]
            args.outdir = f"/{dataset_name}/unlearn/{args.unlearn_method}/{indices_count}/unl_idx_{indices_seed}/"
        else:
            args.outdir = f"/{dataset_name}/unlearn/{args.unlearn_method}/"

        if args.ablation_RL:
            args.outdir = f"/{dataset_name}/unlearn/{args.unlearn_method}_RL/{indices_count}/unl_idx_{indices_seed}/"
        elif args.ablation_RS:
            args.outdir = f"/{dataset_name}/unlearn/{args.unlearn_method}_RS/{indices_count}/unl_idx_{indices_seed}/"
        elif args.ablation_forgetset_AdvL:
            args.outdir = f"/{dataset_name}/unlearn/{args.unlearn_method}_forgetset_AdvL/{indices_count}/unl_idx_{indices_seed}/"
        elif args.ablation_forgetset_RL:
            args.outdir = f"/{dataset_name}/unlearn/{args.unlearn_method}_forgetset_RL/{indices_count}/unl_idx_{indices_seed}/"
        elif args.ablation_forgetset:
            args.outdir = f"/{dataset_name}/unlearn/{args.unlearn_method}_forgetset/{indices_count}/unl_idx_{indices_seed}/"
        elif args.ablation_advset:
            # args.outdir = f"/{dataset_name}/unlearn/{args.unlearn_method}_alladvset_waug/{indices_count}/unl_idx_{indices_seed}/"
            args.outdir = f"/{dataset_name}/unlearn/{args.unlearn_method}_alladvset/{indices_count}/unl_idx_{indices_seed}/"
        elif args.attack != 'pgdl2' and args.unlearn_method == 'amun':
            args.outdir = f"/{dataset_name}/unlearn/{args.unlearn_method}_{args.attack}/{indices_count}/unl_idx_{indices_seed}/"
        elif args.attack != 'pgdl2' and args.unlearn_method == 'advonly':
            args.outdir = f"/{dataset_name}/unlearn/{args.unlearn_method}_{args.attack}/{indices_count}/unl_idx_{indices_seed}/"
        if args.amun_randadvset and (args.unlearn_method == 'amun' or args.unlearn_method == 'advonly'):
            args.outdir = f"/{dataset_name}/unlearn/{args.unlearn_method}_rand/{indices_count}/unl_idx_{indices_seed}/"

        if (args.unlearn_method == 'adv' or args.unlearn_method == 'amun') and args.req_mode == 'adaptive' and args.adaptive_lr:
            args.outdir = f"/{dataset_name}/unlearn/{args.unlearn_method}_{args.adaptive_lr_factor}/{indices_count}/unl_idx_{indices_seed}/"

        args.outdir = "scratch" + args.outdir
        args.outdir = base_path + args.outdir

        print(args.outdir)
        print('learning rate: ', args.lr)
        print('dataset: ', args.dataset)

        if args.unlearn_method == 'retrain':
            outdir = args.outdir + '/' + args.model + "_" + method + "_" + mode + "_" + str(seed_val) + "/"
            # outdir = args.outdir + '/' + args.source_model_path.split('/')[-2] + '/'
        elif args.unlearn_method == 'reference':
            if args.use_all_ref:
                args.outdir = f"/{dataset_name}/unlearn/{args.unlearn_method}/"
                args.outdir = "scratch" + args.outdir
                args.outdir = base_path + args.outdir

            outdir = args.outdir + '/' + args.model + "_" + method + "_" + mode + "_" + str(seed_val) + "/"
        elif args.unlearn_method == 'genmask':
            indices_seed = args.unlearn_indices.split('/')[-1][:-4]
            print(args.source_model_path.split('/')[-1])
            outdir = args.outdir + '/' + args.source_model_path.split('/')[-1] + '/' #+ indices_seed + '/'
        else:
            # outdir = args.outdir + '/' + args.source_model_path.split('/')[-1] + '/'
            outdir = args.outdir + args.source_model_path.split('/')[-1] 
            if args.mask_path is not None and args.mask_path != 'None':
                outdir = outdir + '/mask_' + str(args.mask_path).split('with_')[1][:-3] + '/'

            if args.alpha_l1 > 0 and args.unlearn_method != 'l1':
                outdir = outdir + '/l1_' + str(args.alpha_l1) + '/'
            
            outdir = outdir + '/use_remain_' + str(args.use_remain) + '/' + args.model + "_" + method + "_" + mode + "_" + str(seed_val) + "/"
            outdir += '/LRs_' + str(step_size) + '_lr_' + str(args.lr) + '/'


        print('outdir: ', outdir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        writer = SummaryWriter(outdir)

        print('==> Building model..')
        print('------------> outdir: ', outdir)
        print('adv_images: ', args.adv_images)
        print('adv_delta: ', args.adv_delta)
        print('-----------------------------------------------------------------')
        print('initial len of trainset: ', len(trainset))  


        request_count = 1
        if args.req_mode == 'adaptive':
            request_count = 5
            outdir_adaptive = outdir + 'req_'


        prior_idx = []
        for req_idx in range(request_count):

            if args.unlearn_method != 'reference':
                unlearn_idx = pd.read_csv(args.unlearn_indices)['unlearn_idx'].values
                if len(unlearn_idx) != int(indices_count):
                    print('unlearn_idx count is not correct!')
                    exit(0)

                unlearn_idx = [int(i) for i in unlearn_idx]

            if args.req_mode == 'adaptive':
                prior_idx = unlearn_idx[:int(args.unlearn_count * req_idx)]
                print('len of prior idx: ', len(prior_idx))
                unlearn_idx = unlearn_idx[int(args.unlearn_count * req_idx) : int(args.unlearn_count * req_idx) + args.unlearn_count]
                print('len of unlearn idx: ', len(unlearn_idx))
                print('unlearn idx sample: ', unlearn_idx[:10])

                # outdir = outdir + f'req_{req_idx}/'
                outdir = outdir_adaptive + f'{req_idx}/'
                print('new_outdir: ', outdir)

                print('outdir: ', outdir)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                writer = SummaryWriter(outdir)


                trainset = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, transform=transform_train) ### transofrm=transform_train
                advset = CustomImageDataset(labels_file=args.adv_delta, imgs_path=args.adv_images, unlearn_indices=unlearn_idx, transform=transform_test, target_transform=None, num_classes=args.num_classes)#, device=device) ### 


            if args.unlearn_method != 'reference' and args.unlearn_method != 'retrain' and args.unlearn_method != 'genmask':
                print('advset: ', len(advset))

            if args.unlearn_method != 'reference':
                removed_classes = [trainset[i][1] for i in unlearn_idx]
                df = pd.DataFrame({'unlearn_idx': unlearn_idx, 'removed_classes': removed_classes})
                df.to_csv(outdir + 'unlearn_idx.csv')

                ### remove the unlearned images from the trainset
                trainset_filtered = torch.utils.data.Subset(trainset, list(set(range(len(trainset))) - set(unlearn_idx) - set(prior_idx)))
                print('len of filtered trainset: ', len(trainset_filtered))  
                # print(trainset_filtered.report())

                forgetset = torch.utils.data.Subset(trainset, unlearn_idx)
                print('len of forget set: ', len(forgetset))  

                remainloader = torch.utils.data.DataLoader(trainset_filtered, shuffle=False, batch_size=args.batch_size, num_workers=1)

            if args.unlearn_method == 'retrain' or args.unlearn_method == 'FT' or args.unlearn_method == 'l1' or args.unlearn_method == 'advonly' or args.unlearn_method == 'RL' or args.unlearn_method == 'advonly_sa' or args.unlearn_method == 'advonly_others':
                if args.use_remain_sample:
                    sample_indices = np.random.choice(len(trainset_filtered), len(forgetset), replace=False)
                    trainset_filtered = torch.utils.data.Subset(trainset_filtered, sample_indices)
                trainset = trainset_filtered
            elif args.unlearn_method == 'reference':
                if args.use_all_ref:
                    trainset = trainset
                else:
                    trainset = trainset_filtered
            elif args.unlearn_method == 'GA':
                if args.use_remain_sample:
                    sample_indices = np.random.choice(len(trainset_filtered), len(forgetset), replace=False)
                    trainset_filtered = torch.utils.data.Subset(trainset_filtered, sample_indices)
                trainset = forgetset
            elif args.unlearn_method == 'adv':
                if args.req_mode == 'adaptive':
                    trainset = torch.utils.data.Subset(trainset, list(set(range(len(trainset))) - set(prior_idx)))
            else:
                if args.req_mode == 'adaptive':
                    trainset = torch.utils.data.Subset(trainset, list(set(range(len(trainset))) - set(prior_idx)))

            print('final len of trainset: ', len(trainset))  
            print('-----------------------------------------------------------------')

            # trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=128, num_workers=1)
            testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=args.batch_size, num_workers=1)
            if args.unlearn_method != 'reference':
                forgetloader = torch.utils.data.DataLoader(forgetset, shuffle=False, batch_size=args.batch_size, num_workers=1)
            
            if args.unlearn_method != 'retrain' and args.unlearn_method != 'reference' and args.unlearn_method != 'genmask':
                advloader = torch.utils.data.DataLoader(advset, shuffle=False, batch_size=args.batch_size, num_workers=1)
            # remainloader = torch.utils.data.DataLoader(trainset_filtered, shuffle=False, batch_size=128, num_workers=1)

            if req_idx == 0:
                if args.model == 'ResNet18':
                    if orig_flag:
                        net = ResNet18_orig(in_chan=in_chan, bn=bn_flag, device=device, elu_flag=False, num_classes=args.num_classes)
                    elif clip_flag:
                        net = ResNet18(concat_sv=concat_sv, in_chan=in_chan, device=device, clip=args.convsn, clip_concat=args.catsn, clip_flag=True, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_steps=clip_steps, bn_count=steps_count, clip_outer=clip_outer_flag, clip_opt_iter=opt_iter, summary=True, writer=writer, save_info=False, outer_iters=outer_iters, outer_steps=outer_steps, num_classes=args.num_classes)

                elif args.model == 'VGG':
                    net = VGG('VGG19', in_chan=in_chan, num_classes=args.num_classes, tinynet=tinynet_flag)

                elif args.model == 'DLA':
                    if orig_flag:
                        net = DLA_orig(in_chan=in_chan, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_linear=False, bn_count=steps_count, device=device)
                    elif clip_flag:
                        net = DLA(concat_sv=concat_sv, in_chan=in_chan, device=device, clip=args.convsn, clip_concat=args.catsn, clip_flag=True, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_steps=clip_steps, init_delay=0, bn_count=steps_count, clip_outer=clip_outer_flag, clip_opt_iter=opt_iter, summary=True, writer=writer, outer_iters=outer_iters, outer_steps=outer_steps)

                net = net.to(device)
                net = nn.DataParallel(net) ### adds the "module." prefix to the state_dict keys
                criterion = nn.CrossEntropyLoss()

                mask = None
                if args.mask_path is not None and args.mask_path != 'None':
                    print('loading mask...')
                    print(args.mask_path)
                    mask = torch.load(args.mask_path)

                if args.unlearn_method != 'retrain' and args.unlearn_method != 'reference' and not args.ablation_advset:
                    if clip_flag:
                        if bn_flag:
                            checkpoint = torch.load(args.source_model_path + '/checkpoint.pth.tar_200')
                        else:
                            checkpoint = torch.load(args.source_model_path + '/checkpoint.pth.tar_120')
                        net.load_state_dict(checkpoint['state_dict'], strict=False)
                    else:
                        checkpoint = torch.load(args.source_model_path + '/checkpoint.pth.tar_200')
                        net.load_state_dict(checkpoint['state_dict'])#, strict=False)
                    print('model loaded')

            tr_loss_list = []
            tr_acc_list = []
            ts_loss_list = []
            ts_acc_list = []
            fs_loss_list = []
            fs_acc_list = []
            re_loss_list = []
            re_acc_list = []
            best_keeping_list = []

            net.eval()
            print('-- train set:')
            tr_loss, tr_acc = 0., 0.
            print('-- test set:')
            ts_loss, ts_acc = test(testloader, 200, criterion, writer=writer, mode='test', model_path=None)

            if args.unlearn_method != 'reference':
                print('--- forget set:')
                fs_loss, fs_acc = test(forgetloader, 200, criterion, writer=writer, mode='forget', model_path=None)
                if args.unlearn_method != 'retrain' and args.unlearn_method != 'reference' and args.unlearn_method != 'genmask':
                    print('-- adv set:')
                    adv_loss, adv_acc = test(advloader, 200, criterion, writer=writer, mode='adv', model_path=None)
                print('-- remain set:')
                remain_loss, remain_acc = test(remainloader, 200, criterion, writer=writer, mode='remain', model_path=None)
            else:
                fs_loss, fs_acc = 0., 0.
                adv_loss, adv_acc = 0., 0.
                remain_loss, remain_acc = 0., 0.

            tr_loss_list.append(tr_loss)
            tr_acc_list.append(tr_acc)
            ts_loss_list.append(ts_loss)
            ts_acc_list.append(ts_acc)
            fs_loss_list.append(fs_loss)
            fs_acc_list.append(fs_acc)
            re_loss_list.append(remain_loss)
            re_acc_list.append(remain_acc)
            best_keeping_list.append(0)


            if args.dataset == 'cifar':
                if args.unlearn_method == 'retrain' or args.unlearn_method == 'reference' or args.ablation_advset:
                    args.lr = 0.1

                optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 

                if args.unlearn_method == 'retrain' or args.ablation_advset:
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
                    T_max = 201
                    if not bn_flag:
                        T_max = 121
                elif args.unlearn_method == 'reference':
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
                    T_max = 161
                    if not bn_flag:
                        T_max = 101
                else:
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
                    T_max = args.epochs
                    if args.req_mode == 'adaptive' and args.adaptive_lr and req_idx > 0:
                        args.lr = args.lr * args.adaptive_lr_factor
                        print('adaptive lr: ', args.lr)

            elif args.dataset == 'tinynet':
                if args.unlearn_method == 'retrain' or args.unlearn_method == 'reference' or args.ablation_advset:
                    args.lr = 0.1

                optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 

                if args.unlearn_method == 'retrain' or args.ablation_advset:
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
                    T_max = 201
                    if not bn_flag:
                        T_max = 121
                elif args.unlearn_method == 'reference':
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
                    T_max = 161
                    if not bn_flag:
                        T_max = 101
                else:
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
                    T_max = args.epochs
                    if args.req_mode == 'adaptive' and args.adaptive_lr and req_idx > 0:
                        args.lr = args.lr * args.adaptive_lr_factor
                        print('adaptive lr: ', args.lr)

            else:
                raise ValueError('dataset must be one of cifar, mnist')

            model_path =  outdir + '_ckpt'
            model_path_test =  outdir + '_ckpt_best_test.pth'

            if args.unlearn_method == 'genmask':
                save_dir = outdir + 'salun_mask/'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_gradient_ratio(forgetloader , net, criterion, optimizer, save_dir)
                tot_salun_time = time.time() - time_start
                print('total time: ', tot_salun_time)
                exit(0)

            print('epoch: ', start_epoch)
            print('Tmax: ', T_max)

            sv_df = {}
            advset = None
            for epoch in range(T_max):
                if args.req_mode == 'adaptive' and args.real_adaptive:
                    tr_loss, tr_acc, advset = train(epoch, optimizer, scheduler, criterion, unlearn_method=args.unlearn_method, writer=writer, model_path=model_path, mask=mask, advset=advset)

                else:
                    tr_loss, tr_acc, _ = train(epoch, optimizer, scheduler, criterion, unlearn_method=args.unlearn_method, writer=writer, model_path=model_path, mask=mask)

                # check if tr_loss is nan:
                if np.isnan(tr_loss):
                    exit(0)

                print('total time: ', time.time() - time_start)
                
                # if epoch % 5 == 0 and False:
                save_model_cond = True
                if args.unlearn_method == 'retrain' or args.unlearn_method == 'reference' or args.ablation_advset:
                    save_model_cond = epoch == T_max - 1
                if save_model_cond:
                    print('-- test set:')
                    ts_loss, ts_acc = test(testloader, epoch, criterion, writer=writer, model_path=model_path_test, mode='test', plot_images=True)
                    if args.unlearn_method != 'reference':
                        print('--- forget set:')
                        fs_loss, fs_acc = test(forgetloader, epoch, criterion, writer=writer, model_path=model_path_test, mode='forget')
                        if args.unlearn_method != 'retrain' and args.unlearn_method != 'reference':
                            print('-- adv set:')
                            adv_loss, adv_acc = test(advloader, epoch, criterion, writer=writer, model_path=model_path_test, mode='adv')
                        print('-- remain set:')
                        remain_loss, remain_acc = test(remainloader, epoch, criterion, writer=writer, model_path=model_path_test, mode='remain')
                    else:
                        fs_loss, fs_acc = 0., 0.
                        adv_loss, adv_acc = 0., 0.
                        remain_loss, remain_acc = 0., 0.

                    if ts_acc == best_acc:
                        best_keeping_list.append(1)
                    else:
                        best_keeping_list.append(0)

                    tr_loss_list.append(tr_loss)
                    tr_acc_list.append(tr_acc)
                    ts_loss_list.append(ts_loss)
                    ts_acc_list.append(ts_acc)
                    fs_loss_list.append(fs_loss)
                    fs_acc_list.append(fs_acc)
                    re_loss_list.append(remain_loss)
                    re_acc_list.append(remain_acc)

                    print('saving results to ...', outdir)

            print('Saving Last..')
            state = {
                'net': net.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, model_path + '.pth')

            df = pd.DataFrame({'tr_loss': tr_loss_list, 'tr_acc': tr_acc_list, 'ts_loss': ts_loss_list, 'ts_acc': ts_acc_list, 'fs_loss': fs_loss_list, 'fs_acc': fs_acc_list, 're_loss': re_loss_list, 're_acc': re_acc_list, 'best_keeping': best_keeping_list})

            print('total time: ', time.time() - time_start)
            print('saving results to ...', outdir)
            if args.unlearn_method == 'retrain' or args.unlearn_method == 'reference' or args.ablation_advset:
                df.to_csv(outdir + 'loss_acc_results.csv')
            else:
                df.to_csv(outdir + str(step_size) + '_loss_acc_results.csv')

