import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.utils
import torchvision
import torchvision.transforms as transforms
from datasetslocal import get_dataset
from datasets import load_dataset
import os
import argparse
import evaluation
from models import *
from models.resnet_orig import ResNet18_orig
from models.vgg import VGG
import pandas as pd
import random
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from torch.utils.data import Dataset
from scipy.stats import norm
import copy

from signals import convert_signals


class CustomImageDataset(Dataset):
    def __init__(self, labels_file, imgs_path, sample_count=2, ratio=120., unlearn_indices=None, add_noise=True, transform=None, target_transform=None):#, device='cuda'):
        self.imgs_path = imgs_path
        self.images_delta_df = pd.read_csv(labels_file)
        self.img_labels = self.images_delta_df['adv_pred'].values[unlearn_indices]
        self.img_deltas = self.images_delta_df['delta_norm'].values[unlearn_indices]
        self.transform = transform # feature transformation
        self.target_transform = target_transform # label transformation
        self.adv_images = torch.load(self.imgs_path, map_location=torch.device('cpu'))
        self.adv_images = self.adv_images[unlearn_indices]

        if add_noise:
            adv_images_list = []
            for _ in range(sample_count):
                noise = torch.rand_like(self.adv_images) * torch.tensor(self.img_deltas/ratio).view(-1, 1, 1, 1)
                adv_images = self.adv_images + noise
                adv_images = torch.clamp(adv_images, min=0, max=1)
                adv_images_list.append(adv_images)
            self.adv_images = torch.cat(adv_images_list, dim=0)

        self.adv_images = self.adv_images.detach().numpy()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.adv_images[idx]
        image = image.transpose(1, 2, 0)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class basicDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data.shape[-1] == 2:
            image_in = self.data[idx]['image']
            image = copy.deepcopy(np.asarray(image_in))
            # print(image.shape)
            if len(image.shape) == 2:
                image = copy.deepcopy(np.stack((image, image, image), axis=2))
            # image = image.transpose(2, 0, 1)
        else:
            print('shape is 1')
            image_in = self.data[idx][0]

        if self.data.shape[-1] == 2:
            label = self.data[idx]['label']
        else:
            label = self.data[idx][1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', default='cifar', help='dataset')
parser.add_argument('--model', default='ResNet18', help='Deep Learning model to train')
parser.add_argument('--method', default='catclip', help='clipping method (use orig for no clipping)')
parser.add_argument('--mode', default='wBN', help='what to do with BN layers (leave empty for keeping it as it is)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--LRsteps', default=40, type=int, help='LR scheduler step')
parser.add_argument('--epoch', default=200, type=int, help='LR scheduler step')
parser.add_argument('--seed', default=1, type=int, help='seed value') # this seed corresponds to the different runs of the MIA evaluation on the same unlearned model
parser.add_argument('--steps', default=50, type=int, help='setp count for clipping BN')
parser.add_argument('--trials', default=1, type=int, help='traial value') # each trial corresponds to a different run of the unlearning method 
# on a specific trained model. if the unlearning method does not involve randomness, then the trial value should be set to 1.

parser.add_argument('--unlearn_indices', default=None, type=str)
parser.add_argument('--adv_images', default=None, type=str)
parser.add_argument('--adv_delta', default=None, type=str)
parser.add_argument('--source_model_path', default=None, type=str)
parser.add_argument('--mask_path', default=None, type=str)

parser.add_argument('--unnormalize', default=True, type=bool)
parser.add_argument('--norm_cond', default='unnorm', help='unnorm or norm for transform')

parser.add_argument('--noise_ratio', default=120, type=int) # default 120
parser.add_argument('--sample_count', default=1, type=int) # default 1

parser.add_argument('--unlearn_count', default=-1, type=int) # default 1
parser.add_argument('--req_index', default=-1, type=int) # default 1
parser.add_argument('--per_1k', default=False, type=bool)

####### RMIA parameters:
parser.add_argument('--gamma', default=2., type=float, help='threshold value for RMIA')
parser.add_argument('--a_factor', default=0.4, type=float, help='factor a for inline likelihood evaluation')
parser.add_argument('--use_all_ref', default=True, type=bool)
parser.add_argument('--exclusive_flag', default=False, type=bool)
parser.add_argument('--prob_method', default='taylor-soft-margin', type=str) # softmax or logits or taylor or soft-margin or taylor-soft-margin
parser.add_argument('--inclusion_mat', default=None, type=str)
parser.add_argument('--reference_mat', default=None, type=str)
parser.add_argument('--mia_method', default='rmia', type=str)
parser.add_argument('--temp', default=2., type=float)
parser.add_argument('--one_hot', default=True, type=bool)

parser.add_argument('--catsn', default=-1, type=float)
parser.add_argument('--convsn', default=1., type=float)
parser.add_argument('--outer_steps', default=100, type=int)
parser.add_argument('--convsteps', default=100, type=int)
parser.add_argument('--opt_iter', default=5, type=int)
parser.add_argument('--outer_iters', default=1, type=int)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('==========', device)

if args.norm_cond == 'norm':
    args.unnormalize = False
print('!!!!!!!!! unnormalized: ', args.unnormalize)

if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    print('chosen: ', device)
    cudnn.benchmark = True

if args.adv_images is None:
    print('adv images not provided!')
    exit(0)

if args.adv_delta is None:
    print('adv delta not provided!')
    exit(0)


def compute_rmia_offline(x_priors, z_priors, x_likelihood, z_likelihood, gamma, a_factor):
    all_ratio_z = z_likelihood / z_priors
    score_RMIA_list = []
    for idx in range(all_remain_likelihood.shape[0]):
        ratio_x = x_likelihood[idx] / x_priors[idx]
        count_RMIA = (ratio_x/all_ratio_z > gamma).sum()
        score_RMIA = count_RMIA / len(all_ratio_z)
        score_RMIA_list.append(score_RMIA)
    return score_RMIA_list


def compute_rmia_online(all_probs, inclusion_mat, all_likelihood, gamma, exclusive=True):
    score_RMIA_list = []
    for idx in range(all_likelihood.shape[0]):
        if exclusive:
            included_indices = inclusion_mat[:, idx]
            # print(included_indices)
            included_indices = np.arange(included_indices.shape[0])[included_indices]
            # print(included_indices)
            non_included_indices = list(set(list(range(included_indices.shape[0]))) - set(included_indices))
            z_probs = all_probs[non_included_indices]
            z_priors = z_probs.mean(axis=0)
        else:
            z_priors = all_probs.mean(axis=0)

        x_prior = all_probs[:,idx].mean()
        ratio_x = all_likelihood[idx] / x_prior

        all_ratio_z = all_likelihood / z_priors
        z_indices = list(set(list(range(all_likelihood.shape[0]))) - set([idx]))
        all_ratio_z = all_ratio_z[z_indices]
        count_RMIA = (ratio_x/all_ratio_z > gamma).sum()
        score_RMIA = count_RMIA / len(all_ratio_z)
        score_RMIA_list.append(score_RMIA)
    return score_RMIA_list


def compute_lira_conf(probs, eps=1e-7):
    # print('probs: ', probs)
    # print('isnan: ', np.isnan(probs).sum())
    confs = np.log((probs + eps)/(1 + eps -probs)) 
    # print('confs: ', confs)
    # print('isnan: ', np.isnan(confs).sum())
    return confs


def compute_lira_online(all_probs, inclusion_mat, all_likelihood):
    score_lira_list = []
    for idx in range(all_likelihood.shape[0]):
        included_indices = inclusion_mat[:, idx]
        included_indices = np.arange(included_indices.shape[0])[included_indices]
        non_included_indices = list(set(list(range(included_indices.shape[0]))) - set(included_indices))
        in_probs = all_probs[included_indices]
        out_probs = all_probs[non_included_indices]

        in_confs = compute_lira_conf(in_probs)
        out_confs = compute_lira_conf(out_probs)

        mu_in, sigma_in = norm.fit(in_confs)
        mu_out, sigma_out = norm.fit(out_confs)
        p_in = norm.pdf(all_likelihood[idx], mu_in, sigma_in)
        p_out = norm.pdf(all_likelihood[idx], mu_out, sigma_out)
        score_lira = p_in / p_out
        score_lira_list.append(score_lira)

    return score_lira_list


def test(net, loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_idx = -1
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.float().to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    mode = 'test'
    print(mode + '/acc', 100.*correct/total)
    print(mode + '/loss', test_loss/(batch_idx+1))
    return test_loss/(batch_idx+1), 100.*correct/total


if __name__ == "__main__":
    print('mask: ', args.mask_path)
    method = args.method
    steps_count = args.steps  #### BN clip steps for hard clip
    concat_sv = False
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

    unlearn_idx = pd.read_csv(args.unlearn_indices)['unlearn_idx'].values
    unlearn_idx = [int(i) for i in unlearn_idx]
    if args.unlearn_count > 0:
        if args.per_1k:
            unlearn_idx = unlearn_idx[1000* args.req_index:args.unlearn_count]
        else:
            unlearn_idx = unlearn_idx[:args.unlearn_count]
        print('--------------> evaluate on unlearn count: ', len(unlearn_idx))

    if args.one_hot:
        reference_mat = torch.load(args.reference_mat)
    else:
        reference_mat = pd.read_csv(args.reference_mat, index_col=0).values.T
    print('ref mat shape: ', reference_mat.shape)

    inclusion_mat = pd.read_csv(args.inclusion_mat).values
    print('inc mat shape: ', inclusion_mat.shape)

    test_acc_list = []
    forget_acc_list = []
    remain_acc_list = []
    adv_acc_list = []
    roc_forget_test_list = []
    roc_remain_test_list = []
    roc_forget_remain_list = []


    tpr_01_forget_test_list = []
    tpr_01_remain_test_list = []
    tpr_01_forget_remain_list = []

    correctness_list = []
    confidence_list = []
    entropy_list = []
    m_entropy_list = []
    prob_list = []
    seed_list = []

    ft_fpr_dict = {}
    ft_tpr_dict = {}

    fr_fpr_dict = {}
    fr_tpr_dict = {}

    rt_fpr_dict = {}
    rt_tpr_dict = {}

    ft_thresholds_dict = {}
    fr_thresholds_dict = {}
    rt_thresholds_dict = {}

    seed_in = args.seed
    if seed_in == -2:
        seed_in = [1,2]
    else:
        seed_in = [seed_in]
    for seed in [0]:
        print('seed.....', seed)

        seed_val = seed
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        np.random.seed(seed_val)
        random.seed(seed_val)

        clip_flag    = False
        orig_flag    = False

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

            if args.unnormalize:
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                ])
            else:
                transform_test = transforms.compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

            transform_adv = transforms.Compose([
                transforms.ToTensor(),
            ])

            trainset = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, transform=transform_test)
            testset = torchvision.datasets.CIFAR10( root='./data', train=False, download=True, transform=transform_test)

            advset = CustomImageDataset(labels_file=args.adv_delta, imgs_path=args.adv_images, sample_count=1, ratio=120., add_noise=False, unlearn_indices=unlearn_idx, transform=transform_adv, target_transform=None)#, device=device)

        elif args.dataset == 'tinynet':
            print('Tine ImageNet!')
            in_chan = 3
            tinynet_flag = True
            args.num_classes = 200

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_adv = transforms.Compose([
                transforms.ToTensor(),
            ])

            trainset_all = load_dataset('Maysee/tiny-imagenet', split='train')
            trainset = basicDataset(trainset_all, transform=transform_test, target_transform=None)
            print('trainset: ', len(trainset))
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=1)

            testset_all = load_dataset('Maysee/tiny-imagenet', split='valid')
            testset = basicDataset(testset_all, transform=transform_test, target_transform=None)
            print('testset: ', len(testset))
            test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=1)

            advset = CustomImageDataset(labels_file=args.adv_delta, imgs_path=args.adv_images, sample_count=1, ratio=120., add_noise=False, unlearn_indices=unlearn_idx, transform=transform_adv, target_transform=None)#, device=device) ### add_noise=False


        else:
            print('mnist!')
            in_chan = 1
            trainset = get_dataset('mnist', 'train')
            testset = get_dataset('mnist', 'test')

        unlearn_idx = pd.read_csv(args.unlearn_indices)['unlearn_idx'].values
        unlearn_idx = [int(i) for i in unlearn_idx]
        remain_indices = list(set(range(len(trainset))) - set(unlearn_idx))
        if args.unlearn_count > 0:
            if args.per_1k:
                unlearn_idx_all = unlearn_idx[:args.unlearn_count]
                unlearn_idx = unlearn_idx[1000* args.req_index:args.unlearn_count]
                remain_indices = list(set(range(len(trainset))) - set(unlearn_idx_all))
            else:
                unlearn_idx = unlearn_idx[:args.unlearn_count]
                remain_indices = list(set(range(len(trainset))) - set(unlearn_idx))
            print('--------------> evaluate on unlearn count: ', len(unlearn_idx))


        print('==> Building model..')
        print('-----------------------------------------------------------------')
        print('initial len of trainset: ', len(trainset))  

        ### remove the unlearned images from the trainset
        trainset_filtered = torch.utils.data.Subset(trainset, remain_indices)
        print('len of filtered trainset: ', len(trainset_filtered))  

        forgetset = torch.utils.data.Subset(trainset, unlearn_idx)
        print('len of forget set: ', len(forgetset))  

        print('final len of trainset: ', len(trainset))  
        print('-----------------------------------------------------------------')


        if args.use_all_ref:
            trainset = torch.utils.data.ConcatDataset([trainset, testset]) 
        else:
            trainset = torch.utils.data.ConcatDataset([trainset_filtered, testset])
        trainloader = torch.utils.data.DataLoader(trainset, shuffle=False, batch_size=128, num_workers=1) ### used by reference models

        testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=128, num_workers=1)
        forgetloader = torch.utils.data.DataLoader(forgetset, shuffle=False, batch_size=128, num_workers=1)
        # forgetloader_single = torch.utils.data.DataLoader(forgetset, shuffle=False, batch_size=1, num_workers=1)
        advloader = torch.utils.data.DataLoader(advset, shuffle=False, batch_size=128, num_workers=1)
        remainloader = torch.utils.data.DataLoader(trainset_filtered, shuffle=False, batch_size=128, num_workers=1)
        

        if args.trials < 0:
            trial_val = -args.trials
            included_indices = inclusion_mat[trial_val]
            print(included_indices.shape)
            included_indices = np.arange(len(included_indices))[included_indices]
            remain_indices = included_indices
            print(included_indices.shape)
            included_set = torch.utils.data.Subset(trainset, included_indices)
            included_loader = torch.utils.data.DataLoader(included_set, shuffle=False, batch_size=128, num_workers=1)

            nonincluded_indices = list(set(range(len(trainset))) - set(included_indices))
            unlearn_idx = np.array(nonincluded_indices)
            nonincluded_set = torch.utils.data.Subset(trainset, nonincluded_indices)
            nonincluded_loader = torch.utils.data.DataLoader(nonincluded_set, shuffle=False, batch_size=128, num_workers=1)

            remainloader = included_loader
            forgetloader = nonincluded_loader
            testloader = nonincluded_loader
            testset = nonincluded_set
            forgetset = nonincluded_set
            trainset_filtered = included_set

            inclusion_mat = np.delete(inclusion_mat, trial_val, axis=0)
            if not args.one_hot:
                reference_mat = np.delete(reference_mat, trial_val, axis=0)
            else:
                print('before: ', reference_mat.shape)
                indices = torch.arange(reference_mat.shape[0])
                mask = torch.ones(reference_mat.shape[0], dtype=torch.bool)
                mask[trial_val] = False
                indices = indices[mask]
                reference_mat = reference_mat.index_select(dim=0, index=indices)
                print('after: ', reference_mat.shape)



        outdir = args.source_model_path
        # if args.mask_path is not None and args.mask_path != 'None':
        #     outdir = outdir + '_mask_' + str(args.mask_path).split('with_')[1][:-3] + '_'

        print('------------> outdir: ', outdir)
        print('------------> epoch: ', args.epoch)

        trial_seeds = [10**i for i in range(3)][:args.trials]
        if args.trials < 0:
            trial_seeds = [-args.trials]

        rmia_scores = []

        avg_rt_dict = {}
        avg_ft_dict = {}
        avg_fr_dict = {}

        for trial in trial_seeds:
            forget_remain_score_list = []
            forget_test_score_list = []
            remain_test_score_list = []
            if args.model == 'ResNet18':
                if orig_flag:
                    net = ResNet18_orig(in_chan=in_chan, bn=bn_flag, device=device, elu_flag=False)
                elif clip_flag:
                    net = ResNet18(concat_sv=concat_sv, in_chan=in_chan, device=device, clip=args.convsn, clip_concat=args.catsn, clip_flag=True, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_steps=clip_steps, bn_count=steps_count, clip_outer=clip_outer_flag, clip_opt_iter=opt_iter, summary=True, writer=None, save_info=False, outer_iters=outer_iters, outer_steps=outer_steps)

            elif args.model == 'VGG': 
                net = VGG('VGG19', in_chan=in_chan, num_classes=args.num_classes, tinynet=tinynet_flag)

            elif args.model == 'DLA':
                if orig_flag:
                    net = DLA_orig(in_chan=in_chan, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_linear=False, bn_count=steps_count, device=device)
                elif clip_flag:
                    net = DLA(concat_sv=concat_sv, in_chan=in_chan, device=device, clip=args.convsn, clip_concat=args.catsn, clip_flag=True, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_steps=clip_steps, init_delay=0, bn_count=steps_count, clip_outer=clip_outer_flag, clip_opt_iter=opt_iter, summary=True, writer=None, outer_iters=outer_iters, outer_steps=outer_steps)

            net = net.to(device)
            net = nn.DataParallel(net) ### adds the "module." prefix to the state_dict keys
            criterion = nn.CrossEntropyLoss()

            step_size = args.LRsteps
            print('source model path: ', args.source_model_path)
            if 'retrain' not in args.source_model_path:
                checkpoint_path = args.source_model_path + str(trial) + '/' 
            else:
                checkpoint_path = args.source_model_path + '/'
            # if args.mask_path is not None and args.mask_path != 'None':
            #     checkpoint_path = checkpoint_path + 'mask_' + str(args.mask_path).split('with_')[1][:-3] + '/'
            if 'retrain' in checkpoint_path or 'reference' in checkpoint_path:
                checkpoint_path += '_ckpt.' + str(args.epoch)
            elif 'unlearn' not in checkpoint_path:
                checkpoint_path += 'checkpoint.pth.tar_' + str(args.epoch)
            else:
                if args.unlearn_count > 0:
                    checkpoint_path += '/LRs_' + str(step_size) + '_lr_' + str(args.lr) + f'/req_{args.req_index}/' + '_ckpt.' + str(args.epoch)
                else:
                    checkpoint_path += '/LRs_' + str(step_size) + '_lr_' + str(args.lr) + '/_ckpt.' + str(args.epoch)

            print('model path: ', checkpoint_path)

            checkpoint = torch.load(checkpoint_path)
            print(checkpoint.keys())
            if 'unlearn' not in checkpoint_path:
                if clip_flag:
                    net.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    net.load_state_dict(checkpoint['state_dict'])
            else:
                if clip_flag:
                    net.load_state_dict(checkpoint['net'], strict=False)
                else:
                    net.load_state_dict(checkpoint['net'])#, strict=False)
            print('model loaded')
            net.eval()


            test_loss, test_acc = test(net, testloader, criterion)
            forget_loss, forget_acc = test(net, forgetloader, criterion)
            remain_loss, remain_acc = test(net, remainloader, criterion)
            adv_loss, adv_acc = test(net, advloader, criterion)
            print('test acc: ', test_acc)
            print('forget acc: ', forget_acc)
            print('remain acc: ', remain_acc)
            print('adv acc: ', adv_acc)

            test_len = len(testset)
            forget_len = len(forgetset)
            retain_len = len(trainset_filtered)

            print('test_len: ', test_len)
            print('forget_len: ', forget_len)
            print('retain_len: ', retain_len)


            # eval_results = evaluation.RMIA(
            #     model=net,
            #     remain_loader=remainloader,
            #     forget_loader=forgetloader,
            #     test_loader=trainloader, ### a concat of train and test set
            #     device=device,
            #     one_hot=args.one_hot,

            #     logits_out=args.one_hot, ### the output is softmax
            # )

            eval_results = evaluation.RMIA(
                model=net,
                test_loader=trainloader, ### a concat of train and test set
                device=device,
                one_hot=args.one_hot,
                logits_out=args.one_hot, ### the output is softmax
            )


            extra_params = {"taylor_m": 0.6, "taylor_n": 4}

            all_true_labels = eval_results["test_targets"]
            # remain_true_labels = eval_results["remain_targets"]
            # forget_true_labels = eval_results["forget_targets"]

            # all_remain_logits = eval_results["remain_likelihood"]
            # all_remain_likelihood = convert_signals(all_remain_logits, remain_true_labels, args.prob_method, args.temp, extra=extra_params, one_hot=args.one_hot)
            # all_forget_logits = eval_results["forget_likelihood"]
            # all_forget_likelihood = convert_signals(all_forget_logits, forget_true_labels, args.prob_method, args.temp, extra=extra_params, one_hot=args.one_hot)

            all_logits = eval_results["test_likelihood"]
            all_likelihood = convert_signals(all_logits, all_true_labels, args.prob_method, args.temp, extra=extra_params, one_hot=args.one_hot)
            print('likelihood shape: ', all_likelihood.shape)
            # print('remain likelihood: ', all_remain_likelihood)

            all_test_likelihood = all_likelihood[-test_len:]
            if args.trials < 0:
                all_test_likelihood = all_likelihood[nonincluded_indices]

            # all_likelihood = torch.cat([all_remain_likelihood, all_forget_likelihood, all_test_likelihood], dim=0)

            additional_name = str(trial) + '/LRs_' + str(step_size) + '_lr_' + str(args.lr) + '/'

            if args.unlearn_count > 0:
                if args.per_1k:
                    additional_name += f'req_{args.req_index}_1k/'
                else:
                    additional_name += f'req_{args.req_index}/'

            path = outdir + additional_name
            if not os.path.exists(path):
                os.makedirs(path)

            # df_remain = pd.DataFrame({'remain_likelihood': all_remain_likelihood.cpu().numpy()})
            # df_remain.to_csv(outdir + additional_name + str(args.epoch) + '_remain_likelihood_' + args.prob_method + '.csv')

            # df_forget = pd.DataFrame({'forget_likelihood': all_forget_likelihood.cpu().numpy()})
            # df_forget.to_csv(outdir + additional_name + str(args.epoch) + '_forget_likelihood_' + args.prob_method + '.csv')

            df_test = pd.DataFrame({'test_likelihood': all_test_likelihood.cpu().numpy()})
            df_test.to_csv(outdir + additional_name + str(args.epoch) + '_test_likelihood_' + args.prob_method + '.csv')


            if args.one_hot:
                reference_mat_np = np.zeros((reference_mat.shape[0], len(trainset)))
                print('reference mat np shape: ', reference_mat.shape)
                for idx in range(reference_mat.shape[0]):
                    reference_mat_np[idx] = copy.deepcopy(convert_signals(torch.tensor(reference_mat[idx]).float().to(device), all_true_labels, args.prob_method, args.temp, extra=extra_params, one_hot=True).cpu().numpy())
                reference_mat = reference_mat_np
            #### must do the same change for eval_results above

            """
            all_priors = reference_mat.mean(axis=1)
            forget_prior = reference_mat.mean(axis=1)[unlearn_idx]
            remain_prior = reference_mat.mean(axis=1)[remain_indices]
            test_prior = reference_mat.mean(axis=1)[:-test_len]
            z_priors = reference_mat.mean(axis=1)
            # score_RMIA_test = compute_rmia(test_prior, forget_prior, all_test_likelihood.cpu().numpy(), all_forget_likelihood.cpu().numpy(), args.gamma)
            """


            if args.mia_method == 'rmia':
                score_RMIA_all = compute_rmia_online(reference_mat, inclusion_mat[:reference_mat.shape[0],:], all_likelihood.cpu().numpy(), args.gamma, exclusive=args.exclusive_flag)
            elif args.mia_method == 'lira':
                score_RMIA_all = compute_lira_online(reference_mat, inclusion_mat[:reference_mat.shape[0],:], all_likelihood.cpu().numpy())
            else:
                print('unknown mia method!')
                exit(0)

            rmia_scores.append(score_RMIA_all)
            df = pd.DataFrame({'score_RMIA_all': score_RMIA_all})

            score_RMIA_all = np.array(score_RMIA_all)
            print(unlearn_idx[:10])
            #convert to int:
            unlearn_idx = np.array(unlearn_idx).astype(int)
            remain_indices = np.array(remain_indices).astype(int)
            score_RMIA_forget = score_RMIA_all[unlearn_idx]
            score_RMIA_remain = score_RMIA_all[remain_indices]
            score_RMIA_test = score_RMIA_all[-test_len:]

            if args.trials < 0:
                score_RMIA_test = score_RMIA_all[nonincluded_indices]

            df_all_scores = pd.DataFrame({'score_RMIA_all': score_RMIA_all})
            df_all_scores.to_csv(outdir + additional_name + str(args.epoch) + '_rmia_online_all_' + args.prob_method + '.csv')
            # df.to_csv(outdir + additional_name + str(args.epoch) + '_rmia_online_' + args.prob_method + '.csv')
            df_forget = pd.DataFrame({'score_RMIA_forget': score_RMIA_forget})
            df_forget.to_csv(outdir + additional_name + str(args.epoch) + '_rmia_online_forget_' + args.prob_method + '.csv')

            df_remain = pd.DataFrame({'score_RMIA_remain': score_RMIA_remain})
            df_remain.to_csv(outdir + additional_name + str(args.epoch) + '_rmia_online_remain_' + args.prob_method + '.csv')

            df_test = pd.DataFrame({'score_RMIA_test': score_RMIA_test})
            df_test.to_csv(outdir + additional_name + str(args.epoch) + '_rmia_online_test_' + args.prob_method + '.csv')

            min_length = np.min([test_len, forget_len, retain_len])

            for seed_sub in range(3):
                seed_val = seed_sub
                torch.manual_seed(seed_val)
                torch.cuda.manual_seed_all(seed_val)
                np.random.seed(seed_val)
                random.seed(seed_val)

                if test_len > min_length:
                    samples_test_idx = np.random.choice(test_len, min_length, replace=False)
                else:
                    samples_test_idx = np.arange(test_len)
                # samples_test_idx = samples_test_idx + forget_len + retain_len

                if forget_len > min_length:
                    # samples_forget_idx = np.random.choice(unlearn_idx, min_length, replace=False)
                    samples_forget_idx = np.random.choice(forget_len, min_length, replace=False)
                else:
                    samples_forget_idx = np.arange(forget_len)
                
                if retain_len > min_length:
                    samples_remain_idx = np.random.choice(retain_len, min_length, replace=False)
                else:
                    samples_remain_idx = np.arange(retain_len)

                # samples_test = torch.utils.data.Subset(testset, samples_test_idx)
                samples_test_rmia = score_RMIA_test[samples_test_idx]

                # samples_forget = torch.utils.data.Subset(forgetset, samples_forget_idx)
                samples_forget_rmia = score_RMIA_forget[samples_forget_idx]

                # samples_remain = torch.utils.data.Subset(trainset_filtered, samples_remain_idx)
                samples_remain_rmia = score_RMIA_remain[samples_remain_idx]


                # compute the area under the curve score for each pair of samples above:
                # forget vs test
                forget_and_test = np.concatenate([samples_forget_rmia, samples_test_rmia])
                forget_test_score_list.append(forget_and_test)
                forget_and_test_labels = np.concatenate([np.ones(min_length), np.zeros(min_length)])
                roc_auc_score_ft = roc_auc_score(forget_and_test_labels, forget_and_test)
                ft_fpr, ft_tpr, ft_thresholds = metrics.roc_curve(forget_and_test_labels, forget_and_test)
                roc_forget_test_list.append(roc_auc_score_ft)

                # find tpr at fpr = 0.1%:
                fpr_01_idx = np.where(ft_fpr <= 0.001)[0][-1]
                ft_tpr_01 = ft_tpr[fpr_01_idx]
                tpr_01_forget_test_list.append(ft_tpr_01)
                print('tpr at fpr = 0.1%: ', ft_tpr_01)

                # ft_fpr_dict[seed_sub] = list(ft_fpr)
                # ft_tpr_dict[seed_sub] = list(ft_tpr)
                # ft_thresholds_dict[seed_sub] = list(ft_thresholds)

                # forget vs remain
                forget_and_remain = np.concatenate([samples_forget_rmia, samples_remain_rmia])
                forget_remain_score_list.append(forget_and_remain)
                forget_and_remain_labels = np.concatenate([np.zeros(min_length), np.ones(min_length)])
                roc_auc_score_fr = roc_auc_score(forget_and_remain_labels, forget_and_remain)
                fr_fpr, fr_tpr, fr_thresholds = metrics.roc_curve(forget_and_remain_labels, forget_and_remain)
                roc_forget_remain_list.append(roc_auc_score_fr)

                # find tpr at fpr = 0.1%:
                fpr_01_idx = np.where(fr_fpr <= 0.001)[0][-1]
                fr_tpr_01 = fr_tpr[fpr_01_idx]
                tpr_01_forget_remain_list.append(fr_tpr_01)
                print('tpr at fpr = 0.1%: ', fr_tpr_01)

                # fr_fpr_dict[seed_sub] = list(fr_fpr)
                # fr_tpr_dict[seed_sub] = list(fr_tpr)
                # fr_thresholds_dict[seed_sub] = list(fr_thresholds)

                # remain vs test
                remain_and_test = np.concatenate([samples_remain_rmia, samples_test_rmia])
                remain_test_score_list.append(remain_and_test)
                remain_and_test_labels = np.concatenate([np.ones(min_length), np.zeros(min_length)])
                roc_auc_score_rt = roc_auc_score(remain_and_test_labels, remain_and_test)
                rt_fpr, rt_tpr, rt_thresholds = metrics.roc_curve(remain_and_test_labels, remain_and_test)
                roc_remain_test_list.append(roc_auc_score_rt)

                # find tpr at fpr = 0.1%:
                fpr_01_idx = np.where(rt_fpr <= 0.001)[0][-1]
                rt_tpr_01 = rt_tpr[fpr_01_idx]
                tpr_01_remain_test_list.append(rt_tpr_01)
                print('tpr at fpr = 0.1%: ', rt_tpr_01)

                # rt_fpr_dict[seed_sub] = list(rt_fpr)
                # rt_tpr_dict[seed_sub] = list(rt_tpr)
                # rt_thresholds_dict[seed_sub] = list(rt_thresholds)

                test_acc_list.append(test_acc)
                forget_acc_list.append(forget_acc)
                remain_acc_list.append(remain_acc)
                adv_acc_list.append(adv_acc)
                seed_list.append(seed_sub)

            
            avg_remain_test_scores = np.array(remain_test_score_list).mean(axis=0)
            avg_forget_test_scores = np.array(forget_test_score_list).mean(axis=0)
            avg_forget_remain_scores = np.array(forget_remain_score_list).mean(axis=0)

            rt_fpr, rt_tpr, _ = metrics.roc_curve(remain_and_test_labels, avg_remain_test_scores)
            ft_fpr, ft_tpr, _ = metrics.roc_curve(forget_and_test_labels, avg_forget_test_scores)
            fr_fpr, fr_tpr, _ = metrics.roc_curve(forget_and_remain_labels, avg_forget_remain_scores)

            avg_rt_dict['fpr'] = list(rt_fpr)
            avg_rt_dict['tpr'] = list(rt_tpr)

            avg_ft_dict['fpr'] = list(ft_fpr) 
            avg_ft_dict['tpr'] = list(ft_tpr)

            avg_fr_dict['fpr'] = list(fr_fpr)
            avg_fr_dict['tpr'] = list(fr_tpr)


        df = pd.DataFrame({
            'seed': seed_list,
            'test_acc': test_acc_list,
            'forget_acc': forget_acc_list,
            'remain_acc': remain_acc_list,
            'adv_acc': adv_acc_list,
            'auc_ft': roc_forget_test_list,
            'auc_fr': roc_forget_remain_list,
            'auc_rt': roc_remain_test_list,
            'tpr_01_ft': tpr_01_forget_test_list,
            'tpr_01_fr': tpr_01_forget_remain_list,
            'tpr_01_rt': tpr_01_remain_test_list
        })

        avg_ft_df = pd.DataFrame(avg_ft_dict)
        avg_fr_df = pd.DataFrame(avg_fr_dict)
        avg_rt_df = pd.DataFrame(avg_rt_dict)

        initial_path_incmat = args.inclusion_mat.split('/')[-1].split('.')[0] + '/'
        initial_path_refmat = args.reference_mat.split('/')[-1].split('.')[0].split('__')[1] + '/'

        outdir = outdir + initial_path_incmat + initial_path_refmat

        additional_name = 'LRs_' + str(step_size) + '_lr_' + str(args.lr) + '_'

        if args.unlearn_count > 0:
            outdir += 'LRs_' + str(step_size) + '_lr_' + str(args.lr) + '/'
            if args.per_1k:
                additional_name += f'req_{args.req_index}_1k_'
            else:
                additional_name += f'req_{args.req_index}_'

        print('saving to: ', outdir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        df.to_csv(outdir + additional_name + str(args.epoch) + '_acc_' + args.mia_method + '_' +  args.prob_method + '_exc' + str(args.exclusive_flag) + '.csv')
        print(df)

        avg_ft_df.to_csv(outdir + additional_name + str(args.epoch) + '_avg_ft_roc_' + args.mia_method + '_' +  args.prob_method + '_exc' + str(args.exclusive_flag) + '.csv')
        avg_fr_df.to_csv(outdir + additional_name + str(args.epoch) + '_avg_fr_roc_' + args.mia_method + '_' +  args.prob_method + '_exc' + str(args.exclusive_flag) + '.csv')
        avg_rt_df.to_csv(outdir + additional_name + str(args.epoch) + '_avg_rt_roc_' + args.mia_method + '_' +  args.prob_method + '_exc' + str(args.exclusive_flag) + '.csv')

        # with open(outdir + additional_name + str(args.epoch) + '_ft_fpr_' + args.mia_method + '_' +  args.prob_method + '_exc' + str(args.exclusive_flag) + '.json', 'w') as f:
        #     json.dump(ft_fpr_dict, f)

        # with open(outdir + additional_name + str(args.epoch) + '_ft_tpr_' + args.mia_method + '_' +  args.prob_method + '_exc' + str(args.exclusive_flag) + '.json', 'w') as f:
        #     json.dump(ft_tpr_dict, f)

        # with open(outdir + additional_name + str(args.epoch) + '_ft_thresholds_' + args.mia_method + '_' +  args.prob_method + '_exc' + str(args.exclusive_flag) + '.json', 'w') as f:
        #     json.dump(ft_thresholds_dict, f)


        # with open(outdir + additional_name + str(args.epoch) + '_fr_fpr_' + args.mia_method + '_' +  args.prob_method + '_exc' + str(args.exclusive_flag) + '.json', 'w') as f:
        #     json.dump(fr_fpr_dict, f)

        # with open(outdir + additional_name + str(args.epoch) + '_fr_tpr_' + args.mia_method + '_' +  args.prob_method + '_exc' + str(args.exclusive_flag) + '.json', 'w') as f:
        #     json.dump(fr_tpr_dict, f)

        # with open(outdir + additional_name + str(args.epoch) + '_fr_thresholds_' + args.mia_method + '_' +  args.prob_method + '_exc' + str(args.exclusive_flag) + '.json', 'w') as f:
        #     json.dump(fr_thresholds_dict, f)


        # with open(outdir + additional_name + str(args.epoch) + '_rt_fpr_' + args.mia_method + '_' +  args.prob_method + '_exc' + str(args.exclusive_flag) + '.json', 'w') as f:
        #     json.dump(rt_fpr_dict, f)

        # with open(outdir + additional_name + str(args.epoch) + '_rt_tpr_' + args.mia_method + '_' +  args.prob_method + '_exc' + str(args.exclusive_flag) + '.json', 'w') as f:
        #     json.dump(rt_tpr_dict, f)

        # with open(outdir + additional_name + str(args.epoch) + '_rt_thresholds_' + args.mia_method + '_' +  args.prob_method + '_exc' + str(args.exclusive_flag) + '.json', 'w') as f:
        #     json.dump(rt_thresholds_dict, f)


        df_avg = df.mean(axis=0)
        df_avg_final = df_avg[['forget_acc', 'remain_acc', 'test_acc', 'auc_ft', 'auc_fr', 'auc_rt', 'tpr_01_ft', 'tpr_01_fr', 'tpr_01_rt']]
        # df_avg_final['avg'] = df_avg_final.mean(axis=0)
        df_avg_final['adv_acc'] = df_avg['adv_acc']
        df_avg_final.to_csv(outdir + additional_name + str(args.epoch) + '_avg_acc_' + args.mia_method + '_' +  args.prob_method + '_exc' + str(args.exclusive_flag) + '.csv')
        print('saving to: ', outdir + additional_name + str(args.epoch) + '_avg_acc_' + args.mia_method + '_' +  args.prob_method + '_exc' + str(args.exclusive_flag) + '.csv')

        rmia_scores = np.array(rmia_scores)
        rmia_scores = rmia_scores.mean(axis=0)
        df = pd.DataFrame({'score_RMIA_all': rmia_scores})
        df.to_csv(outdir + additional_name + str(args.epoch) + '_' + args.mia_method + '_online_scores_avg_' + args.prob_method + '_exc' + str(args.exclusive_flag) + '.csv')
        print('saving to : ', outdir + additional_name + str(args.epoch) + '_' + args.mia_method + '_online_scores_avg_' + args.prob_method + '_exc' + str(args.exclusive_flag) + '.csv')

