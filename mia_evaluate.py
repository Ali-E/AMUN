import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.utils
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import evaluation
from models import *
from models.resnet_orig import ResNet18_orig
import pandas as pd
import random
from torch.utils.data import Dataset

from helper import CustomImageDataset


# class CustomImageDataset(Dataset):
#     def __init__(self, labels_file, imgs_path, unlearn_indices=None, transform=None, target_transform=None):
#         self.imgs_path = imgs_path
#         self.images_delta_df = pd.read_csv(labels_file)
#         self.img_labels = self.images_delta_df['adv_pred'].values[unlearn_indices]
#         self.img_deltas = self.images_delta_df['delta_norm'].values[unlearn_indices]
#         self.transform = transform # feature transformation
#         self.target_transform = target_transform # label transformation
#         self.adv_images = torch.load(self.imgs_path, map_location=torch.device('cpu'))
#         self.adv_images = self.adv_images[unlearn_indices]
#         self.adv_images = self.adv_images.detach().numpy()
# 
#     def __len__(self):
#         return len(self.img_labels)
# 
#     def __getitem__(self, idx):
#         image = self.adv_images[idx]
#         image = image.transpose(1, 2, 0)
#         label = self.img_labels[idx]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label


os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', default='cifar', help='dataset')
parser.add_argument('--model', default='ResNet18', help='Deep Learning model to train')
parser.add_argument('--method', default='orig', help='clipping method (use orig for no clipping)')
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

parser.add_argument('--catsn', default=-1, type=float)
parser.add_argument('--convsn', default=1., type=float)
parser.add_argument('--outer_steps', default=100, type=int)
parser.add_argument('--convsteps', default=100, type=int)
parser.add_argument('--opt_iter', default=5, type=int)
parser.add_argument('--outer_iters', default=1, type=int)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('==========', device)

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


def test(loader, criterion):
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

    test_acc_list = []
    forget_acc_list = []
    remain_acc_list = []
    adv_acc_list = []
    avg_diff_list = []

    correctness_list = []
    confidence_list = []
    entropy_list = []
    m_entropy_list = []
    prob_list = []
    seed_list = []

    seed_in = args.seed
    if seed_in == -2:
        seed_in = [1,2]
    else:
        seed_in = [seed_in]
    for seed in seed_in:
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

            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            trainset = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10( root='./data', train=False, download=True, transform=transform)

            advset = CustomImageDataset(labels_file=args.adv_delta, imgs_path=args.adv_images,unlearn_indices=unlearn_idx, transform=transform, target_transform=None)

        else:
            print('only cifar is supported for basic MIA, use RMIA instead!')
            exit(0)

        unlearn_idx = pd.read_csv(args.unlearn_indices)['unlearn_idx'].values

        print('==> Building model..')
        print('-----------------------------------------------------------------')
        print('initial len of trainset: ', len(trainset))  

        ### remove the unlearned images from the trainset
        trainset_filtered = torch.utils.data.Subset(trainset, list(set(range(len(trainset))) - set(unlearn_idx)))
        print('len of filtered trainset: ', len(trainset_filtered))  

        forgetset = torch.utils.data.Subset(trainset, unlearn_idx)
        print('len of forget set: ', len(forgetset))  
        print('final len of trainset: ', len(trainset))  
        print('-----------------------------------------------------------------')

        trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=128, num_workers=1)
        testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=128, num_workers=1)
        forgetloader = torch.utils.data.DataLoader(forgetset, shuffle=False, batch_size=128, num_workers=1)
        advloader = torch.utils.data.DataLoader(advset, shuffle=False, batch_size=128, num_workers=1)
        remainloader = torch.utils.data.DataLoader(trainset_filtered, shuffle=False, batch_size=128, num_workers=1)

        outdir = args.source_model_path
        # if args.mask_path is not None and args.mask_path != 'None':
        #     outdir = outdir + '_mask_' + str(args.mask_path).split('with_')[1][:-3] + '_'

        print('------------> outdir: ', outdir)
        print('------------> epoch: ', args.epoch)

        trial_seeds = [10**i for i in range(3)][:args.trials]
        for trial in trial_seeds:
            if args.model == 'ResNet18':
                if orig_flag:
                    net = ResNet18_orig(in_chan=in_chan, bn=bn_flag, device=device, elu_flag=False)
                elif clip_flag:
                    net = ResNet18(concat_sv=concat_sv, in_chan=in_chan, device=device, clip=args.convsn, clip_concat=args.catsn, clip_flag=True, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_steps=clip_steps, bn_count=steps_count, clip_outer=clip_outer_flag, clip_opt_iter=opt_iter, summary=True, writer=None, save_info=False, outer_iters=outer_iters, outer_steps=outer_steps)

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
            if 'retrain' in checkpoint_path:
                checkpoint_path += '_ckpt.' + str(args.epoch)
            elif 'unlearn' not in checkpoint_path:
                checkpoint_path += 'checkpoint.pth.tar_' + str(args.epoch)
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
                    net.load_state_dict(checkpoint['net'])
            print('model loaded')

            net.eval()
            test_loss, test_acc = test(testloader, criterion)
            forget_loss, forget_acc = test(forgetloader, criterion)
            remain_loss, remain_acc = test(remainloader, criterion)
            adv_loss, adv_acc = test(advloader, criterion)
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

            random_indices = np.random.choice(list(range(retain_len)), size=test_len, replace=False)
            print('random_indices: ', random_indices[:10])
            shadow_train = torch.utils.data.Subset(trainset_filtered, random_indices)
            shadow_train_loader = torch.utils.data.DataLoader(
                shadow_train, batch_size=256, shuffle=False
            )

            eval_results = evaluation.SVC_MIA(
                shadow_train=shadow_train_loader,
                shadow_test=testloader,
                target_train=None,
                target_test=forgetloader,
                model=net,
            )


            retrain_res_5000 = {'UA': 94.49, 'RA': 100.0, 'TA': 94.33, 'MIA': 12.53}
            retrain_res_25000 = {'UA': 92.09, 'RA': 100.0, 'TA': 91.85, 'MIA': 16.78}

            retrain_res = {'UA': 0, 'RA': 0, 'TA': 0, 'MIA': 0}
            if unlearn_idx.shape[0] == 5000:
                retrain_res = retrain_res_5000
            elif unlearn_idx.shape[0] == 25000:
                retrain_res = retrain_res_25000

            print('retrain res: ', retrain_res)
            test_acc_diff = np.abs(test_acc - retrain_res['TA'])
            print('test acc diff: ', test_acc_diff)
            forget_acc_diff = np.abs(forget_acc - retrain_res['UA'])
            print('forget acc diff: ', forget_acc_diff)
            remain_acc_diff = np.abs(remain_acc - retrain_res['RA'])
            print('remain acc diff: ', remain_acc_diff)
            confidence_diff = np.abs(100*eval_results["confidence"] - retrain_res['MIA'])
            print('confidence diff: ', confidence_diff)
            print('confidence: ', 100*eval_results['confidence'])

            print('-------------->>>>>>>>', {'UA': forget_acc, "RA": remain_acc, "TA": test_acc, "MIA": 100*eval_results["confidence"]})

            avg_diff = (test_acc_diff + forget_acc_diff + remain_acc_diff + confidence_diff) / 4.
            avg_diff_list.append(avg_diff)
            test_acc_list.append(test_acc)
            forget_acc_list.append(forget_acc)
            remain_acc_list.append(remain_acc)
            adv_acc_list.append(adv_acc)

            correctness_list.append(eval_results["correctness"])
            confidence_list.append(100. * eval_results["confidence"])
            entropy_list.append(eval_results["entropy"])
            m_entropy_list.append(eval_results["m_entropy"])
            prob_list.append(eval_results["prob"])
            seed_list.append(seed)

        df = pd.DataFrame({
            'seed': seed_list,
            'test_acc': test_acc_list,
            'forget_acc': forget_acc_list,
            'remain_acc': remain_acc_list,
            'adv_acc': adv_acc_list,
            'avg_diff': avg_diff_list,

            'correctness': correctness_list,
            'confidence': confidence_list,
            'entropy': entropy_list,
            'm_entropy': m_entropy_list,
            'prob': prob_list
        })

        additional_name = 'LRs_' + str(step_size) + '_lr_' + str(args.lr) + '_'
        if 'retrain' not in args.source_model_path:
            df.to_csv(outdir + additional_name + str(args.epoch) + '_mia_SVC_results.csv')
        else:
            df.to_csv(outdir + str(args.epoch) + '_mia_SVC_results.csv')

        df_avg = df.mean(axis=0)
        df_avg_final = df_avg[['forget_acc', 'remain_acc', 'test_acc', 'confidence', 'avg_diff']]
        df_avg_final['adv_acc'] = df_avg['adv_acc']
        df_avg_final.to_csv(outdir + additional_name + str(args.epoch) + '_avg_mia_SVC_results.csv')

