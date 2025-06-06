import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.utils
import torchvision
import torchvision.transforms as transforms
from datasets import get_dataset
import os
import argparse
import evaluation
from models import *
from models.resnet_orig import ResNet18_orig
import pandas as pd
import random
from torch.utils.data import Dataset


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

####### RMIA parameters:
parser.add_argument('--gamma', default=0.1, type=float, help='threshold value for RMIA')
parser.add_argument('--a_factor', default=0.4, type=float, help='factor a for inline likelihood evaluation')
parser.add_argument('--use_all_ref', default=True, type=bool)
parser.add_argument('--prob_method', default='logits', type=str)
parser.add_argument('--inclusion_mat', default=None, type=str)
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

    test_acc_list = []
    forget_acc_list = []
    remain_acc_list = []
    adv_acc_list = []

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

            advset = CustomImageDataset(labels_file=args.adv_delta, imgs_path=args.adv_images, ratio=120., unlearn_indices=unlearn_idx, transform=transform_adv, target_transform=None)#, device=device)

        else:
            print('mnist!')
            in_chan = 1
            trainset = get_dataset('mnist', 'train')
            testset = get_dataset('mnist', 'test')

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


        if args.use_all_ref:
            print('using concatenated trainset and testset')
            trainset = torch.utils.data.ConcatDataset([trainset, testset]) 
        else:
            trainset = torch.utils.data.ConcatDataset([trainset_filtered, testset])

        # if args.inclusion_mat is not None:
        inclusion_mat = pd.read_csv(args.inclusion_mat).values
        print('inclusion_mat shape: ', inclusion_mat.shape)
        
        trainloader = torch.utils.data.DataLoader(trainset, shuffle=False, batch_size=128, num_workers=1) ### used by reference models

        testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=128, num_workers=1)
        forgetloader = torch.utils.data.DataLoader(forgetset, shuffle=False, batch_size=128, num_workers=1)
        forgetloader_single = torch.utils.data.DataLoader(forgetset, shuffle=False, batch_size=1, num_workers=1)
        advloader = torch.utils.data.DataLoader(advset, shuffle=False, batch_size=128, num_workers=1)
        remainloader = torch.utils.data.DataLoader(trainset_filtered, shuffle=False, batch_size=128, num_workers=1)

        outdir = args.source_model_path
        # if args.mask_path is not None and args.mask_path != 'None':
        #     outdir = outdir + '_mask_' + str(args.mask_path).split('with_')[1][:-3] + '_'

        print('------------> outdir: ', outdir)
        print('------------> epoch: ', args.epoch)

        if args.trials == -32:
            trial_seeds = [i for i in range(32)]
            all_prob_mat = np.zeros((32, len(trainset)))
        elif args.trials == -64:
            trial_seeds = [i for i in range(64)]
            all_prob_mat = np.zeros((64, len(trainset)))
        elif args.trials == -128:
            trial_seeds = [i for i in range(128)]
            all_prob_mat = np.zeros((128, len(trainset)))
            if args.one_hot:
                all_prob_mat = np.zeros((128, len(trainset), 10))
        elif args.trials == -4:
            trial_seeds = [i for i in range(4)]
            all_prob_mat = np.zeros((4, len(trainset)))
            if args.one_hot:
                all_prob_mat = np.zeros((4, len(trainset), 10))
        else:
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
            checkpoint_path = args.source_model_path + str(trial) + '/' 
            # if args.mask_path is not None and args.mask_path != 'None':
            #     checkpoint_path = checkpoint_path + 'mask_' + str(args.mask_path).split('with_')[1][:-3] + '/'

            checkpoint_path += '_ckpt.' + str(args.epoch)

            print('model path: ', checkpoint_path)

            checkpoint = torch.load(checkpoint_path)
            print(checkpoint.keys())
            if clip_flag:
                net.load_state_dict(checkpoint['net'], strict=False)
            else:
                net.load_state_dict(checkpoint['net'])#, strict=False)
            print('model loaded')
            net.eval()

            included_indices = inclusion_mat[trial]
            print(included_indices.shape)
            included_indices = np.arange(len(included_indices))[included_indices]
            print(included_indices.shape)
            included_set = torch.utils.data.Subset(trainset, included_indices)
            included_loader = torch.utils.data.DataLoader(included_set, shuffle=False, batch_size=128, num_workers=1)

            nonincluded_indices = list(set(range(len(trainset))) - set(included_indices))
            nonincluded_set = torch.utils.data.Subset(trainset, nonincluded_indices)
            nonincluded_loader = torch.utils.data.DataLoader(nonincluded_set, shuffle=False, batch_size=128, num_workers=1)


            eval_results = evaluation.RMIA(
                model=net,
                remain_loader=trainloader,
                forget_loader=included_loader,
                test_loader=nonincluded_loader,
                device=device,
                one_hot=args.one_hot,
                logits_out=True
                # prob_method=args.prob_method
            )
            all_remain_likelihood = eval_results["remain_likelihood"]
            print(all_remain_likelihood.shape)

            all_prob_mat[trial] = all_remain_likelihood.cpu().numpy()

            non_included_loss, non_included_acc = test(net, nonincluded_loader, criterion)
            included_loss, included_acc = test(net, included_loader, criterion)
            remain_loss, remain_acc = test(net, trainloader, criterion)
            adv_loss, adv_acc = test(net, advloader, criterion)
            print('nonincluded acc: ', non_included_acc)
            print('included acc: ', included_acc)
            print('all acc: ', remain_acc)
            if remain_acc < 50:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>, all acc: ', remain_acc)
            print('adv acc: ', adv_acc)

            print('test_len: ', len(nonincluded_set))
            print('forget_len: ', len(included_set))
            print('retain_len: ', len(trainset))

            test_acc_list.append(non_included_acc)
            forget_acc_list.append(included_acc)
            remain_acc_list.append(remain_acc)
            adv_acc_list.append(adv_acc)
            seed_list.append(seed)


        df = pd.DataFrame({
            'seed': seed_list,
            'non_included_acc': test_acc_list,
            'included_acc': forget_acc_list,
            'all_acc': remain_acc_list,
            'adv_acc': adv_acc_list,

        })

        initial_path_incmat = args.inclusion_mat.split('/')[-1].split('.')[0][5:] + '_'

        df.to_csv(outdir + initial_path_incmat + str(args.epoch) + '_acc_results.csv')

        df_avg = df.mean(axis=0)
        df_avg_final = df_avg[['included_acc', 'non_included_acc', 'all_acc']]
        # df_avg_final['avg'] = df_avg_final.mean(axis=0)
        df_avg_final['adv_acc'] = df_avg['adv_acc']
        df_avg_final.to_csv(outdir + initial_path_incmat +  str(args.epoch) + '_avg_acc_results.csv')


        # convert the prob matrix to a dataframe:
        if args.one_hot:
            torch.save(torch.tensor(all_prob_mat).float(), outdir + initial_path_incmat + str(args.epoch) + '_prob_matrix_' + args.prob_method + '_onehot.pt')
        else:
            prob_df = pd.DataFrame(all_prob_mat.T, columns=['seed_'+str(i) for i in trial_seeds])
            prob_df.to_csv(outdir + initial_path_incmat + str(args.epoch) + '_prob_matrix_' + args.prob_method + '.csv')

