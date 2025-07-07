import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from datasetslocal import get_dataset
from datasets import load_dataset
import argparse
import evaluation
from models import *
from models.resnet_orig import ResNet18_orig
import pandas as pd
from models.vgg import VGG
import random
from helper import basicDataset


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', default='cifar', help='dataset')
parser.add_argument('--model', default='ResNet18', help='Deep Learning model to train')
parser.add_argument('--method', default='catclip', help='clipping method (use orig for no clipping)')
parser.add_argument('--mode', default='wBN', help='what to do with BN layers (leave empty for keeping it as it is)')
parser.add_argument('--epoch', default=200, type=int, help='epoch to use for evaluation')
parser.add_argument('--seed', default=1, type=int, help='seed value') # this seed corresponds to the different runs of the MIA evaluation on the same unlearned model
parser.add_argument('--steps', default=50, type=int, help='setp count for clipping BN')
parser.add_argument('--trials', default=1, type=int, help='traial value') # each trial corresponds to a different run of the unlearning method 
# on a specific trained model. if the unlearning method does not involve randomness, then the trial value should be set to 1.

parser.add_argument('--unlearn_indices', default=None, type=str)
parser.add_argument('--source_model_path', default=None, type=str)
parser.add_argument('--num_classes', default=10, type=int, help='number of classes in the dataset')

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

if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    print('chosen: ', device)
    cudnn.benchmark = True

print('model: ', args.model)
print('dataset: ', args.dataset)


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

    test_acc_list = []
    forget_acc_list = []
    remain_acc_list = []
    correctness_list = []
    confidence_list = []
    entropy_list = []
    m_entropy_list = []
    prob_list = []
    seed_list = []

    seed_in = [1]
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

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

            trainset = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, transform=transform_test)
            testset = torchvision.datasets.CIFAR10( root='./data', train=False, download=True, transform=transform_test)

        elif args.dataset == 'tinynet':
            print('Tine ImageNet!')
            in_chan = 3
            tinynet_flag = True
            args.num_classes = 200

            transform_test = transforms.Compose([
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

        else:
            print('mnist!')
            in_chan = 1
            trainset = get_dataset('mnist', 'train')
            testset = get_dataset('mnist', 'test')

        print('==> Building model..')
        print('-----------------------------------------------------------------')
        print('initial len of trainset: ', len(trainset))  
        print('final len of trainset: ', len(trainset))  
        print('-----------------------------------------------------------------')


        print('using concatenated trainset and testset')
        trainset = torch.utils.data.ConcatDataset([trainset, testset]) 

        # if args.inclusion_mat is not None:
        inclusion_mat = pd.read_csv(args.inclusion_mat).values
        print('inclusion_mat shape: ', inclusion_mat.shape)
        
        trainloader = torch.utils.data.DataLoader(trainset, shuffle=False, batch_size=128, num_workers=1) ### used by reference models
        testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=128, num_workers=1)

        outdir = args.source_model_path

        print('------------> outdir: ', outdir)
        print('------------> epoch: ', args.epoch)

        trial_seeds = -args.trials if args.trials < 0 else args.trials
        trial_seeds = [int(i) for i in range(trial_seeds)]
        all_prob_mat = np.zeros((len(trial_seeds), len(trainset)))
        if args.one_hot:
            all_prob_mat = np.zeros((len(trial_seeds), len(trainset), args.num_classes))

        for trial in trial_seeds:
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

            checkpoint_path = args.source_model_path + str(trial) + '/' 
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
            included_indices = [int(i) for i in included_indices]
            included_set = torch.utils.data.Subset(trainset, included_indices)
            included_loader = torch.utils.data.DataLoader(included_set, shuffle=False, batch_size=128, num_workers=1)

            nonincluded_indices = list(set(range(len(trainset))) - set(included_indices))
            nonincluded_set = torch.utils.data.Subset(trainset, nonincluded_indices)
            nonincluded_loader = torch.utils.data.DataLoader(nonincluded_set, shuffle=False, batch_size=128, num_workers=1)

            eval_results = evaluation.RMIA_old(
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
            print('nonincluded acc: ', non_included_acc)
            print('included acc: ', included_acc)
            print('all acc: ', remain_acc)
            if remain_acc < 50:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>, all acc: ', remain_acc)

            print('test_len: ', len(nonincluded_set))
            print('forget_len: ', len(included_set))
            print('retain_len: ', len(trainset))

            test_acc_list.append(non_included_acc)
            forget_acc_list.append(included_acc)
            remain_acc_list.append(remain_acc)
            seed_list.append(seed)

        df = pd.DataFrame({
            'seed': seed_list,
            'non_included_acc': test_acc_list,
            'included_acc': forget_acc_list,
            'all_acc': remain_acc_list,
        })

        initial_path_incmat = args.inclusion_mat.split('/')[-1].split('.')[0][5:] + '_'

        df.to_csv(outdir + initial_path_incmat + str(args.epoch) + '_acc_results.csv')

        df_avg = df.mean(axis=0)
        df_avg_final = df_avg[['included_acc', 'non_included_acc', 'all_acc']]
        df_avg_final.to_csv(outdir + initial_path_incmat +  str(args.epoch) + '_avg_acc_results.csv')

        # convert the prob matrix to a dataframe:
        if args.one_hot:
            torch.save(torch.tensor(all_prob_mat).float(), outdir + initial_path_incmat + str(args.epoch) + '_prob_matrix_' + args.prob_method + '_onehot.pt')
        else:
            prob_df = pd.DataFrame(all_prob_mat.T, columns=['seed_'+str(i) for i in trial_seeds])
            prob_df.to_csv(outdir + initial_path_incmat + str(args.epoch) + '_prob_matrix_' + args.prob_method + '.csv')
