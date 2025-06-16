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


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def discretize(x):
    return torch.round(x * 255) / 255


def FGSM_perturb(x, y, model=None, bound=None, criterion=None):
    device = model.parameters().__next__().device
    model.zero_grad()
    x_adv = x.detach().clone().requires_grad_(True).to(device)
    pred = model(x_adv)
    loss = criterion(pred, y)
    loss.backward()
    grad_sign = x_adv.grad.data.detach().sign()
    x_adv = x_adv + grad_sign * bound
    x_adv = discretize(torch.clamp(x_adv, 0.0, 1.0))
    return x_adv.detach()


def expand_model(model):
    last_fc_name = None
    last_fc_layer = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_fc_name = name
            last_fc_layer = module

    if last_fc_name is None:
        raise ValueError("No Linear layer found in the model.")

    num_classes = last_fc_layer.out_features

    bias = last_fc_layer.bias is not None

    new_last_fc_layer = nn.Linear(
        in_features=last_fc_layer.in_features,
        out_features=num_classes + 1,
        bias=bias,
        device=last_fc_layer.weight.device,
        dtype=last_fc_layer.weight.dtype,
    )

    with torch.no_grad():
        new_last_fc_layer.weight[:-1] = last_fc_layer.weight
        if bias:
            new_last_fc_layer.bias[:-1] = last_fc_layer.bias

    parts = last_fc_name.split(".")
    current_module = model
    for part in parts[:-1]:
        current_module = getattr(current_module, part)
    setattr(current_module, parts[-1], new_last_fc_layer)


def imshow(img, path=None):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()
    plt.savefig(path)


class simpleDataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

        self.data = self.data.detach().cpu().numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = image.transpose(1, 2, 0)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class CustomImageDataset(Dataset):
    def __init__(self, labels_file, imgs_path, unlearn_indices=None, transform=None, target_transform=None, num_classes=10, ablation_RL=False, ablation_RS=False, ablation_forgetset_RL=False, ablation_forgetset_AdvL=False, image_orig=None):#, device='cuda'):
        self.imgs_path = imgs_path
        self.num_classes = num_classes
        self.ablation_RL = ablation_RL
        self.ablation_forgetset_RL = ablation_forgetset_RL
        self.images_delta_df = pd.read_csv(labels_file)
        self.img_labels = self.images_delta_df['adv_pred'].values[unlearn_indices]
        self.img_deltas = self.images_delta_df['delta_norm'].values[unlearn_indices]
        self.transform = transform # feature transformation
        self.target_transform = target_transform # label transformation
        if not ablation_RS and not ablation_forgetset_RL and not ablation_forgetset_AdvL:
            self.adv_images = torch.load(self.imgs_path, map_location=torch.device('cpu'))
            self.adv_images = self.adv_images[unlearn_indices]
        else:
            self.adv_images = image_orig

        if ablation_RL or ablation_forgetset_RL:
            self.true_labels = self.images_delta_df['label'].values[unlearn_indices]

        if ablation_RS:
            ## choose adv images to be img_delta radius away from adv_images. In this case
            ## the adv_images images are the same as the original images not adv images.
            rand_vectors = torch.randn_like(self.adv_images)
            rand_vectors_norm = torch.norm(rand_vectors, p=2, dim=(1,2,3), keepdim=False)
            unit_vectors = rand_vectors / rand_vectors_norm.view(-1, 1, 1, 1)
            scaled_unit_vectors = unit_vectors * torch.tensor(self.img_deltas).view(-1, 1, 1, 1)
            check_norms = torch.norm(scaled_unit_vectors, p=2, dim=(1,2,3), keepdim=False)
            self.adv_images = self.adv_images + scaled_unit_vectors
            self.adv_images = torch.clamp(self.adv_images, min=0, max=1)

        self.adv_images = self.adv_images.detach().numpy()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.adv_images[idx]
        image = image.transpose(1, 2, 0)
        label = self.img_labels[idx]
        if self.ablation_RL or self.ablation_forgetset_RL:
            ## choose a random label other than the true label and adv label:
            label = np.random.choice([i for i in range(self.num_classes) if i != self.true_labels[idx] and i != self.img_labels[idx]]) 
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class RLDataset(Dataset):
    def __init__(self, forgetset, new_classes=None, num_classes=10, noise_level=0.01, add_noise=False):
        self.image_set = forgetset
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.num_classes = num_classes
        self.new_classes = new_classes

    def __len__(self):
        return len(self.image_set)

    def __getitem__(self, idx):
        image = self.image_set[idx][0]
        if self.new_classes is not None:
            label = self.new_classes[idx]
        else:
            true_label = self.image_set[idx][1]
            label = np.random.choice([i for i in range(self.num_classes) if i != true_label]) # random label

        if self.add_noise:
            noise = torch.randn_like(image) * self.noise_level
            adv_images = self.adv_images + noise
            adv_images = torch.clamp(adv_images, min=0, max=1)

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
            if len(image.shape) == 2:
                image = copy.deepcopy(np.stack((image, image, image), axis=2))
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


unlearn_indices_check = pd.read_csv(args.unlearn_indices)['unlearn_idx'].values
# if args.unlearn_count > 0:
#     unlearn_indices_check = unlearn_indices_check[args.start_idx: args.start_idx + args.unlearn_count]
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
    # net = torch.nn.DataParallel(net)
    print('chosen: ', device)
    cudnn.benchmark = True

# if args.adv_images is None:
#     print('adv images not provided!')
#     exit(0)
# 
# if args.adv_delta is None:
#     print('adv delta not provided!')
#     exit(0)

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

            # model = copy.deepcopy(net).to(device)
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
        if len(unlearn_idx) == 5000:
            included_indices_file = 'keep_files/keep_m128_d55000_s0.csv'
        elif len(unlearn_idx) == 10000:
            included_indices_file = 'keep_files/keep_m128_d105000_s0.csv'
        elif len(unlearn_idx) == 25000:
            included_indices_file = 'keep_files/keep_m128_d35000_s0.csv'
        else:
            print('unknown unlearn_idx count!')
            exit(0)

        if args.use_all_ref:
            if args.dataset == 'cifar':
                included_indices_file = 'keep_files/keep_m128_d60000_s0.csv'
            elif args.dataset == 'tinynet':
                included_indices_file = 'keep_files/keep_m128_d110000_s0.csv'
            else:
                print('unknown dataset!')
                exit(0)


        included_indices_all = pd.read_csv(included_indices_file, header=0).values
        print('seed: ', args.seed, included_indices_all.shape)
        included_indices = included_indices_all[args.seed]

        trainset_combined = torch.utils.data.ConcatDataset([trainset, testset])
        if epoch == 0:
            print('row id:', args.seed)
            print('sum included: ', included_indices.sum())
            print('len of combined trainset: ', len(trainset_combined))  
        inc_indices = [int(i) for i in np.array(list(range(len(trainset_combined))))[included_indices]]
        trainset_included = torch.utils.data.Subset(trainset_combined, inc_indices)
        if epoch == 0:
            print('len of included trainset: ', len(trainset_included))  
        trainset_combined = trainset_included


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

    # net.eval() 
    # tr_loss, tr_acc = test(trainloader, epoch, criterion, writer, mode='train right before', model_path=None)
    # net.train()

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

        ###### plot images and print labels:
        if args.ablation_advset and ii < 10:
            # if args.dataset == 'cifar':
            #     names_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            # elif args.dataset == 'tinynet':
            #     names_labels = [str(i) for i in range(200)]
            # fig, axs = plt.subplots(ncols=10, figsize=(20, 3))
            # fig.suptitle('first batch of the trainloader')
            # for ii in range(10):
            axs[ii].imshow((inputs[ii]).permute(1, 2, 0).detach().cpu()); axs[ii].axis('off')
            axs[ii].set_title(f'Label: {names_labels[targets[ii]]}')
            ii += 1
            # fig.savefig(outdir + f'_{args.dataset}_1stBatch_e' + str(epoch) + '.png')
            # plt.close()
        #############################################
                

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


    # writer.add_scalar('train/acc', 100.*correct/total, epoch)
    # writer.add_scalar('train/loss', train_loss/(batch_idx+1), epoch)
    # writer.add_scalar('train/time', tot_time, epoch)

    print('train - acc', 100.*correct/total)
    print('train - loss', train_loss/(batch_idx+1))
    
    # net.eval() 
    # tr_loss, tr_acc = test(trainloader, epoch, criterion, writer, mode='train right after', model_path=None)
    # net.train()

    scheduler.step()

    print('Saving..')
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
        # 'scheduler': scheduler.state_dict(),
        # 'optimizer': optimizer.state_dict(),
    }

    # if not os.path.isdir('checkpoint'):
    #     os.mkdir('checkpoint')
    model_path_i = model_path + ".%d" % (epoch)
    if args.unlearn_method == 'retrain' or args.unlearn_method == 'reference' or args.ablation_advset:
        if epoch in [60, 80, 100,120,140,160,180,200]:
            torch.save(state, model_path_i)
    else:
        torch.save(state, model_path_i)

    # directory = model_path + '_sd_' + str(args.seed) + '_ep_' + str(epoch) + '.pth'
    # print(directory, save_checkpoints)
    # if epoch >=15 and save_checkpoints:
    #     torch.save(state, directory)

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

            ###### plot images and print labels:
            if plot_images and args.ablation_advset and ii < 10:
                # if args.dataset == 'cifar':
                #     names_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                # elif args.dataset == 'tinynet':
                #     names_labels = [str(i) for i in range(200)]

                predictions = torch.argmax(outputs, dim=1)
                # fig, axs = plt.subplots(ncols=10, figsize=(20, 3))
                # fig2, axs2 = plt.subplots(ncols=10, figsize=(20, 3))
                # fig.suptitle('first batch of the trainloader with correct labels')
                # fig2.suptitle('first batch of the trainloader with output labels')
                # for ii in range(10):
                axs[ii].imshow((inputs[ii]).permute(1, 2, 0).detach().cpu()); axs[ii].axis('off')
                axs[ii].set_title(f'Label: {names_labels[targets[ii]]}')

                axs2[ii].imshow((inputs[ii]).permute(1, 2, 0).detach().cpu()); axs2[ii].axis('off')
                axs2[ii].set_title(f'Label: {names_labels[predictions.int()[ii]]}')
                ii += 1

                # fig.savefig(outdir + f'{args.dataset}_test_correct_1stBatch_e' + str(epoch) + '.png')
                # fig2.savefig(outdir + f'{args.dataset}_test_out_1stBatch_e' + str(epoch) + '.png')
                # plt.close('all')
            #############################################

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
                # 'scheduler': scheduler.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'lr': scheduler.get_last_lr(),
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

    unlearn_idx = pd.read_csv(args.unlearn_indices)['unlearn_idx'].values
    unlearn_idx = [int(i) for i in unlearn_idx]
    # if args.unlearn_count > 0:
    #     unlearn_idx = unlearn_idx[args.start_idx : args.start_idx + args.unlearn_count]

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

            # train_loader = load_train_data(img_size, randaug_magnitude, batch_size)
            # val_loader = load_val_data(img_size, batch_size if not args.throughput else 32)

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
            # train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

            testset_all = load_dataset('Maysee/tiny-imagenet', split='valid')
            # test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=1)


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
        indices_count = len(unlearn_idx) # args.unlearn_indices.split('/')[-2]

        args.outdir = f"/{dataset_name}/unlearn/{args.unlearn_method}/{indices_count}/unl_idx_{indices_seed}/"
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

        # if args.unlearn_method == 'RL':
        #     args.outdir = f"/{dataset_name}/unlearn/{args.unlearn_method}_wrand_waug/{indices_count}/unl_idx_{indices_seed}/"

        if (args.unlearn_method == 'adv' or args.unlearn_method == 'amun') and args.req_mode == 'adaptive' and args.adaptive_lr:
            args.outdir = f"/{dataset_name}/unlearn/{args.unlearn_method}_{args.adaptive_lr_factor}/{indices_count}/unl_idx_{indices_seed}/"

        # args.outdir = f"/{dataset_name}/unlearn/{args.unlearn_method}_pgd10/{indices_count}/unl_idx_{indices_seed}/"
        
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

            removed_classes = [trainset[i][1] for i in unlearn_idx]
            df = pd.DataFrame({'unlearn_idx': unlearn_idx, 'removed_classes': removed_classes})
            df.to_csv(outdir + 'unlearn_idx.csv')

            ### remove the unlearned images from the trainset
            trainset_filtered = torch.utils.data.Subset(trainset, list(set(range(len(trainset))) - set(unlearn_idx) - set(prior_idx)))
            print('len of filtered trainset: ', len(trainset_filtered))  
            # print(trainset_filtered.report())


            # if args.dataset == 'tinynet':
            #     forgetset_all = torch.utils.data.Subset(trainset_all, unlearn_idx)
            #     forgetset = basicDataset(forgetset_all, transform=transform_train, target_transform=None)
            # else:
            forgetset = torch.utils.data.Subset(trainset, unlearn_idx)
            print('len of forget set: ', len(forgetset))  
            # print(forgetset.report())

            # if args.unlearn_method == 'RL':
            #     print('new forgetset for RL')
            #     new_classes = []
            #     for i in range(len(forgetset)):
            #         _, true_label = forgetset[i]
            #         random_label = np.random.choice([j for j in range(args.num_classes) if j != true_label])
            #         new_classes.append(random_label)

            #     if args.dataset == 'tinynet':
            #         trainset_tmp = basicDataset(trainset_all, transform=transform_adv, target_transform=None)
            #     elif args.dataset == 'cifar':
            #         trainset_tmp = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, transform=transform_adv) ### transofrm=transform_train
            #     forgetset = torch.utils.data.Subset(trainset_tmp, unlearn_idx)


            # advset_filtered = torch.utils.data.Subset(advset, unlearn_idx)
            # print('len of advset set: ', len(advset_filtered))  

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
            forgetloader = torch.utils.data.DataLoader(forgetset, shuffle=False, batch_size=args.batch_size, num_workers=1)
            
            if args.unlearn_method != 'retrain' and args.unlearn_method != 'reference' and args.unlearn_method != 'genmask':
                advloader = torch.utils.data.DataLoader(advset, shuffle=False, batch_size=args.batch_size, num_workers=1)
            # remainloader = torch.utils.data.DataLoader(trainset_filtered, shuffle=False, batch_size=128, num_workers=1)

            if req_idx == 0:
                if args.model == 'ResNet18':
                    if orig_flag:
                        net = ResNet18_orig(in_chan=in_chan, bn=bn_flag, device=device, elu_flag=False, num_classes=args.num_classes)
                        # net = ResNet18_orig(in_chan=in_chan, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_linear=False, bn_count=steps_count, device=device)
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
            print('--- forget set:')
            fs_loss, fs_acc = test(forgetloader, 200, criterion, writer=writer, mode='forget', model_path=None)
            if args.unlearn_method != 'retrain' and args.unlearn_method != 'reference' and args.unlearn_method != 'genmask':
                print('-- adv set:')
                adv_loss, adv_acc = test(advloader, 200, criterion, writer=writer, mode='adv', model_path=None)
            print('-- remain set:')
            remain_loss, remain_acc = test(remainloader, 200, criterion, writer=writer, mode='remain', model_path=None)
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
                print('saving results to ...', outdir)
                
                # if epoch % 5 == 0 and False:
                if True:
                    print('-- test set:')
                    ts_loss, ts_acc = test(testloader, epoch, criterion, writer=writer, model_path=model_path_test, mode='test', plot_images=True)
                    print('--- forget set:')
                    fs_loss, fs_acc = test(forgetloader, epoch, criterion, writer=writer, model_path=model_path_test, mode='forget')
                    if args.unlearn_method != 'retrain' and args.unlearn_method != 'reference':
                        print('-- adv set:')
                        adv_loss, adv_acc = test(advloader, epoch, criterion, writer=writer, model_path=model_path_test, mode='adv')
                    print('-- remain set:')
                    remain_loss, remain_acc = test(remainloader, epoch, criterion, writer=writer, model_path=model_path_test, mode='remain')

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

