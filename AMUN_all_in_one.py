import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models import *
from models.resnet_orig import ResNet18_orig
import pandas as pd
import random
import time
from pgdl2_modified import PGDL2
from helper import *
    

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Unlearning')
parser.add_argument('--source_model_path', type=str)
parser.add_argument('--outdir', type=str)
parser.add_argument('--unlearn_indices', type=str)
parser.add_argument('--unlearn_method', default='advonly', type=str)

parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--LRsteps', default=5, type=int, help='LR scheduler step')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=128, type=int, help='number of classes in the dataset')
parser.add_argument('--seed', default=1, type=int, help='seed value')

parser.add_argument('--use_remain', default=True, type=bool)
parser.add_argument('--remain', default='use', type=str)
parser.add_argument('--attack', default='pgdl2', type=str)

args = parser.parse_args()


if args.remain != 'use':
    args.use_remain = False
else:
    args.use_remain = True
print('use remain flag: ', args.use_remain)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('==========', device)
if device == 'cuda':
    print('chosen: ', device)
    cudnn.benchmark = True
args.device = device


# Training function
def train(epoch, optimizer, scheduler, criterion, unlearn_method='advonly', model_path="./checkpoints/", advset=None, save_model=False):
    print('\nEpoch: %d' % epoch)
    print('unlearn method: ', unlearn_method)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = -1

    print('\ninside train function :')
    print('trainset :', len(trainset) )
    print('unl idx :', len(unlearn_idx) )

    if args.use_remain:
        trainset_combined = torch.utils.data.ConcatDataset([trainset, advset])
    else:
        if unlearn_method == 'advonly':
            print('only advset is being used!')
            trainset_combined = advset
        else:
            trainset_combined = torch.utils.data.ConcatDataset([forgetset, advset])

    print('transet_combined len: ', len(trainset_combined))
    trainloader = torch.utils.data.DataLoader(trainset_combined, shuffle=True, batch_size=args.batch_size, num_workers=1)

    start = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if epoch == 0 and batch_idx == 0:
            print('inputs shape: ', inputs.shape)
        inputs, targets = inputs.float().to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    tot_time = time.time() - start
    print('time: ', tot_time)
    print('train - acc', 100.*correct/total)
    print('train - loss', train_loss/(batch_idx+1))
    scheduler.step()

    if save_model:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
        }
        model_path_i = model_path + ".%d" % (epoch)
        torch.save(state, model_path_i)

    net.eval()
    return train_loss/(batch_idx+1), 100.*correct/total, advset


# Test function
def test(net, loader, criterion, mode='test'):
    global best_acc
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

    print(mode + '/acc', 100.*correct/total)
    print(mode + '/loss', test_loss/(batch_idx+1))
    return test_loss/(batch_idx+1), 100.*correct/total


if __name__ == "__main__":
    seed_val = args.seed
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
    step_size = args.LRsteps

    unlearn_indices = pd.read_csv(args.unlearn_indices)['unlearn_idx'].values
    count_unlearn = len(unlearn_indices)
    print('count_unlearn: ', count_unlearn)
    unlearn_idx = [int(i) for i in unlearn_indices]

    outdir = args.outdir
    print('outdir: ', outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    #####################################################################################
    #================================== Loading data =================================#
    #####################################################################################
    print('==> Preparing data..')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, transform=transform_train) ### transofrm=transform_train
    trainset_clean = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, transform=transform_test) ### transofrm=transform_train
    testset = torchvision.datasets.CIFAR10( root='./data', train=False, download=True, transform=transform_test)


    #####################################################################################
    #============ building the forgetset and the set that will be used for unlearning =======#
    #####################################################################################
    print('===> Building the forget set and the filtered train set...')

    removed_classes = [trainset[i][1] for i in unlearn_idx]
    df = pd.DataFrame({'unlearn_idx': unlearn_idx, 'removed_classes': removed_classes})

    ### remove the unlearned images from the trainset
    trainset_filtered = torch.utils.data.Subset(trainset, list(set(range(len(trainset))) - set(unlearn_idx)))
    print('len of filtered trainset: ', len(trainset_filtered))  

    forgetset = torch.utils.data.Subset(trainset_clean, unlearn_idx)
    print('len of forget set: ', len(forgetset))  


    if args.unlearn_method == 'advonly':
        trainset = trainset_filtered

    print('final len of trainset: ', len(trainset))  
    print('-----------------------------------------------------------------')

    testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=args.batch_size, num_workers=1)
    forgetloader = torch.utils.data.DataLoader(forgetset, shuffle=False, batch_size=args.batch_size, num_workers=1)
    remainloader = torch.utils.data.DataLoader(trainset_filtered, shuffle=False, batch_size=args.batch_size, num_workers=1)


    #####################################################################################
    #============================== loading the original model =========================#
    #####################################################################################
    print('===> Loading the original model...')

    net = ResNet18_orig(device=device)
    net = net.to(device)
    net = nn.DataParallel(net) ### adds the "module." prefix to the state_dict keys
    criterion = nn.CrossEntropyLoss()

    checkpoint = torch.load(args.source_model_path)
    net.load_state_dict(checkpoint['state_dict'])
    print('model loaded')
    net.eval()

    print('-- test set:')
    # ts_loss, ts_acc = test(net, testloader, criterion, mode='test')
    ts_loss, ts_acc = 0., 0.
    print('--- forget set:')
    # fs_loss, fs_acc = test(net, forgetloader, criterion, mode='forget')
    fs_loss, fs_acc = 0., 0.
    print('-- remain set:')
    # remain_loss, remain_acc = test(net, remainloader, criterion, mode='remain')
    remain_loss, remain_acc = 0., 0.

    ts_loss_list = [ts_loss]
    ts_acc_list = [ts_acc]
    fs_loss_list = [fs_loss]
    fs_acc_list = [fs_acc]
    re_loss_list = [remain_loss]
    re_acc_list = [remain_acc]


    #####################################################################################
    #============================ computing the adversarial set ========================#
    #####################################################################################
    print('===> Computing the adversarial set...')
    
    net.eval()
    percentile = 50

    transform_adv = transforms.Compose([
        transforms.ToTensor(),
    ])

    # consider a sample of 50 forget set images to find the initial epsilons
    indices_sample = list(range(min(50, len(forgetset))))
    advset_init, df_adv_init = compute_advset(args, forgetset, net, initial_eps=0.01, outdir=outdir, transform=transform_adv, indices=indices_sample)
    print('adv df: ', df_adv_init)

    smallest_eps = df_adv_init['smallest_eps'].values
    ## compute the corresponding percentile of the smallest epsilons
    eps_percentile = np.percentile(smallest_eps, percentile)
    print(f'{percentile}th percentile of smallest epsilons: ', eps_percentile)

    # compute the adversarial set for the rest of forget set images
    indices_others = list(set(range(len(forgetset))) - set(indices_sample))
    advset_others, df_adv_others = compute_advset(args, forgetset, net, initial_eps=eps_percentile, outdir=outdir, transform=transform_adv, indices=indices_others)

    # concat the init and others adv sets
    advset = torch.utils.data.ConcatDataset([advset_init, advset_others])
    print('length of final advset: ', len(advset))
    advloader = torch.utils.data.DataLoader(advset, shuffle=False, batch_size=args.batch_size, num_workers=1)
    adv_loss, adv_acc = test(net, advloader, criterion, mode='adv')


    #####################################################################################
    #============================ fine-tuning the model ==============================#
    #####################################################################################
    print('===> Fine-tuning the model...')

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    T_max = args.epochs
    print('Tmax: ', T_max)
    model_path =  outdir + '/_ckpt'

    for epoch in range(T_max):
        time_start = time.time()
        tr_loss, tr_acc, _ = train(epoch, optimizer, scheduler, criterion, unlearn_method=args.unlearn_method, model_path=model_path, advset=advset)
        print('epoch time: ', time.time() - time_start)
        
        print('-- test set:')
        ts_loss, ts_acc = test(net, testloader, criterion, mode='test')
        print('--- forget set:')
        fs_loss, fs_acc = test(net, forgetloader, criterion, mode='forget')
        print('-- adv set:')
        adv_loss, adv_acc = test(net, advloader, criterion, mode='adv')
        print('-- remain set:')
        remain_loss, remain_acc = test(net, remainloader, criterion, mode='remain')


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

    df = pd.DataFrame({'ts_loss': ts_loss_list, 'ts_acc': ts_acc_list, 'fs_loss': fs_loss_list, 'fs_acc': fs_acc_list, 're_loss': re_loss_list, 're_acc': re_acc_list})
    print('saving results to ...', outdir)
    df.to_csv(outdir + '/loss_acc_results.csv')

