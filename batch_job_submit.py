import os
import argparse
from helper import get_data
import pandas as pd
import time


parser = argparse.ArgumentParser(description='Submit batch jobs for unlearning experiments')
parser.add_argument('--job_type', default='train', help='type of job to run (train-evaluate)')
parser.add_argument('--dataset', default='cifar', help='dataset')
parser.add_argument('--model', default='ResNet18', help='Deep Learning model to train')
parser.add_argument('--method', default='orig', help='clipping method (use orig for no clipping)')
parser.add_argument('--mode', default='wBN')
parser.add_argument('--seed', default=-1, type=int)
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--LRsteps', default=40, type=int, help='LR scheduler step')
parser.add_argument('--widen_factor', default=1, type=int, help='widen factor for WideResNet')

parser.add_argument('--attack', default='pgdl2', type=str)
parser.add_argument('--remain', default='use', type=str)

parser.add_argument('--others_adv', action='store_true')
parser.add_argument('--adaptive', action='store_true')
parser.add_argument('--req_index', default=0, type=int) # default 1

parser.add_argument('--ablation_test', default=-1, type=int)
parser.add_argument('--salun_ratio', default='0.5', type=str, help='ratio of masking in salun')

parser.add_argument('--unlearn_method', default='adv', type=str)
parser.add_argument('--adv_images', default=None, type=str)
parser.add_argument('--adv_delta', default=None, type=str)
parser.add_argument('--unlearn_indices', default=None, type=str)
parser.add_argument('--model_path', default=None, type=str)
parser.add_argument('--model_count', default=1, type=int)
parser.add_argument('--trials', default=1, type=int)
parser.add_argument('--mask_path', default=None, type=str)
parser.add_argument('--inclusion_mat_path', default='keep_files/keep_m128_d60000_s0.csv', type=str)
parser.add_argument('--reference_mat', default='logs/scratch/cifar_ResNet18/unlearn/reference/ResNet18_orig__m128_d60000_s0_160_prob_matrix_logits_onehot.pt', type=str) ### cifar-18

args = parser.parse_args()
print('!!!!!!!! use_remain: ', args.remain)

if args.others_adv:
    print('others adv is in use!')

if args.dataset == 'tinynet':
    print('tinynet is in use!')
    args.inclusion_mat_path = '~/Amun/keep_files/keep_m128_d110000_s0.csv'
    args.reference_mat = '/work/hdd/bcwm/aebrahimpour/amun/logs/correct/scratch/tinynet_VGG_unnorm/unlearn/reference/VGG_orig__m128_d110000_s0_160_prob_matrix_logits_onehot.pt'

req_mode = 'single'
if args.adaptive:
    print('adaptive is in use!')
    req_mode = 'adaptive'
    unlearn_count = int(1000 * (1+ args.req_index))
else:
    unlearn_count = -1

if args.method == 'clip' and args.mode == 'noBN':
    args.reference_mat = '/work/hdd/bcwm/aebrahimpour/amun/logs/correct/scratch/cifar_unnorm/unlearn/reference/ResNet18_fastclip_cs100_noBN_m128_d60000_s0_100_prob_matrix_logits_onehot.pt'
    print('clip noBN reference mat in use!')

if __name__ == '__main__':
    dataset = args.dataset
    model = args.model
    if args.seed == -1:
        # seed_list = [10**i for i in range(5)]
        seed_list = [10**i for i in range(3)]
        # seed_list = [10**i for i in range(3, 5)]
    elif args.seed == -4:
        seed_list = [i for i in range(4)]
    elif args.seed == -32:
        seed_list = [i for i in range(32)]
    elif args.seed == -64:
        seed_list = [i for i in range(64)]
    elif args.seed == -128:
        seed_list = [i for i in range(128)]
    else:
        seed_list = [args.seed]

    if args.job_type == 'RMIA_ref':
        seed_list = [1]  # for RMIA reference, we only need one seed

    convsn_list = [1.]
    steps = 50 # this is clipBN steps

    method = args.method
    print('method: ', method)
    if method == 'all':
        methods = ['orig', 'fastclip_tlower_cs100']
    elif method == 'clip':
        methods = ['clip'] # ['fastclip_cs100']
    elif method[:4] == 'fast':
        methods = ['fastclip_tlower_cs100', 'fastclip_tlower_cs50']
    else:
        methods = [method]

    if args.model  not in ['ResNet18', 'simpleConv', 'wideResnet', 'VGG', 'vit']:
        raise ValueError('model must be one of ResNet18, DLA, SimpleConv')


    source_model_seeds = [1, 10, 100] # sms
    if args.model_count == 1:
        source_model_seeds = [-1] # sms
    else:
        source_model_seeds = source_model_seeds[:args.model_count]

    mode = args.mode
    if mode == 'all':
        modes = ['wBN', 'noBN']
    else:
        modes = [mode]

    for mode in modes:
        for method in methods:
            if method == 'orig':
                convsn_list_tmp = [1.0]
            else:
                convsn_list_tmp = convsn_list
                print('method: ', method)
            for convsn in convsn_list_tmp:
                for seed in seed_list:
                    for sms in source_model_seeds:
                        try:
                            if args.job_type == 'train':
                                command = f"sbatch slurm_scripts/job_submit.slurm {method} {mode} {seed} {convsn} {args.widen_factor} {args.model} {args.lr} {args.dataset}"
                            elif args.job_type == 'advdata': # to perform adversarial attack and find adv examples and delta values
                                command = f"sbatch slurm_scripts/job_eval_submit.slurm {method} {mode} {seed} {convsn} {args.widen_factor} {args.model} {args.lr} {args.dataset} {args.model_path} {args.epoch} {args.attack}"
                            elif args.job_type == 'MIA':
                                if sms < 0:
                                    command = f"sbatch slurm_scripts/job_mia_submit.slurm {method} {mode} {seed} {args.model} {args.dataset} {args.adv_images} {args.adv_delta} {args.unlearn_indices} {args.model_path} {args.epoch} {args.mask_path} {args.lr} {args.LRsteps} {args.trials}"
                                else:
                                    if args.adv_images is not None and args.adv_delta is not None:
                                        adv_images_parts = args.adv_images.split('/adv_data/')
                                        adv_images = adv_images_parts[0][:-1] + f'{sms}/adv_data/' + adv_images_parts[1]

                                        adv_delta_parts = args.adv_delta.split('/adv_data/')
                                        adv_delta = adv_delta_parts[0][:-1] + f'{sms}/adv_data/' + adv_delta_parts[1]
                                    else:
                                        adv_images = None
                                        adv_delta = None

                                    args.unlearn_method = args.model_path.split('unlearn/')[1].split('/')[0]

                                    print(args.unlearn_method)
                                    if args.unlearn_method == 'retrain':
                                        print('model is retrain!')
                                        model_name = args.model_path.split('/')[-1]
                                    else:
                                        if args.mask_path is None:
                                            print('mask path is None')
                                            model_name = args.model_path.split('/')[-3]
                                        else:
                                            model_name = args.model_path.split('/')[-4]

                                    print('model name: ', model_name)

                                    model_path_parts = args.model_path.split(model_name)
                                    if args.unlearn_method == 'retrain':
                                        model_path = model_path_parts[0] + model_name + f'{sms}/'
                                    else:
                                        model_path = model_path_parts[0] + model_name[:-1] + f'{sms}/' + model_path_parts[1] 

                                    print('model path: ', model_path)

                                    command = f"sbatch slurm_scripts/job_mia_submit.slurm {method} {mode} {seed} {args.model} {args.dataset} {adv_images} {adv_delta} {args.unlearn_indices} {model_path} {args.epoch} {args.mask_path} {args.lr} {args.LRsteps} {args.trials}"

                            elif args.job_type == 'RMIA':
                                if sms < 0:
                                    command = f"sbatch slurm_scripts/job_rmia_submit.slurm {method} {mode} {seed} {args.model} {args.dataset} {args.adv_images} {args.adv_delta} {args.unlearn_indices} {args.model_path} {args.epoch} {args.mask_path} {args.lr} {args.LRsteps} {args.trials} {args.inclusion_mat_path} {args.reference_mat} {unlearn_count} {args.req_index}"
                                else:
                                    if args.adv_images is not None and args.adv_delta is not None:
                                        adv_images_parts = args.adv_images.split('/adv_data/')
                                        adv_images = adv_images_parts[0][:-1] + f'{sms}/adv_data/' + adv_images_parts[1]

                                        adv_delta_parts = args.adv_delta.split('/adv_data/')
                                        adv_delta = adv_delta_parts[0][:-1] + f'{sms}/adv_data/' + adv_delta_parts[1]
                                    else:
                                        adv_images = None
                                        adv_delta = None

                                    args.unlearn_method = args.model_path.split('unlearn/')[1].split('/')[0]

                                    print(args.unlearn_method)
                                    if args.unlearn_method == 'retrain':
                                        print('model is retrain!')
                                        model_name = args.model_path.split('/')[-1]
                                    else:
                                        if args.mask_path is None:
                                            print('mask path is None')
                                            model_name = args.model_path.split('/')[-3]
                                        else:
                                            model_name = args.model_path.split('/')[-4]

                                    print('model name: ', model_name)

                                    model_path_parts = args.model_path.split(model_name)
                                    if args.unlearn_method == 'retrain':
                                        model_path = model_path_parts[0] + model_name + f'{sms}/'
                                    else:
                                        model_path = model_path_parts[0] + model_name[:-1] + f'{sms}/' + model_path_parts[1] 

                                    print('model path: ', model_path)

                                    command = f"sbatch slurm_scripts/job_rmia_submit.slurm {method} {mode} {seed} {args.model} {args.dataset} {adv_images} {adv_delta} {args.unlearn_indices} {model_path} {args.epoch} {args.mask_path} {args.lr} {args.LRsteps} {args.trials} {args.inclusion_mat_path} {args.reference_mat} {unlearn_count} {args.req_index}"

                            elif args.job_type == 'RMIA_ref':
                                command = f"sbatch slurm_scripts/job_rmia_ref_submit.slurm {method} {mode} {args.model} {args.dataset} {args.model_path} {args.epoch} {args.trials} {args.inclusion_mat_path}"

                            elif args.job_type == 'advunlearn':
                                if args.unlearn_method == 'reference':
                                    if args.dataset == 'cifar':
                                        dataset_size = 60000
                                    elif args.dataset == 'tinynet':
                                        dataset = 110000

                                    keep = get_data(dataset_size=dataset_size)
                                    df = pd.DataFrame(keep)
                                    outdir = 'keep_files'
                                    if not os.path.exists(outdir):
                                        os.makedirs(outdir)
                                    filename = f'{outdir}/keep_m128_d{dataset_size}_s0.csv'
                                    df.to_csv(filename, index=False)

                                    file_flag = False
                                    while not file_flag:
                                        try:
                                            tmp = pd.read_csv(filename, header=0).values
                                            print('seed: 0', tmp[0].shape)
                                            file_flag = True
                                        except:
                                            print('sleeping for 2 seconds to wait for indices file to be readable')
                                            time.sleep(2.0)
                                            continue

                                if sms < 0:
                                    command = f"sbatch slurm_scripts/job_adv_unlearn_submit.slurm {method} {mode} {seed} {convsn} {args.model} {args.lr} {args.dataset} {args.adv_images} {args.adv_delta} {args.unlearn_indices} {args.model_path} {args.unlearn_method} {args.mask_path} {args.LRsteps} {args.epoch} {args.remain} {args.attack} {args.salun_ratio} {req_mode} {args.ablation_test}"
                                else:
                                    if not args.others_adv:
                                        if args.adv_images is not None and args.adv_delta is not None:
                                            adv_images_parts = args.adv_images.split('/adv_data/')
                                            adv_images = adv_images_parts[0][:-1] + f'{sms}/adv_data/' + adv_images_parts[1]

                                            adv_delta_parts = args.adv_delta.split('/adv_data/')
                                            adv_delta = adv_delta_parts[0][:-1] + f'{sms}/adv_data/' + adv_delta_parts[1]
                                        else:
                                            adv_images = None
                                            adv_delta = None

                                        command = f"sbatch slurm_scripts/job_adv_unlearn_submit.slurm {method} {mode} {seed} {convsn} {args.model} {args.lr} {args.dataset} {adv_images} {adv_delta} {args.unlearn_indices} {args.model_path}{sms} {args.unlearn_method} {args.mask_path} {args.LRsteps} {args.epoch} {args.remain} {args.attack} {args.salun_ratio} {req_mode} {args.ablation_test}"
                                    else:
                                        other_sms = sms*10
                                        if other_sms == 1000:
                                            other_sms = 1

                                        adv_images_parts = args.adv_images.split('/adv_data/')
                                        adv_images = adv_images_parts[0][:-1] + f'{other_sms}/adv_data/' + adv_images_parts[1]

                                        adv_delta_parts = args.adv_delta.split('/adv_data/')
                                        adv_delta = adv_delta_parts[0][:-1] + f'{other_sms}/adv_data/' + adv_delta_parts[1]

                                        command = f"sbatch slurm_scripts/job_adv_unlearn_submit.slurm {method} {mode} {seed} {convsn} {args.model} {args.lr} {args.dataset} {adv_images} {adv_delta} {args.unlearn_indices} {args.model_path}{sms} {args.unlearn_method} {args.mask_path} {args.LRsteps} {args.epoch} {args.remain} {args.attack} {args.salun_ratio} {req_mode} {args.ablation_test}"

                            print(command)
                            os.system(command)

                        except Exception as e:
                            print(e)
                            continue

