import os
import argparse
import argparse


parser = argparse.ArgumentParser(description='Hyperparameter tuning for unlearning')
parser.add_argument('--dataset', default='cifar', help='dataset (cifar or tinynet)')
parser.add_argument('--method', default='amun', help='unlearning method. use "all" for all methods')
parser.add_argument('--model', default='ResNet18', help='Deep Learning model to train (e.g., ResNet18, VGG)')
parser.add_argument('--seed', default=1, type=int, help='unlearning seed to use. -1 for all seeds')
parser.add_argument('--epoch', default=10, type=int, help='number of epochs for unlearning phase')
parser.add_argument('--count', default=5000, type=int, help='number of unlearn indices')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate for unlearning phase')
parser.add_argument('--LRsteps', default=1, type=int, help='LR scheduler step')
parser.add_argument('--adaptive', action='store_true')
parser.add_argument('--lipschitz', action='store_true')
parser.add_argument('--no_remain', action='store_true')
parser.add_argument('--ablation', action='store_true')
args = parser.parse_args()


if __name__ == '__main__':
    count = args.count # number of unlearn indices
    epoch_count = args.epoch # number of epochs for unlearning phase
    adaptive = args.adaptive # True for adaptive unlearning, False for non-adaptive unlearning
    dataset = args.dataset # 'cifar' or 'imagenet'
    model = args.model # or 'VGG'
    use_remain = not args.no_remain # True to use remaining data, False to not use remaining data
    lipschitz = args.lipschitz

    model_count = 3

    if lipschitz:
        bn_flag = 'noBN'
        method_short = 'clip'
        method_complete = 'clip1.0'
    else:
        bn_flag = 'wBN'
        method_short = 'orig'
        method_complete = 'orig'

    attack_name = ''
    attack = 'pgdl2'

    if use_remain:
        remain = 'use'
    else:
        remain = 'no_use'

    dataset_name = f'{dataset}_{model}'

    if args.lr == -1:
        lr_list = [1/10**i for i in range(1, 6)] + [0.05, 0.005]
    else:
        lr_list = [args.lr] 
    print('lr list: ', lr_list)

    if args.seed == -1:
        unlearn_seed_list = [1, 10, 100]
    elif args.seed == -2:
        unlearn_seed_list = [10, 100]
    else:
        unlearn_seed_list = [args.seed]

    if args.LRsteps == -1:
        step_list = [1, 5, 10]
    else:
        step_list = [args.LRsteps]
    print('step list: ', step_list)

    if args.method == 'all':
        methods = ['amun', 'amun_sa', 'advonly', 'advonly_sa', 'salun', 'RL', 'FT', 'GA', 'BS', 'l1']
    else:
        methods = [args.method]

    # salun_ratio = ['0.1', '0.3', '0.5', '0.7', '0.9']
    salun_ratio = ['0.5']

    if args.ablation:
        ablation_test_list = [-1,1,2,3,4,5]
        methods = ['amun', 'advonly']
    else:
        ablation_test_list = [-1]
    print('methods: ', methods)


    for unlearn_seed in unlearn_seed_list:
        for lr in lr_list:
            for step in step_list:
                for method in methods:
                    if method in ['salun', 'amun_sa', 'advonly_sa']:
                        salun_ratio_list = salun_ratio
                    else:
                        salun_ratio_list = ['0.5']

                    for ratio in salun_ratio_list:
                        for ablation_test in ablation_test_list:
                            command = f"python batch_job_submit.py --job_type advunlearn --attack {attack} --model {model} --method {method_short} --mode {bn_flag} --seed 1 --dataset {dataset} --adv_images ~/AMUN/logs/scratch/{dataset_name}/vanilla_{method_complete}_{bn_flag}_1/adv_data/seed_1/adv_tensor{attack_name}.pt --adv_delta ~/AMUN/logs/scratch/{dataset_name}/vanilla_{method_complete}_{bn_flag}_1/adv_data/seed_1/smallest_eps{attack_name}.csv --unlearn_indices ~/AMUN/unlearn_indices/{dataset}/{count}/seed_{unlearn_seed}.csv --model_path ~/AMUN/logs/scratch/{dataset_name}/vanilla_{method_complete}_{bn_flag}_  --unlearn_method {method} --model_count {model_count} --LRsteps {step} --lr {lr}  --remain {remain} --salun_ratio {ratio} --epoch {epoch_count} --ablation_test {ablation_test}"

                            if adaptive:
                                command += ' --adaptive'

                            print(command)
                            os.system(command)


