import os
import argparse


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--job_type', default='unlearn', type=str, help='The type of job to run')
parser.add_argument('--dataset', default='cifar', help='dataset')
parser.add_argument('--model', default='ResNet18', help='Deep Learning model to train')
parser.add_argument('--method', default='all', help='clipping method (use orig for no clipping)')
parser.add_argument('--mode', default='wBN')

parser.add_argument('--filter', default='None', help='how to filter the train data')
parser.add_argument('--unlearn', default=-1, type=int, help='number of samples to unlearn')
parser.add_argument('--path', default=None, type=str)
parser.add_argument('--save_checkpoints', default=False, type=bool)
parser.add_argument('--up_sample', default=0, type=int)

parser.add_argument('--seed', default=-1, type=int)
args = parser.parse_args()


if __name__ == '__main__':
    filter_method = args.filter
    unlearn_count_input = args.unlearn
    score_path = args.path
    save_checkpoints = args.save_checkpoints
    job_type = args.job_type

    dataset = args.dataset
    model = args.model
    if args.seed == -1:
        if job_type != 'evaluate' and filter_method == 'None' and unlearn_count_input == 0:
            # seed_list = [0,1, 2, 3, 4, 5]
            # seed_list = [0,6,7,8,9]
            # seed_list = list(range(0,10))
            # seed_list = list(range(50))
            seed_list = [0]

        else:
            # seed_list = [10**i for i in range(1,6)]
            seed_list = [10**i for i in range(1,3)]
            # seed_list = list(range(0,50))
    else:
        seed_list = [args.seed]

    if unlearn_count_input == -1:
        # unlearn_count_list = [0,5,10,20,50,100,200,500,1000,2000]
        # unlearn_count_list = [5,10,20,50,100,200,500,1000,2000,3000,4000]
        # unlearn_count_list = [3000,4000,4900,4950,4990,5000]
        # unlearn_count_list = [4000,4500,4900,4950,4990]
        # unlearn_count_list = [0,5,100,1000]
        # unlearn_count_list = [0,1000,2000,3000,4000,4500,4900]
        # unlearn_count_list = [1000,2000,3000,4000,4500,4900]
        unlearn_count_list = [50, 500, 5000, 25000]
    else:
        unlearn_count_list = [unlearn_count_input]

    print(seed_list)
        

    steps = 50 # this is clipBN steps if activated

    method = args.method
    if method == 'all':
        methods = ['lip4conv', 'gouk', 'nsedghi', 'miyato', 'orig', 'fastclip_cs50']
    elif method[:4] == 'fast' or method == 'clip':
        methods = ['fastclip_cs50']
    else:
        methods = [method]

    if args.model  not in ['ResNet18', 'ResNet9', 'DLA', 'SimpleConv']:
        raise ValueError('model must be one of ResNet18, DLA, SimpleConv')

    mode = args.mode
    if mode == 'all':
        modes = ['wBN', 'noBN', 'clipBN_hard']
    elif mode == 'regular':
        modes = ['wBN', 'noBN']
    elif mode == 'clipBN':
        modes = ['clipBN_hard']
    elif mode == 'BN':
        modes = ['wBN','clipBN_hard']
    else:
        modes = [mode]

    for mode in modes:
        for method in methods:
            if method == 'orig' and (mode == 'clipBN_hard'):
                continue
            for unlearn_count in unlearn_count_list:
                for seed in seed_list:
                    try:
                        # command = f"python main_jobsubmit.py --dataset {dataset} --method {method} --mode {mode} --seed {seed} --steps {steps} --model {model} --filter {filter_method} --unlearn {unlearn_count} --path {score_path} --save_checkpoints {save_checkpoints}"

                        # command = f"python main.py --dataset {dataset} --method {method} --mode {mode} --seed {seed} --steps {steps} --model {model} --filter {filter_method} --unlearn {unlearn_count} --path {score_path} --save_checkpoints {save_checkpoints}"

                        if job_type == 'unlearn':
                            command = f"sbatch unlearn_job_submit.slurm {dataset} {method} {model} {mode} {seed} {filter_method} {unlearn_count} {args.up_sample}"
                        elif job_type == 'train':
                            command = f"sbatch job_submit.slurm {dataset} {method} {mode} {seed} {steps} {model}"
                        elif job_type == 'evaluate':
                            # score path here is the base classifier path
                            command = f"sbatch evaluate_job_submit.slurm {dataset} {method} {model} {mode} {seed} {unlearn_count} {score_path}"
                        
                        print(command)
                        os.system(command)

                    except Exception as e:
                        print(e)
                        continue
