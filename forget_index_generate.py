import pandas as pd
import numpy as np
import sys
import os

if __name__ == '__main__':
    seed = 0
    dataset_name = sys.argv[1]  # 'cifar', 'tinynet'
    if dataset_name == 'cifar':
        dataset_size = 50000 # cifar-10
    elif dataset_name == 'tinynet':
        dataset_size = 100000 # tiny-imagenet
    else:
        print('unknown dataset!')
        exit(0)
    
    forget_size = int(sys.argv[2])  # number of unlearn indices, e.g., 10000

    for seed in [1,10,100]:
        np.random.seed(seed)
        unl_idx = np.random.choice(dataset_size, forget_size, replace=False)
        print(unl_idx.shape)
        print(np.sum(unl_idx))
        df = pd.DataFrame({'unlearn_idx': unl_idx})
        # save to file:
        outdir = f'unlearn_indices/{dataset_name}/{forget_size}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        df.to_csv(f'{outdir}/seed_{seed}.csv', index=False)

