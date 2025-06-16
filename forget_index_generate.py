import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import sys


if __name__ == '__main__':
    dataset = sys.argv[1] # cifar or tinynet
    if dataset not in ['cifar', 'tinynet']:
        raise ValueError("Dataset must be either 'cifar' or 'tinynet'.")
    dataset_size = int(sys.argv[2]) # 50000 for cifar10 and 90000 for tiny-imagenet
    forgetset_size = int(sys.argv[3]) # 5000 for cifar10 and 10000 for tiny-imagenet

    dir_path = 'unlearn_indices/' + dataset + '/' + str(forgetset_size)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for seed in [1,10,100]:
        np.random.seed(seed)
        unl_idx = np.random.choice(dataset_size, forgetset_size, replace=False)
        print(unl_idx.shape)
        print(np.sum(unl_idx))
        df = pd.DataFrame({'unlearn_idx': unl_idx})
        print(df.head())
        df.to_csv(dir_path + f'/seed_{seed}.csv', index=False)

