import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics


def get_data(model_count=64, seed=0, dataset_size=40000, pkeep=0.5):
    np.random.seed(seed)
    keep = np.random.uniform(0,1,size=(model_count, dataset_size))
    order = keep.argsort(0)
    keep = order < int(pkeep * model_count)
    # keep = np.array(keep, dtype=bool)
    return keep


if __name__ == '__main__':
    model_count = 128
    seed = 0
    # dataset_size = 60000 # cifar-10
    dataset_size = 110000 # tiny-imagenet
    pkeep = 0.5
    keep = get_data(model_count, seed, dataset_size, pkeep)
    print(keep)
    print(keep.shape)
    print(np.sum(keep, axis=0))
    print(np.sum(keep, axis=1))
    print(np.sum(keep))

    # save to file:
    df = pd.DataFrame(keep)
    df.to_csv('keep_files/keep_m' + str(model_count) + '_d' + str(dataset_size) + '_s' + str(seed) + '.csv', index=False)

