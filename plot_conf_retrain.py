import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import rc
import sys
import argparse
from scipy.stats import norm

parser = argparse.ArgumentParser(description='Plotting script for retraining results')
parser.add_argument('--count', default='', type=str, help='unlearn count')
parser.add_argument('--path', default='', type=str, help='path to summary csv file')
args = parser.parse_args()

def compute_conf(probs):
    confs = np.log(probs/(1.-probs)) 
    return confs

def fit_gaussian(data):
    mu, std = norm.fit(data)
    return mu, std

def plot_gaussian(ax, mu, std, color, label, linestyle):
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, color=color, label=label, linestyle=linestyle)


if __name__ ==  '__main__':

    sb.set_context(
        "talk",
        font_scale=1,
    )
    sb.set_style('white')
    sb.set_style('ticks')

    linestyles = ['solid', 'dashed', 'dotted', 'dashdot', 'dashdot', 'dashdot', 'dotted', 'dotted','dotted']
    ####################################################################################


    fig, ax = sb.mpl.pyplot.subplots(1, 1, figsize=(6.5, 4))
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    colors = sb.mpl_palette('tab10', n_colors=9)

    init_dir = 'plots/retrain_confs_'
    file_name = args.path 
    div = 10

    df_forget = pd.read_csv(file_name + 'forget_probs.csv')
    df_remain = pd.read_csv(file_name + 'remain_probs.csv')
    df_test = pd.read_csv(file_name + 'test_probs.csv')



    ax.hist(
        compute_conf(df_remain['prob']),
        label='Remain',
        color=colors[0],
        density=True,
        # linestyle=linestyles[1],
        alpha=0.5,
        bins=20)

    mu, std = fit_gaussian(compute_conf(df_remain['prob']))
    plot_gaussian(ax, mu, std, colors[0], 'Remain', linestyles[0])

    
    ax.hist(
        compute_conf(df_test['prob']),
        label='Test',
        color=colors[1],
        density=True,
        # linestyle=linestyles[2],
        alpha=0.5,
        bins=20)

    mu, std = fit_gaussian(compute_conf(df_test['prob']))
    plot_gaussian(ax, mu, std, colors[1], 'Test', linestyles[1])


    ax.hist(
        compute_conf(df_forget['prob']),
        label='Forget',
        color=colors[2],
        density=True,
        # linestyle=linestyles[0],
        alpha=0.5,
        bins=20)
    
    mu, std = fit_gaussian(compute_conf(df_forget['prob']))
    plot_gaussian(ax, mu, std, colors[2], 'Forget', linestyles[2])

    # z = ax[0]
    z = ax
    # z.set_ylabel(r"Accuracy", fontsize=14)
    z.set_xlabel(r"Confidence", fontsize=14)
    # z.set_xticks([50, 100, 150, 200], ['50', '100', '150', '200'], minor=True)

    z.tick_params(direction='in', length=6, width=2, colors='k', which='major')
    z.tick_params(direction='in', length=4, width=1, colors='gray', which='minor')
    z.set_aspect(aspect='auto', adjustable='datalim')
    z.minorticks_on()

    # z.legend(loc='center left', fontsize=10, bbox_to_anchor=(1, 0.5))
    z.legend(loc=0, fontsize=10)


    sb.despine(left=True, offset=10, trim=False)
    sb.mpl.pyplot.tight_layout()
        

    file_name = init_dir + file_name.split('/')[-1]
    new_file_name = file_name[:-4] + '_acc_' + args.count
    print(new_file_name)
    plt.savefig(new_file_name, dpi=300)
    plt.clf()



