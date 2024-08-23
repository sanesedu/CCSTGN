import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import DetCurveDisplay
from scipy.stats import norm

keys_full_name = {
    'bs': 'Batch size',
    'lr': 'Learning rate',
    'wd': 'Weight decay',
    'lambda_': r'$\lambda$',
    'n_epochs': 'Number of epochs',
    'memory_dim': 'Memory dimensionality',
    'n_neighbors': 'Number of neighbors',
    'n_layers': 'Number of layers',
    'agg_function': 'Aggregation function'
}

def comparison_plot(df, key='', path='.', title=''):

    fig, ax1 = plt.subplots()

    width = 0.2  # width of the bars
    x = np.arange(len(df[key]))

    # Plot 'AUC' and 'F1' on left y-axis
    ax1.bar(x - 3*width/2, df['AUC'], width, color='tab:blue', alpha=0.5, label='AUC')
    ax1.bar(x - width/2, df['F1'], width, color='tab:orange', alpha=0.5, label='F1')
    ax1.set_ylabel('Percentage (%) for AUC and F1')
    ax1.tick_params(axis='y')

    # Plot second y-axis for 'FPR' and 'FNR' sharing same x-axis
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, df['FPR'], width, color='tab:green', alpha=0.5, label='FPR')
    ax2.bar(x + 3*width/2, df['FNR'], width, color='tab:red', alpha=0.5, label='FNR')
    ax2.set_ylabel('Percentage (%) for FPR and FNR')
    ax2.tick_params(axis='y')

    # Set xticks as the discrete values in 'key'
    ax1.set_xticks(x)
    ax1.set_xticklabels(df[key].unique())
    key_xlabel = keys_full_name[key] if key in keys_full_name.keys() else key
    ax1.set_xlabel(key_xlabel)

    # To ensure that the right y-label is not clipped
    fig.tight_layout()  

    # ask matplotlib for the plotted objects and their labels
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(title)

    plt.savefig(f'{path}/results.eps', dpi=300, bbox_inches='tight')
    plt.savefig(f'{path}/results.png', dpi=300, bbox_inches='tight')

def evolution_plots(df, path='.', title='', filename=''):
    
    fig1, ax1 = plt.subplots()
    x = np.arange(df.shape[0])

    ax1.plot(x, df['Train loss'], color='tab:blue', label='Train loss')
    ax1.plot(x, df['Validation loss'], color='tab:orange', label='Validation loss')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')

    ax1.set_xticks(x)
    ax1.set_xticklabels(x)
    ax1.set_xlabel('Epoch')
    
    ax1.legend(loc='upper right')
    
    fig1.tight_layout()  
    plt.title(title)
    
    plt.savefig(f'{path}/{filename}_loss.eps', dpi=300, bbox_inches='tight')
    plt.savefig(f'{path}/{filename}_loss.png', dpi=300, bbox_inches='tight')

    fig2, ax2 = plt.subplots()
    x = np.arange(df.shape[0])
    
    # Plot 'AUC' and 'F1' on left y-axis
    ax2.plot(x, df['AUC'], color='tab:blue', label='AUC')
    ax2.plot(x, df['F1'], color='tab:orange', label='F1')
    ax2.set_ylabel('Percentage (%) for AUC and F1')
    ax2.tick_params(axis='y')

    # Plot second y-axis for 'FPR' and 'FNR' sharing same x-axis
    ax3 = ax2.twinx()
    ax3.plot(df['FPR'], color='tab:green', label='FPR')
    ax3.plot(df['FNR'], color='tab:red', label='FNR')
    ax3.set_ylabel('Percentage (%) for FPR and FNR')
    ax3.tick_params(axis='y')

    ax2.set_xticks(x)
    ax2.set_xticklabels(x)
    ax2.set_xlabel('Epoch')
    
    # ask matplotlib for the plotted objects and their labels
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    fig2.tight_layout()  
    plt.title(title)
    
    plt.savefig(f'{path}/{filename}_metrics.eps', dpi=300, bbox_inches='tight')
    plt.savefig(f'{path}/{filename}_metrics.png', dpi=300, bbox_inches='tight')

def det_plot(fprs, fnrs, idxs, thr, path='.', names=[]):
    
    fig, ax = plt.subplots()
    
    for i in range(len(names)):
        DetCurveDisplay(fpr=fprs[i], fnr=fnrs[i], estimator_name=names[i]).plot(ax=ax)
    
    ax.legend(loc='best')

    plt.savefig(f'{path}.eps', dpi=300, bbox_inches='tight')
    plt.savefig(f'{path}.png', dpi=300, bbox_inches='tight')

