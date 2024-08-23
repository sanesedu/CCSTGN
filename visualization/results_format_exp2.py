import pickle
import datetime
import argparse
import numpy as np
import pandas as pd
import sys
import os
import re
import ast
from metrics import det_plot

RESULTS_ORDER = ['MLR', 'MLP', 'CCSTGN-rs', 'CCSTGN-iso', 'CCSTGN-rs(ssl)', 'CCSTGN-iso(ssl)', 'CCSTGN-rs(ft)', 'CCSTGN-iso(ft)']

def time_format(minutes):
    total_seconds = int(minutes * 60)

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    time_string = f'{hours:02d}:{minutes:02d}:{seconds:02d}'

    return time_string

def tuning(param_name, dataset, lambda_):

    regex = re.compile(f'{param_name}*')

    variants = ['rs', 'iso']

    for variant in variants:
        rootdir = f'./experiments/tuning_CCTGN-{variant}_{dataset}'
        for root, dirs, files in os.walk(rootdir):
          for cdir in dirs:
            if regex.match(cdir):
                df = pd.read_csv(f'{rootdir}/{cdir}/results/results.csv')

                print(f'{variant} results\n')

                for row in range(df.shape[0]):

                    time_string = time_format(df.iloc[row, 7])

                    p = f'${df.iloc[row, 0]}$' if param_name == 'agg_function' else f'${df.iloc[row, 0]:.4g}$'

                    print(f'& {p} & ${df.iloc[row, 2]:.4f}$ & ${df.iloc[row, 3]:.4f}$ & ${df.iloc[row, 4]:.4f}$ & ${df.iloc[row, 5]:.4f}$ & ${time_string}$\\\\')

                print()

def get_mean_std(values):

    m = np.mean(values)
    v = np.std(values)

    return m, v

def evaluation(model, dataset, lambda_, fix):

    path = f'./experiments/evaluation_{model}_{dataset}'

    for item in os.listdir(path):

        if os.path.isdir(os.path.join(path, item)):

            print (f'Item: {item}\n')

            df = pd.read_csv(f'{path}/{item}/results/results.csv')

            cols = [1,2,3,4]

            res = f'{model} & '

            for c in cols:
                m, v = get_mean_std(df.iloc[:, c].values * 100 if fix and c > 2 else df.iloc[:, c].values)
                res += f'${m:.3f} \\scriptstyle{{\\pm {v:.3f}}}$ & '
            
            # f_omega
            f_omega = (1 - lambda_) * df.iloc[:, 3].values + lambda_ * df.iloc[:, 4].values 
            m, v = get_mean_std(f_omega * 100 if fix else f_omega)
            res += f'${m:.3f} \\scriptstyle{{\\pm {v:.3f}}}$ & '

            # time
            m, _ = get_mean_std(df.iloc[:, 6].values)
            time_string = time_format(m)

            res += f'${time_string}$ \\\\'

            print(res)
            print()

def get_val_results():
   
    print('\multirow{2}{*}{Method} & \multirow{2}{*}{AUC} & \\multicolumn{2}{|c|}{FPR $\\approx 0.01$} & \\multicolumn{2}{|c|}{FPR $\\approx 0.1$} & \\multicolumn{2}{|c|}{FPR $\\approx 1.0$} & \multirow{2}{*}{Time}\\\\')
    print('& & F1 & FNR & F1 & FNR & F1 & FNR & \\\\')
    print('\\hline')

    pattern = r'^validation_(.*?)_NF-UNSW-NB15-v2$' 
    regex = re.compile(pattern)

    rootdir = f'./experiments'
    
    for rdir in os.listdir(rootdir):
        full_path = os.path.join(rootdir, rdir)
        
        if os.path.isdir(full_path) and regex.match(rdir):
            model_name = regex.search(rdir).group(1)
            
            for edir in os.listdir(full_path):
                df = pd.read_csv(f'{rootdir}/{rdir}/{edir}/results/results.csv')
            
                # name
                res = f'{model_name} & '
                # auc
                res += f'${df.iloc[0, 1]:.3f}$ & '
                
                f1s = ast.literal_eval(df.iloc[0, 2]) 
                fprs = ast.literal_eval(df.iloc[0, 3]) 
                fnrs = ast.literal_eval(df.iloc[0, 4]) 
                
                for i in range(len(f1s)):
                    res += f'${f1s[i]:.3f}$ & ${fnrs[i]:.3f}$ & '
                    #res += f'${f1s[i]:.3f}$ & ${fprs[i]:.3f}$ & ${fnrs[i]:.3f}$ & '

                time_string = time_format(df.iloc[0, 6])
                res += f'${time_string}$'

                print(f'{res}\\\\')

def get_eval_results():
   
    print('\multirow{2}{*}{Method} & \\multicolumn{2}{|c|}{$\\text{FPR}_{val} \\approx 0.01$} & \\multicolumn{2}{|c|}{$\\text{FPR}_{val} \\approx 0.1$} & \\multicolumn{2}{|c}{$\\text{FPR}_{val} \\approx 1.0$} \\\\')
    print('& F1 $\\uparrow$ & FNR $\\downarrow$ & F1 $\\uparrow$ & FNR $\\downarrow$ & F1 $\\uparrow$ & FNR $\\downarrow$ \\\\')
    print('\\hline')

    pattern = r'^evaluation_(.*?)_NF-UNSW-NB15-v2$' 
    
    regex = re.compile(pattern)

    rootdir = f'./experiments'
   
    results = {}

    for rdir in os.listdir(rootdir):
        full_path = os.path.join(rootdir, rdir)
        
        if os.path.isdir(full_path) and regex.match(rdir):
            model_name = regex.search(rdir).group(1)
            
            for edir in os.listdir(full_path):
                df = pd.read_csv(f'{rootdir}/{rdir}/{edir}/results/results.csv')
            
                # name
                res = f'{model_name} '
                # auc
                #res += f'${df.iloc[:, 1]:.3f}$ & '
                
                f1s = np.zeros((3, df.shape[0]))
                fprs = np.zeros((3, df.shape[0]))
                fnrs = np.zeros((3, df.shape[0]))

                for row in range(df.shape[0]):
                    f1s_aux = ast.literal_eval(df.iloc[row, 2]) 
                    for i in range(len(f1s_aux)):
                        f1s[i][row] = f1s_aux[i]

                    fprs_aux = ast.literal_eval(df.iloc[row, 3]) 
                    for i in range(len(fprs_aux)):
                        fprs[i][row] = fprs_aux[i]
                    
                    fnrs_aux = ast.literal_eval(df.iloc[row, 4]) 
                    for i in range(len(fnrs_aux)):
                        fnrs[i][row] = fnrs_aux[i]
                    
                for i in range(len(f1s)):
                    f1m, f1v = get_mean_std(f1s[i])
                    fnrm, fnrv = get_mean_std(fnrs[i])
                    res += f'& ${f1m:.3f} \\scriptstyle{{\\pm {f1v:.3f}}}$ & ${fnrm:.3f} \\scriptstyle{{\\pm {fnrv:.3f}}}$ '
                    
                results[model_name] = f'{res}\\\\'
    
    for i, n in enumerate(RESULTS_ORDER):
        if i % 2 == 0:
            print('\\hline')
        print(results[n])

def det_visualization(res):
    
    fprs = []
    fnrs = []
    thrs = []
    idxs = []
    names = []
    
    if res == 'validation':
        pattern = r'^validation_(.*?)_NF-UNSW-NB15-v2$' 
    elif res == 'evaluation':
        pattern = r'^evaluation_(.*?)_NF-UNSW-NB15-v2$' 

    regex = re.compile(pattern)

    rootdir = f'./experiments'
    for rdir in os.listdir(rootdir):
        full_path = os.path.join(rootdir, rdir)
        
        if os.path.isdir(full_path) and regex.match(rdir):
            model_name = regex.search(rdir).group(1)
            names.append(model_name)             
            print(f'{model_name} - {rdir}')
            
            for edir in os.listdir(full_path):
                print(f'Loading pkl')
                
                if res == 'validation':
                    filename = f'{rootdir}/{rdir}/{edir}/results/det_curve_{model_name}.pkl'
                elif res == 'evaluation':
                    filename = f'{rootdir}/{rdir}/{edir}/results/det_curve_{model_name}_0.pkl'

                with open(filename, 'rb') as f:
                    if res == 'validation':
                        fprs_aux, fnrs_aux, thrs_aux, idxs_aux = pickle.load(f)
                    elif res == 'evaluation':
                        fprs_aux, fnrs_aux, thrs_aux = pickle.load(f)
                    
                fprs.append(fprs_aux)
                fnrs.append(fnrs_aux)
                thrs.append(thrs_aux)
                
                if res == 'validation':
                    idxs.append(idxs_aux)
                
    # sort the results if all computed
    if len(names) == len(RESULTS_ORDER):

        fprs_order = []
        fnrs_order = []
        thrs_order = []
        
        for n in RESULTS_ORDER:
            idx = names.index(n)
            fprs_order.append(fprs[idx])
            fnrs_order.append(fnrs[idx])
            thrs_order.append(thrs[idx])
        
        det_plot(fprs_order, fnrs_order, idxs, thrs_order, './det_curves', RESULTS_ORDER)
    else:        
        det_plot(fprs, fnrs, idxs, thrs, './det_curves', names)


if __name__ == '__main__':

    # argument parser configuration
    parser = argparse.ArgumentParser('Supervised training interface')
    parser.add_argument('--task', type=str, help='Name of task', default='tuning', choices=['tuning', 'evaluation', 'validation', 'det'])
    parser.add_argument('--param', type=str, help='Name of parameter')
    parser.add_argument('--model', type=str, help='Name of the model')
    parser.add_argument('--dataset', type=str, help='Name of the dataset', default='NF-UNSW-NB15-v2')
    parser.add_argument('--lambda_', type=float, default=0.75, help='Bias for FPR or FNR [0,1]')
    parser.add_argument('--fix', help='Convert FPR and FNR to percentage to fix missing factor on certain evaluation results', action=argparse.BooleanOptionalAction)
    parser.add_argument('--res', type=str, default='validation', choices=['validation', 'evaluation'], help='Name of results to consider')

    args = parser.parse_args()

    if args.task == 'tuning':
        tuning(args.param, args.dataset, args.lambda_)
    elif args.task == 'evaluation':
        get_eval_results()
    elif args.task == 'validation':
        get_val_results()
    elif args.task == 'det':
        det_visualization(args.res)
