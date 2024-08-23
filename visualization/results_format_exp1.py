
import datetime
import argparse
import numpy as np
import pandas as pd
import sys
import os
import re


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
            
            try:
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
                total_times = df.iloc[:, 6].values
                testing_times = []
                
                with open(f'{path}/{item}/output.log') as f:
                    for line in f:
                        if "Testing" in line:
                            testing_times.append(float(line.split()[6]))
                
                training_times = [total_times[i] - testing_times[i] for i in range(len(total_times))]

                print(total_times)
                print(testing_times)
                print(training_times)

                tr_m, _ = get_mean_std(training_times)
                tr_time_string = time_format(tr_m)
                
                te_m, _ = get_mean_std(testing_times)
                te_time_string = time_format(te_m)

                res += f'${tr_time_string}$ & ${te_time_string}$\\\\'

                print(res)
                print()

            except:
                print("wrong format")

if __name__ == '__main__':

    # argument parser configuration
    parser = argparse.ArgumentParser('Supervised training interface')
    parser.add_argument('--task', type=str, help='Name of task', default='tuning', choices=['tuning', 'evaluation'])
    parser.add_argument('--param', type=str, help='Name of parameter')
    parser.add_argument('--model', type=str, help='Name of the model')
    parser.add_argument('--dataset', type=str, help='Name of the dataset', default='NF-UNSW-NB15-v2')
    parser.add_argument('--lambda_', type=float, default=0.75, help='Bias for FPR or FNR [0,1]')
    parser.add_argument('--fix', help='Convert FPR and FNR to percentage to fix missing factor on certain evaluation results', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.task == 'tuning':
        tuning(args.param, args.dataset, args.lambda_)
    elif args.task == 'evaluation':
        evaluation(args.model, args.dataset, args.lambda_, args.fix)

