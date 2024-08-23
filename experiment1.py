import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from time import time, strftime
import pickle
from pathlib import Path
import logging
import sys
import os
import matplotlib.pyplot as plt

from sklearn.metrics import det_curve, roc_auc_score, f1_score, confusion_matrix, average_precision_score, ConfusionMatrixDisplay

from baselines.mlp import MLP
from baselines.mlr import MLR
from model.ccstgn import CCSTGN
from data_processing.flow_dataset import FlowDataset
from visualization.metrics import comparison_plot, evolution_plots

from modules.classifier_module import ClassifierModule

def reset_random_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)

def train(device, training_mode, model_name, model, dataloader, optimizer, loss_criterion, n_classes, classifier_head=None):

    train_loss = 0

    if model_name == 'ccstgn':
        if training_mode == 'probing':
            model.eval()
        else:
            model.train()

        if classifier_head is not None:
            classifier_head.train()

    for i, (node_ids, timestamps, flow_features, labels) in enumerate(tqdm(dataloader)):

        node_ids, timestamps, flow_features, labels = node_ids.to(device), timestamps.to(device), flow_features.to(device), labels.flatten().to(device)
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=n_classes).float()

        optimizer.zero_grad()

        if model_name == 'ccstgn':
            if training_mode == 'probing':
                with torch.no_grad():
                    flow_embeddings, memory_embeddings = model(node_ids, timestamps, flow_features)
            else:
                flow_embeddings, memory_embeddings = model(node_ids, timestamps, flow_features)

            if classifier_head is not None:
                logits = classifier_head(flow_embeddings, memory_embeddings)
                loss = loss_criterion(logits, one_hot_labels)
            else:
                # target=1 to maximize the cosine similarity of the two inputs
                loss = loss_criterion(flow_embeddings, memory_embeddings, torch.ones(node_ids.shape[0]).to(device))
        else:
            logits = model(flow_features)
            loss = loss_criterion(logits, one_hot_labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        if model_name == 'ccstgn' and training_mode != 'probing':
            model.detach_memory()

    avg_train_loss = train_loss / (i + 1)

    return avg_train_loss

def test(device, model_name, model, dataloader, loss_criterion, n_classes, lambda_, classifier_head=None, thr=None):

    test_loss = 0

    probs_positive = []
    ground_truths = []

    with torch.no_grad():
        model.eval()
        if classifier_head is not None:
            classifier_head.eval()
        for i, (node_ids, timestamps, flow_features, labels) in enumerate(tqdm(dataloader)):

            node_ids, timestamps, flow_features, labels = node_ids.to(device), timestamps.to(device), flow_features.to(device), labels.flatten().to(device)
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=n_classes).float()

            if model_name == 'ccstgn':
                flow_embeddings, memory_embeddings = model(node_ids, timestamps, flow_features)

                if classifier_head is not None:
                    logits = classifier_head(flow_embeddings, memory_embeddings)
                    loss = loss_criterion(logits, one_hot_labels)
                else:
                    # target=1 to maximize the cosine similarity of the two inputs
                    loss = loss_criterion(flow_embeddings, memory_embeddings, torch.ones(node_ids.shape[0]).to(device))
            else:
                logits = model(flow_features)
                loss = loss_criterion(logits, one_hot_labels)

            test_loss += loss.item()

            if model_name != 'ccstgn' or classifier_head is not None:
                probs = torch.nn.functional.softmax(logits, dim=1)
            else:
                cosine_sim = torch.nn.functional.cosine_similarity(flow_embeddings, memory_embeddings)
                probs = torch.zeros(node_ids.shape[0], 2)
                probs[:, 0] = cosine_sim
                probs[:, 1] = -cosine_sim
                probs = torch.nn.functional.softmax(probs, dim=1)

            prob_positive = probs[:, 1] # for binary case

            probs_positive.extend(prob_positive.tolist())
            ground_truths.extend(labels.tolist())

    avg_test_loss = test_loss / (i + 1)
    auc = roc_auc_score(ground_truths, probs_positive) * 100.

    if thr is not None:
        predictions = [1 if probs_positive[k] >= thr else 0 for k in range(len(probs_positive))]

        f1 = f1_score(ground_truths, predictions) * 100.
        conf_mat = confusion_matrix(ground_truths, predictions, normalize='true')

        # since the matrix is normalized by true label
        # these computations are correct because they operate over the rows
        tn, fp, fn, tp = conf_mat.ravel()

        fpr = (fp / (fp + tn)) * 100.
        fnr = (fn / (fn + tp)) * 100.

    else:
        fprs, fnrs, thrs = det_curve(ground_truths, probs_positive)

        min_val = float('inf')
        min_idx = -1

        for idx in range(len(thrs)):
            val = (1 - lambda_) * fprs[idx] + lambda_ * fnrs[idx]

            if val < min_val:
                min_val = val
                min_idx = idx

        fpr = fprs[min_idx] * 100.
        fnr = fnrs[min_idx] * 100.
        thr = thrs[min_idx]

        predictions = [1 if probs_positive[k] >= thr else 0 for k in range(len(probs_positive))]

        f1 = f1_score(ground_truths, predictions) * 100.
        conf_mat = confusion_matrix(ground_truths, predictions, normalize='true')

    return avg_test_loss, auc, f1, fpr, fnr, thr, conf_mat

if __name__ == '__main__':

    # argument parser configuration
    parser = argparse.ArgumentParser('Supervised training interface')
    parser.add_argument('--dataset', type=str, help='Dataset name (e.g. NF-ToN-IoT, NF-UNSW-NB15-v2)', default='NF-UNSW-NB15-v2')
    parser.add_argument('--bs', type=int, default=7500, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate [0,1]')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay [0,1]')
    parser.add_argument('--lambda_', type=float, default=0.75, help='Bias for FPR or FNR [0,1]')
    parser.add_argument('--n_epochs', type=int, default=4, help='Number of epochs')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--memory_dim', type=int, default=40, help='Dimensionality of node representation in memory')
    parser.add_argument('--n_neighbors', type=int, default=5, help='Number of neighbors to consider in message-passing')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers for GNN')
    parser.add_argument('--task', type=str, help='Classification task', default='binary', choices=['binary', 'multi'])
    parser.add_argument('--model', type=str, help='Model', default='ccstgn', choices=['ccstgn', 'mlp', 'mlr'])
    parser.add_argument('--agg_function', type=str, help='Aggregation function', default='softmax', choices=['mean', 'softmax', 'cosine'])
    parser.add_argument('--variant', type=str, help='Variant of CCSTGN', default='rs', choices=['rs', 'iso'])
    parser.add_argument('--experiment', type=str, help='Type of experiment', default='tuning', choices=['preliminary', 'tuning', 'evaluation', 'validation'])
    parser.add_argument('--tuning_key', type=str, help='Key of parameter to tune', default='bs', choices=['bs', 'lr', 'wd', 'lambda_', 'n_epochs', 'memory_dim', 'n_neighbors', 'n_layers', 'agg_function'])
    parser.add_argument('--tuning_values', nargs='+', help='Values to explore for parameter tuned')
    parser.add_argument('--val_thr', type=float, default=None, help='Threshold value for testing obtained in validation')
    parser.add_argument('--training', type=str, help='Type of training', default='supervised', choices=['supervised', 'self-supervised', 'probing', 'finetuning'])
    parser.add_argument('--transfer_path', type=str, default='', help='Path to model to use for transfer learning.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Cast tuning values to appropriate type
    if args.experiment == 'tuning':
        val = args.tuning_values[0]
        if val.isnumeric():
            tuning_values = list(map(int, args.tuning_values))
        elif val.replace('.', '').isnumeric():
            tuning_values = list(map(float, args.tuning_values))
        else:
            tuning_values = args.tuning_values

    # specify and create paths for the inputs and outputs
    if args.experiment == 'evaluation':
        train_data_path = f'datasets/{args.dataset}/data/{args.dataset}_trainval.csv'
        test_data_path = f'datasets/{args.dataset}/data/{args.dataset}_test.csv'
    else:
        train_data_path = f'datasets/{args.dataset}/data/{args.dataset}_train.csv'
        val_data_path = f'datasets/{args.dataset}/data/{args.dataset}_val.csv'

    mapping_path = f'datasets/{args.dataset}/data/{args.dataset}_mapping.pkl'

    current_time = strftime("%Y%m%d-%H%M%S")

    if args.model == 'ccstgn':
        model_name = f'{args.model.upper()}-{args.variant}'

        if args.training == 'self-supervised':
            model_name += '(ssl)'
        elif args.training == 'finetuning':
            model_name += '(ft)'
        elif args.training == 'probing':
            model_name += '(pb)'
    else:
        model_name = f'{args.model.upper()}'

    if args.experiment == 'tuning':
        filename = f'{args.tuning_key}_{current_time}'
    else:
        filename = f'{current_time}'

    SAVE_PATH = f'./experiments/{args.experiment}_{model_name}_{args.dataset}/{filename}'
    FIGURES_PATH = f'{SAVE_PATH}/figures'
    MODELS_PATH = f'{SAVE_PATH}/models'
    RESULTS_PATH = f'{SAVE_PATH}/results'

    Path(SAVE_PATH).mkdir(parents=True)
    Path(FIGURES_PATH).mkdir(parents=True)
    Path(MODELS_PATH).mkdir(parents=True)
    Path(RESULTS_PATH).mkdir(parents=True)

    # logging configuration
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%d/%b/%Y %H:%M:%S",)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # send debugging-level messages to disk
    fh = logging.FileHandler(f'{SAVE_PATH}/output.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # output warning-level messages to stdin
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    logger.addHandler(ch)

    # display selected args values
    logger.info(args)

    # access remaining input arguments for model
    with open(mapping_path, 'rb') as f:
        id_mapping = pickle.load(f)

    logger.info("Mapping loaded")

    # access info data
    info_data_path = f'datasets/{args.dataset}/data/{args.dataset}_info.csv'
    info_data = pd.read_csv(info_data_path)

    n_nodes = int(info_data['n_nodes'][0])
    logger.info(f'Number of nodes = {n_nodes}')

    input_dim = int(info_data['feat_dim'][0])
    logger.info(f'Input features dimensions = {input_dim}')

    if args.task == 'binary':
        n_classes = 2
        class_counts = torch.tensor(info_data['bin_counts'][:n_classes].values.tolist())
        class_names = info_data['bin_labels'][:n_classes].values.tolist()
    else:
        n_classes = int(info_data['n_attacks'][0])
        class_counts = torch.tensor(info_data['multi_counts'][:n_classes].values.tolist())
        class_names = info_data['multi_labels'][:n_classes].values.tolist()

    logger.info(f'Number of classes = {n_classes}')

    # compute class weights according to the 'balanced' compute_class_weight formula in sklearn
    n_samples = info_data['train_flows'][0]
    class_weights = n_samples / (n_classes * class_counts)
    class_weights = class_weights.flatten()

    logger.info(f'Class weights = {class_weights.tolist()}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')

    class_weights = class_weights.float().to(device)

    logger.info(f'Saving results to {SAVE_PATH}')

    # Prepare results dictionary and tuning parameters
    results_columns = ['Loss', 'AUC', 'F1', 'FPR', 'FNR', 'Threshold', 'Time']

    if args.experiment == 'tuning':

        n = len(tuning_values)

        param_grid = {
            'bs': [args.bs] * n,
            'lr': [args.lr] * n,
            'wd': [args.wd] * n,
            'lambda_': [args.lambda_] * n,
            'n_epochs': [args.n_epochs] * n,
            'memory_dim': [args.memory_dim] * n,
            'n_neighbors': [args.n_neighbors] * n,
            'n_layers': [args.n_layers] * n,
            'agg_function': [args.agg_function] * n
        }

        param_grid[args.tuning_key] = tuning_values

        results_columns.insert(0, args.tuning_key)

    else:
        n = args.n_runs

    # Results dictionary
    results = pd.DataFrame(columns=results_columns)

    # set random seed
    reset_random_seed(args.seed)

    # access parameters
    bs = args.bs
    lr = args.lr
    wd = args.wd
    lambda_ = args.lambda_
    n_epochs = args.n_epochs
    memory_dim = args.memory_dim
    n_neighbors = args.n_neighbors
    n_layers = args.n_layers
    agg_function = args.agg_function

    for r in range(n):

        if args.experiment == 'tuning':
            # reset random seeds when tuning for fair comparison
            reset_random_seed(args.seed)

            # access parameters
            bs = param_grid['bs'][r]
            lr = param_grid['lr'][r]
            wd = param_grid['wd'][r]
            lambda_ = param_grid['lambda_'][r]
            n_epochs = param_grid['n_epochs'][r]
            memory_dim = param_grid['memory_dim'][r]
            n_neighbors = param_grid['n_neighbors'][r]
            n_layers = param_grid['n_layers'][r]
            agg_function = param_grid['agg_function'][r]

            # Prepare results dictionary for this run
            run_results_columns = ['Train loss', 'Validation loss', 'AUC', 'F1', 'FPR', 'FNR', 'Threshold', 'Time']
            run_results = pd.DataFrame(columns=run_results_columns)

        dataloader_params = {
            'batch_size': bs,
            'shuffle': False,
            'num_workers': 6,
            'pin_memory': True
        }

        logger.info(f'Starting run {r}')

        train_data = FlowDataset(train_data_path, args.task)
        train_loader = DataLoader(train_data, **dataloader_params)

        if args.experiment == 'evaluation':
            test_data = FlowDataset(test_data_path, args.task)
            test_loader = DataLoader(test_data, **dataloader_params)
        else:
            val_data = FlowDataset(val_data_path, args.task)
            val_loader = DataLoader(val_data, **dataloader_params)

        logger.info('Dataset loaded')

        if args.model == 'ccstgn':
            model = CCSTGN(args.variant, agg_function, n_nodes, input_dim, memory_dim, n_classes, id_mapping, n_neighbors, n_layers, device)
            model.to(device)

            if args.training == 'supervised':
                classifier_head = ClassifierModule(input_dim, memory_dim, n_classes, device)
                classifier_head.to(device)
                optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier_head.parameters()), lr=lr, weight_decay=wd)
            elif args.training == 'self-supervised':
                classifier_head = None
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            else:
                # load pretrained ccstgn model
                state_dict = torch.load(args.transfer_path)
                model.load_state_dict(state_dict)

                params = list(model.parameters())

                if args.training == 'probing':

                    # freeze parameters of pretrained ccstgn
                    for param in model.parameters():
                        param.requires_grad = False

                    params = []

                classifier_head = ClassifierModule(input_dim, memory_dim, n_classes, device)
                classifier_head.to(device)
                optimizer = torch.optim.Adam(params + list(classifier_head.parameters()), lr=lr, weight_decay=wd)
        else:
            if args.model == 'mlp':
                model = MLP(input_dim, memory_dim, n_classes, device)
            elif args.model == 'mlr':
                model = MLR(input_dim, n_classes, device)

            classifier_head = None
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        if args.training == 'self-supervised':
            loss_criterion = torch.nn.CosineEmbeddingLoss()
        else:
            loss_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        initial_time = time()

        for ep in range(n_epochs):

            # Training
            logger.info(f'Run {r} - Epoch {ep} -> Start of training')
            ep_start_time = time()

            if args.model == 'ccstgn':
                logger.info(f'Initialize memory')
                model.memory_module.init_memory()

            train_loss = train(device, args.training, args.model, model, train_loader, optimizer, loss_criterion, n_classes, classifier_head=classifier_head)

            train_runtime = (time() - ep_start_time)/60.
            logger.info(f'Run {r} - Epoch {ep} - Training -> time: {train_runtime:.2f} min, loss: {train_loss}')

            if args.experiment == 'tuning':

                # Validation
                logger.info(f'Run {r} - Epoch {ep} -> Start of validation')
                start_time = time()

                loss, auc, f1, fpr, fnr, thr, conf_mat = test(device, args.model, model, val_loader, loss_criterion, n_classes, lambda_, classifier_head=classifier_head)

                val_runtime = (time() - start_time)/60.
                logger.info(f'Run {r} - Epoch {ep} - Validation -> time: {val_runtime:.2f} min, loss: {loss}, AUC: {auc}, F1: {f1}, FPR: {fpr}, FNR: {fnr}, thr: {thr}, Confusion matrix: {conf_mat.tolist()}')

                ep_runtime = (time() - ep_start_time)/60.

                run_results.loc[ep] = [train_loss, loss, auc, f1, fpr, fnr, thr, ep_runtime]

            ep_runtime = (time() - ep_start_time)/60.

        # Disable logging debug messages
        logging.getLogger('matplotlib').setLevel(logging.INFO)

        # Save and plot evolution of run if tuning
        if args.experiment == 'tuning':
            run_results.to_csv(f'{RESULTS_PATH}/run{r}_results.csv', index=False)
            logger.info(f'Run {r} results saved')

            evolution_plots(run_results, path=FIGURES_PATH, title=f'{model_name} | Run {r}', filename=f'run{r}')
            logger.info(f'Run {r} results plotted')
        elif args.experiment == 'evaluation':
            # Testing
            logger.info(f'Run {r} -> Start of testing')
            start_time = time()

            loss, auc, f1, fpr, fnr, thr, conf_mat = test(device, args.model, model, test_loader, loss_criterion, n_classes, lambda_, classifier_head=classifier_head, thr=args.val_thr)

            test_runtime = (time() - start_time)/60.

            logger.info(f'Run {r} - Testing -> time: {test_runtime:.2f} min, loss: {loss}, AUC: {auc}, F1: {f1}, FPR: {fpr}, FNR: {fnr}, thr: {thr}, Confusion matrix: {conf_mat.tolist()}')
        elif args.experiment == 'validation':
            # Testing
            logger.info(f'Run {r} -> Start of validation')
            start_time = time()

            loss, auc, f1, fpr, fnr, thr, conf_mat = test(device, args.model, model, val_loader, loss_criterion, n_classes, lambda_, classifier_head=classifier_head)

            test_runtime = (time() - start_time)/60.

            logger.info(f'Run {r} - Validation -> time: {test_runtime:.2f} min, loss: {loss}, AUC: {auc}, F1: {f1}, FPR: {fpr}, FNR: {fnr}, thr: {thr}, Confusion matrix: {conf_mat.tolist()}')

        total_runtime = (time() - initial_time)/60.
        logger.info(f'Total runtime: {total_runtime}')

        # Add results
        run_results = [loss, auc, f1, fpr, fnr, thr, total_runtime]
        if args.experiment == 'tuning':
            run_results.insert(0, param_grid[args.tuning_key][r])

        results.loc[r] = run_results

        logger.info(f'Results for run {r} stored')

        # Save confusion matrices
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=class_names)
        disp.plot(cmap='Blues', values_format='.5g')
        plt.savefig(f'{FIGURES_PATH}/confusion_matrix_{r}.eps', dpi=300, bbox_inches='tight')
        plt.savefig(f'{FIGURES_PATH}/confusion_matrix_{r}.png', dpi=300, bbox_inches='tight')

        logger.info(f'Figures for run {r} saved')

        # Save model
        torch.save(model.state_dict(), f'{MODELS_PATH}/model_{r}.pt')

        logger.info(f'Model for run {r} saved')

    results.to_csv(f'{RESULTS_PATH}/results.csv', index=False)
    logger.info(f'Results saved')

    # Generate and save result visualization plots
    if args.experiment == 'tuning':
        comparison_plot(results, key=args.tuning_key, path=FIGURES_PATH, title=model_name)
    else:
        results.reset_index(names='Run', inplace=True)
        comparison_plot(results, key='Run', path=FIGURES_PATH, title=model_name)

    logger.info(f'Results plotted')

