import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import pickle
import logging
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import category_encoders as ce

IN_PATH = ''
TRAIN_PATH = ''
VAL_PATH = ''
TRAINVAL_PATH = ''
TEST_PATH = ''
INV_MAP_PATH = ''
MAP_PATH = ''
INFO_PATH = ''
LABEL_PATH = ''
LOGGING_PATH = ''

def construct_mapping(data, filename, logger):

    mapping = []
    inv_mapping = {}
    new_ids = np.zeros(len(data))

    id_count = 0

    for idx in tqdm(range(data.shape[0])):

        # sorted -> channel instead of directed channel
        ips = set(sorted(data.iloc[idx, :].tolist()))
        ip_index = '-'.join(ips)

        # if ID has not been seen yet
        if ip_index not in inv_mapping.keys():

            # add corresponding mappings
            mapping.append(ips)
            inv_mapping[ip_index] = id_count

            # increase id count
            id_count += 1

        # new id corresponds to the id associated to this ChannelID
        new_ids[idx] = inv_mapping[ip_index]

    logger.info(f'Number of channels: {id_count}')

    with open(filename, 'wb') as f:
        pickle.dump(mapping, f)

    logger.info(f'Mapping saved')

    return new_ids

def mapping(dataset, format_name, logger):

    # load dataset
    if format_name == 'nfv2':
        cols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']
    elif format_name == 'cic':
        cols = ['Src IP', 'Dst IP']
    data = pd.read_csv(IN_PATH, usecols=cols)

    logger.info('Dataset loaded')

    #data.dropna(inplace=True)

    # Encode Channel IDs and construct graph as adjacency list
    new_channel_ids = construct_mapping(data, MAP_PATH, logger)
    data.insert(loc=0, column='CHANNEL_ID', value=new_channel_ids.astype(int))

    # Drop unnecessary columns
    data.drop(columns=cols, inplace=True)
    logger.info('Inverse mapping generated')

    data.to_csv(INV_MAP_PATH, index=False)
    logger.info('Inverse mapping saved')

def prepare(dataset, format_name, logger):

    # load dataset
    data = pd.read_csv(IN_PATH)
    logger.info('Dataset loaded')

    # Combine source and destination IPs to define directect communication channel
    if format_name == 'nfv2':
        cols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']
    elif format_name == 'cic':
        cols = ['Flow ID', 'Src IP', 'Dst IP']

    # Drop unnecessary columns
    data.drop(columns=cols, inplace=True)

    # Add new channel IDs
    inv_mapping = pd.read_csv(INV_MAP_PATH)
    data = inv_mapping.join(data)

    logger.info('Channel IDs loaded')

    # Generate timestamps
    if format_name == 'nfv2':
        # Generate 'estimated timestamp' from sequential order
        data.reset_index(names='Timestamp', inplace=True)
    elif format_name == 'cic':
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %I:%M:%S %p')
        data['Timestamp'] = data['Timestamp'].apply(lambda x: x.value) / 10**9 # convert to seconds
        data = data.sort_values('Timestamp')
        data.reset_index(drop=True, inplace=True)
        data['Timestamp'] = data['Timestamp'].apply(lambda x: x - data['Timestamp'][0]) # shift beginning to 0
    logger.info('Timestamps generated')

    # deal with inf values in column Flow Pkts/s
    # problem is that Flow Duration = 0 and Flow Byts/s = 0.0
    # then this value should be 0.0 as well
    if format_name == 'cic':
        inf_indices = np.where(np.isinf(data.loc[:, 'Flow Pkts/s']))[0]
        data.iloc[inf_indices, 19] = 0.0
        logger.info('Infs removed')

    # Split data into train (70%), validation (10%), and test (20%) sets without shuffling
    trainval_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    train_data, val_data = train_test_split(trainval_data, test_size=1/8, shuffle=False)
    logger.info('70-10-20 split generated')

    # store preprocessed train and test data
    train_data.to_csv(TRAIN_PATH, index=False)
    logger.info('Train data saved')

    val_data.to_csv(VAL_PATH, index=False)
    logger.info('Validation data saved')

    test_data.to_csv(TEST_PATH, index=False)
    logger.info('Test data saved')

def encode(dataset, format_name, logger):

    # load dataset
    label_col = 'Attack'
    label_data = pd.read_csv(IN_PATH, usecols=[label_col])
    train_data = pd.read_csv(TRAIN_PATH)
    val_data = pd.read_csv(VAL_PATH)
    test_data = pd.read_csv(TEST_PATH)
    logger.info('Datasets loaded')

    pd.set_option('future.no_silent_downcasting', True)

    # encode label column

    # assign value to each label depending on their frequency in dataset
    # lower value -> higher frequency
    label_values = label_data[label_col].value_counts().to_dict() # assume all known a priori

    for i, k in enumerate(label_values.keys()):
        label_values[k] = i

    train_data.replace({f'{label_col}': label_values}, inplace=True)
    val_data.replace({f'{label_col}': label_values}, inplace=True)
    test_data.replace({f'{label_col}': label_values}, inplace=True)

    logger.info('Multi-class label encoded')

    def binary_encode(col, n_bits):

        nonlocal train_data, val_data, test_data

        col = [col]
        possible_values = list(range(0, 2**n_bits))
        possible_df = pd.DataFrame(possible_values, columns=col)

        encoder = ce.BinaryEncoder(cols=col)
        encoder.fit(possible_df)

        new_data = encoder.transform(train_data[col]).astype('boolean')
        new_cols = new_data.columns.tolist()

        train_data = train_data.join(new_data).drop(columns=col)
        val_data = val_data.join(encoder.transform(val_data[col]).astype('boolean')).drop(columns=col)
        test_data = test_data.join(encoder.transform(test_data[col]).astype('boolean')).drop(columns=col)

        return new_cols

    # encode categorical columns
    if format_name == 'nfv2':
        cat_cols = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS', 'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'FTP_COMMAND_RET_CODE']
        n_bits = [16, 16, 8, 8, 9, 8, 8, 16, 8, 16, 16, 10]
    elif format_name == 'cic':
        cat_cols = ['Src Port', 'Dst Port', 'Protocol']
        n_bits = [16, 16, 8]

    new_cols = []

    for i in tqdm(range(len(cat_cols))):
        n_col = binary_encode(cat_cols[i], n_bits[i])
        new_cols.extend(n_col)

    logger.info('Binarized features encoded')

    # scale features to have 0 mean and unit variance
    non_feature_cols = ['Timestamp', 'CHANNEL_ID', 'Label', 'Attack']
    non_scale_cols = non_feature_cols + new_cols

    cols_to_norm = train_data.drop(columns=non_scale_cols).columns
    scaler = StandardScaler()

    train_data[cols_to_norm] = scaler.fit_transform(train_data[cols_to_norm])
    test_data[cols_to_norm] = scaler.transform(test_data[cols_to_norm])
    val_data[cols_to_norm] = scaler.transform(val_data[cols_to_norm])

    logger.info('Feature columns standardized')

    # create trainval data
    trainval_data = pd.concat([train_data, val_data])
    logger.info('Trainval data created')

    # reset indices of dataframe
    val_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    logger.info('Validation and tests set indices reset')

    # compute info
    info_data = pd.DataFrame(columns=['n_nodes', 'feat_dim', 'n_attacks', 'train_flows', 'bin_labels', 'bin_counts', 'multi_labels', 'multi_counts'])

    # use same order of labels as in the encoding of whole dataset
    info_data['multi_labels'] = list(label_values.keys())
    multi_counts = [(train_data[label_col] == t).sum() for t in list(label_values.values())]
    info_data['multi_counts'] = multi_counts
    info_data.loc[0, 'n_attacks'] = len(multi_counts)

    info_data.loc[[0,1], 'bin_labels'] = ['Benign', 'Attack']
    bin_counts = [(train_data['Label'] == t).sum() for t in [0,1]]
    info_data.loc[[0,1], 'bin_counts'] = bin_counts

    # assume we know number of allowed devices in network, else compute from train + estimate of test/val
    info_data.loc[0, 'n_nodes'] = len(np.unique(pd.read_csv(INV_MAP_PATH).values))

    info_data.loc[0, 'feat_dim'] = len(train_data.drop(columns=non_feature_cols).columns)
    info_data.loc[0, 'train_flows'] = train_data.shape[0]
    logger.info('Info generated')

    # store preprocessed train and test data
    train_data.to_csv(TRAIN_PATH, index=False)
    logger.info('Train data saved')

    trainval_data.to_csv(TRAINVAL_PATH, index=False)
    logger.info('Trainval data saved')

    val_data.to_csv(VAL_PATH, index=False)
    logger.info('Validation data saved')

    test_data.to_csv(TEST_PATH, index=False)
    logger.info('Test data saved')

    info_data.to_csv(INFO_PATH, index=False)
    logger.info('Info saved')

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Data preprocessing interface')
    parser.add_argument('--dataset', type=str, help='Dataset name (e.g. NF-ToN-IoT, NF-UNSW-NB15-v2)', default='NF-UNSW-NB15-v2')
    parser.add_argument('--task', type=str, choices=['mapping', 'prepare', 'encode'], help='Preprocessing task')

    args = parser.parse_args()

    IN_PATH = f'./datasets/{args.dataset}/data/{args.dataset}.csv'
    TRAIN_PATH = f'./datasets/{args.dataset}/data/{args.dataset}_train.csv'
    VAL_PATH = f'./datasets/{args.dataset}/data/{args.dataset}_val.csv'
    TRAINVAL_PATH = f'./datasets/{args.dataset}/data/{args.dataset}_trainval.csv'
    TEST_PATH = f'./datasets/{args.dataset}/data/{args.dataset}_test.csv'
    INV_MAP_PATH = f'./datasets/{args.dataset}/data/{args.dataset}_inv_mapping.csv'
    MAP_PATH = f'./datasets/{args.dataset}/data/{args.dataset}_mapping.pkl'
    INFO_PATH = f'./datasets/{args.dataset}/data/{args.dataset}_info.csv'
    LOGGING_PATH = f'./datasets/{args.dataset}/data/logs/'

    Path(LOGGING_PATH).mkdir(parents=True, exist_ok=True)

    # logging configuration
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%d/%b/%Y %H:%M:%S",)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # send debugging-level messages to disk
    fh = logging.FileHandler(f'{LOGGING_PATH}/{args.task}.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # output warning-level messages to stdin
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    logger.addHandler(ch)

    # display selected args values
    logger.info(args)

    if args.dataset[:2] == 'NF' and args.dataset[-2:] == 'v2':
        format_name = 'nfv2'
    elif args.dataset[:3] == 'CIC':
        format_name = 'cic'

    if args.task == 'mapping':
        mapping(args.dataset, format_name, logger)
    elif args.task == 'prepare':
        prepare(args.dataset, format_name, logger)
    elif args.task == 'encode':
        encode(args.dataset, format_name, logger)

