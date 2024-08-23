
import torch
from torch.utils.data import Dataset
import pandas as pd

class FlowDataset(Dataset):

    def __init__(self, file, task):

        self.flows = pd.read_csv(file)

        self.id_column = 'CHANNEL_ID'

        self.timestamp_column = 'Timestamp'

        if task == 'binary':
            self.label_column = 'Label'
            self.flows.drop(columns=['Attack'], axis=1, inplace=True)
        else:
            self.label_column = 'Attack'
            self.flows.drop(columns=['Label'], axis=1, inplace=True)

    def __len__(self):
        return self.flows.shape[0]

    def __getitem__(self, idx):

        data = self.flows.iloc[[idx]]

        node_ids = torch.tensor(data[self.id_column].values)
        timestamps = torch.tensor(data[self.timestamp_column].values, dtype=torch.float32)
        labels = torch.tensor(data[self.label_column].values)
        features = torch.tensor(data.drop(columns=[self.id_column, self.timestamp_column, self.label_column], axis=1).values.astype(float), dtype=torch.float32).flatten()

        return node_ids, timestamps, features, labels

