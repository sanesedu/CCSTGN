
import torch
from torch import nn

class MLP(nn.Module):
    '''
    Multi-layer Perceptron
    '''

    def __init__(self, input_dim, memory_dim, n_classes, device):
        '''
        Initialization of the model

        Arguments
        ---------
            input_dim : int
                Dimensionality of input features
            memory_dim : int
                Dimensionality of node embeddings stored in memory
            n_classes : int
                Number of classes for classification
            device : 'cuda' | 'cpu'
                Device for the computations

        Returns
        -------
            None
        '''

        super().__init__()

        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.device = device

        hidden_dim1 = (input_dim + memory_dim) // 2
        hidden_dim2 = (memory_dim + n_classes) // 2
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, n_classes),
        )

    def forward(self, flow_features):
        '''
        Forward pass of the model

        Arguments
        ---------
            flow_features : torch.Tensor [batch_size, input_dim]
                Features corresponding to the observed flows

        Returns
        -------
            logits : torch.Tensor [batch_size, n_classes]
                Logits associated to each class for each observed flow
            probabilities : torch.Tensor [batch_size, n_classes]
                Probabilities associated to each class for each observed flow
        '''

        logits = self.classifier(flow_features)

        return logits

