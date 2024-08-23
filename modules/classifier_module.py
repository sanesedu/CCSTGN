
import torch
from torch import nn

class ClassifierModule(nn.Module):
    '''
    Classifier module
    '''

    def __init__(self, input_dim, memory_dim, n_classes, device):
        '''
        Initialization of the module

        Arguments
        ---------
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

        self.memory_dim = memory_dim
        self.device = device

        hidden_dim = (2*memory_dim + n_classes) // 2
        self.classifier = nn.Sequential(
            nn.Linear(2*memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, flow_embeddings, predictions):
        '''
        Forward pass of the module

        Arguments
        ---------
            flow_embeddings : torch.Tensor [batch_size, memory_dim]
                ID of the nodes corresponding to the observed flows
            predictions : torch.Tensor [batch_size, memory_dim]
                Timestamps corresponding to the observed flows

        Returns
        -------
            classifications : torch.Tensor [batch_size, n_classes]
                Probabilities associated to each class for each observed flow
        '''

        # compute similarity scores between predicted and observed_flows
        classifications = self.classifier(torch.cat([flow_embeddings, predictions], dim=1))

        return classifications

