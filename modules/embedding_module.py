
import torch
from torch import nn

class EmbeddingModule(nn.Module):
    '''
    Embedding module
    '''

    def __init__(self, input_dim, memory_dim, device):
        '''
        Initialization of the module

        Arguments
        ---------
            input_dim : int
                Dimensionality of input features
            memory_dim : int
                Dimensionality of node embeddings stored in memory
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

        hidden_dim = (input_dim + memory_dim) // 2
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )

    def forward(self, flow_features):
        '''
        Forward pass of the module

        Arguments
        ---------
            flow_features : torch.Tensor [batch_size, input_dim]
                Features corresponding to the observed flows

        Returns
        -------
            flow_embeddings : torch.Tensor [batch_size, memory_dim]
                Embeddings of the observed flow features
        '''

        # compute embedding for current flows
        flow_embeddings = self.projection(flow_features)

        return flow_embeddings

