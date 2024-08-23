
import torch
from torch import nn

class ISO_MLP(nn.Module):
    '''
    MLP for GIN-inspired predictions
    '''

    def __init__(self, memory_dim, device):
        '''
        Initialization of the module

        Arguments
        ---------
            memory_dim : int
                Dimensionality of node embeddings stored in memory
            device : 'cuda' | 'cpu'
                Device for the computations

        Returns
        -------
            None
        '''

        super().__init__()

        self.memory_dim = memory_dim
        self.device = device

        self.mlp = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim)
        )

    def forward(self, neighborhood_sum):
        '''
        Forward pass of the module

        Arguments
        ---------
            neighborhood_sum : torch.Tensor [batch_size, memory_dim]
                Embeddings resulting after summing over the sampled spatiotemporal neighborhoods

        Returns
        -------
            predictions : torch.Tensor [batch_size, memory_dim]
                Predictions after applying 2-layer MLP to inputs
        '''

        predictions = self.mlp(neighborhood_sum)

        return predictions

