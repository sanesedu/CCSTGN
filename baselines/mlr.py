
import torch
from torch import nn

class MLR(nn.Module):
    '''
    Multinomial Logistic Regression
    '''

    def __init__(self, input_dim, n_classes, device):
        '''
        Initialization of the model

        Arguments
        ---------
            input_dim : int
                Dimensionality of input features
            n_classes : int
                Number of classes for classification
            device : 'cuda' | 'cpu'
                Device for the computations

        Returns
        -------
            None
        '''

        super().__init__()

        self.device = device

        self.linear = nn.Linear(input_dim, n_classes)

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
        '''

        logits = self.linear(flow_features)

        return logits

