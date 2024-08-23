
import torch
from torch import nn

from modules.memory_module import MemoryModule
from modules.embedding_module import EmbeddingModule

class CCSTGN(nn.Module):
    '''
    Channel-Centric Spatio-Temporal Graph Network
    '''

    def __init__(self, variant, agg_function, n_nodes, input_dim, memory_dim, n_classes, id_mapping, n_neighbors, n_layers, device):
        '''
        Initialization of the model

        Arguments
        ---------
            variant : 'rs' | 'iso'
                Variant of the CCTGN model
            n_nodes : int
                Maximum number of nodes allowed in memory
            input_dim : int
                Dimensionality of input features
            memory_dim : int
                Dimensionality of node embeddings stored in memory
            n_classes : int
                Number of classes for classification
            id_mapping : list
                Mapping to convert node ID to pair of IPs in order to build static neighborhood
            n_neighbors : int
                Number of neighbors to consider in the message-passing scheme
            n_layers : int
                Number of layers for the predictive GNN model
            device : 'cuda' | 'cpu'
                Device for the computations

        Returns
        -------
            None
        '''

        super(CCSTGN, self).__init__()

        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.device = device
        self.n_layers = n_layers
        self.agg_function = agg_function

        self.memory_module = MemoryModule(variant, agg_function, n_nodes, input_dim, memory_dim, id_mapping, n_neighbors, n_layers, device)

        self.embedding_module = EmbeddingModule(input_dim, memory_dim, device)

    def detach_memory(self):
        '''
        Detach memory and storage to avoid backpropagation until beginning
        '''
        self.memory_module.detach()

    def forward(self, node_ids, timestamps, flow_features):
        '''
        Forward pass of the model

        Arguments
        ---------
            node_ids : torch.Tensor [batch_size, 1]
                ID of the nodes corresponding to the observed flows
            timestamps : torch.Tensor [batch_size, 1]
                Timestamps corresponding to the observed flows
            flow_features : torch.Tensor [batch_size, input_dim]
                Features corresponding to the observed flows

        Returns
        -------
            flow_embeddings : torch.Tensor [batch_size, memory_dim]
                Embeddings of the observed flow features
            memory_embeddings : torch.Tensor [batch_size, memory_dim]
                Embeddings from memory of observed node IDs
        '''

        # update memory with previously stored information for observed node IDs
        self.memory_module.update(node_ids)

        # use existing memory + graph structure to generate embedding
        # [batch_size, memory_dim]
        memory_embeddings = self.memory_module.compute_memory_embedding(node_ids, timestamps, self.n_layers - 1)

        # compute embedding for current flows
        # [batch_size, memory_dim]
        flow_embeddings = self.embedding_module(flow_features)

        # store embeddings and timestamps for next memory update
        self.memory_module.store(node_ids, flow_embeddings, timestamps, memory_embeddings)

        return flow_embeddings, memory_embeddings

