
import torch
from torch import nn
import numpy as np
import math
from collections import defaultdict

from modules.iso_mlp import ISO_MLP

import sys

class MemoryModule(nn.Module):
    '''
    Memory module
    '''

    def __init__(self, variant, agg_function, n_nodes, input_dim, memory_dim, id_mapping, n_neighbors, n_layers, device):
        '''
        Initialization of the module

        Arguments
        ---------
            variant : 'rs' | 'iso'
                Variant of the CCTGN model
            agg_function : 'mean' | 'softmax'
            n_nodes : int
                Maximum number of nodes allowed in memory
            memory_dim : int
                Dimensionality of node embeddings stored in memory
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

        super().__init__()

        self.variant = variant
        self.n_nodes = n_nodes + 1 # extra 0 vector at last pos to simplify neighbor aggregation when neighborhood too small
        self.filler_node = n_nodes
        self.memory_dim = memory_dim
        self.n_neighbors = n_neighbors + 1 # extra neighbor is node itself
        self.device = device
        self.n_layers = n_layers
        self.id_mapping = id_mapping
        self.agg_function = agg_function

        # dictionary to avoid recomputing temporal degrees for a given prediction
        self.temporal_degrees = {}

        # create parameters for the specific variant
        if variant == 'rs':
            self.W = nn.Parameter(torch.empty((memory_dim, memory_dim)))
            nn.init.kaiming_uniform_(self.W)
        elif variant == 'iso':
            self.mlps = torch.nn.ModuleList([ISO_MLP(memory_dim, device) for _ in range(n_layers)])
            self.eps = nn.Parameter(torch.zeros(n_layers))

        # mechanism for memory update
        self.memory_updater = nn.GRUCell(input_size=memory_dim, hidden_size=memory_dim)

        # Create memory
        self.init_memory()

    def init_memory(self):
        '''
        Initialize memory to 0

        Arguments
        ---------
            None

        Returns
        -------
            None
        '''

        self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dim)).to(self.device), requires_grad=False)
        self.last_update = nn.Parameter(torch.ones(self.n_nodes).to(self.device) * -1, requires_grad=False)
        self.flow_embedding_storage = defaultdict(list)
        self.timestamp_storage = defaultdict(list)
        if self.agg_function == 'cosine':
            self.memory_embedding_storage = defaultdict(list)


    def compute_neighborhood(self, node_id, timestamp, root=True):
        '''
        Compute the spatiotemporal neighborhood for a particular node ID and timestamp

        Arguments
        ---------
            node_id : int
                ID of the node in consideration
            timestamp : float
                Timestamp for this particular instance of the node ID
            root : bool
                Flag to determine whether we need to recurse to find the degrees of the neighbors of this node

        Returns
        -------
            spatiotemporal_neighbors : torch.Tensor [self.n_neighbors,]
                IDs of the self.n_neighbors most recently updated neighbors of the considered node
            spatiotemporal_degrees : torch.Tensor [self.n_neighbors,]
                Spatiotemporal degrees of the spatiotemporal_neighbors
        '''

        # > -1 to filter out nodes that have not yet being observed
        # filter out nodes beyond temporal threshold (> avoids considering the node itself, which will be manually added later on)
        temporal_neighbors = torch.where((self.last_update > self.last_update[node_id]) & (self.last_update < timestamp))[0]

        # sort the temporal neighbors by decreasing last update
        neighbor_update_times = self.last_update[temporal_neighbors].flatten()
        sorted_times_indices = torch.argsort(neighbor_update_times * -1)
        sorted_temporal_neighbors = temporal_neighbors[sorted_times_indices]

        # create result tensors
        spatiotemporal_neighbors = torch.ones(self.n_neighbors, dtype=torch.int64).to(self.device) * self.filler_node
        spatiotemporal_neighbors[0] = node_id
        spatiotemporal_degrees = torch.ones(self.n_neighbors).to(self.device)

        # if node is filler there is no need to proceed
        if node_id != self.filler_node:

            # get node's channel IPs
            node_ips = self.id_mapping[node_id]

            # count number of found neighbors
            neighbor_count = 1

            # filter out nodes that are not adjacent (based on channel IPs)
            for n_id in sorted_temporal_neighbors:

                # get potential neighbor's channel IPs
                n_ips = self.id_mapping[n_id]

                # if there are IPs in common
                if not node_ips.isdisjoint(n_ips):

                    # increase degree of root node
                    spatiotemporal_degrees[0] += 1

                    # recurse only for root node and within neighborhood size
                    if root and spatiotemporal_degrees[0] <= self.n_neighbors:

                        # only compute temporal degrees of neighbors for rs variant
                        if self.variant == 'rs':

                            if n_id in self.temporal_degrees.keys():
                                st_dg = self.temporal_degrees[n_id]
                            else:
                                _, st_dgs = self.compute_neighborhood(n_id, timestamp, False)
                                st_dg = st_dgs[0]
                                # save computed temporal degree to avoid recomputations
                                self.temporal_degrees[n_id] = st_dg

                            # store temporal degree of found neighbor
                            spatiotemporal_degrees[neighbor_count] = st_dg

                        # store id of found neighbor
                        spatiotemporal_neighbors[neighbor_count] = n_id
                        neighbor_count += 1

                    # after finding all sampled neighbors can break in non-rs variants
                    # since the node's temporal degree does not need to be computed
                    if neighbor_count == self.n_neighbors and self.variant != 'rs':
                        break

        # save computed temporal degree to avoid recomputations
        self.temporal_degrees[node_id] = spatiotemporal_degrees[0]

        return spatiotemporal_neighbors, spatiotemporal_degrees

    def get_spatiotemporal_neighbors(self, node_ids, timestamps):
        '''
        Compute the spatiotemporal neighborhood for each node ID by retrieving the n_neighbors most
        recently updated neighbors.

        Arguments
        ---------
            node_ids : torch.Tensor [batch_size, 1]
                ID of the nodes corresponding to the observed flows
            timestamps : torch.Tensor [batch_size, 1]
                Timestamps corresponding to the observed flows

        Returns
        -------
            neighbors : torch.Tensor [batch_size, self.n_neighbors]
                ID of the most recently updated neighbors for each considered node ID
            degrees : torch.Tensor [self.n_neighbors,]
                Spatiotemporal degrees of the neighbors
        '''

        # save computations to avoid recomputing
        neighborhoods = {}

        # create result matrices
        neighbors = torch.zeros(node_ids.shape[0], self.n_neighbors, dtype=torch.int64).to(self.device)
        degrees = torch.zeros(node_ids.shape[0], self.n_neighbors).to(self.device)

        for idx, nid in enumerate(node_ids.cpu().numpy()):

            # get int to avoid problems with tensors as keys
            nid = nid[0]

            # proceed if temporal neighborhood for node has not been computed, otherwise simply retrieve it
            if nid not in neighborhoods.keys():

                # retrieve neighborhood and corresponding temporal degrees
                st_neighbors, st_degrees = self.compute_neighborhood(nid, timestamps[idx].cpu().numpy()[0])

                neighborhoods[nid] = (st_neighbors, st_degrees)

            # add this node's neighbors and degrees
            stn, std = neighborhoods[nid]
            neighbors[idx] = stn
            degrees[idx] = std

        return neighbors, degrees

    def compute_memory_embedding(self, node_ids, timestamps, layer):
        '''
        Compute memory embeddings from memory information

        Arguments
        ---------
            node_ids : torch.Tensor [batch_size, 1]
                ID of the nodes corresponding to the observed flows
            timestamps : torch.Tensor [batch_size, 1]
                Timestamps corresponding to the observed flows
            layer : int
                Index of layer of the GNN being computed

        Returns
        -------
            memory_embeddings : torch.Tensor [batch_size, memory_dim]
                Predicted flow features
        '''

        # reset outdated stored temporal degrees at the start of the prediction
        if layer == self.n_layers - 1:
            self.temporal_degrees.clear()

        # compute temporal neighborhoods according to most recently updated nodes
        # [batch_size, n_neighbors]
        neighbors, degrees = self.get_spatiotemporal_neighbors(node_ids, timestamps)

        # compute layer's embeddings
        if layer == 0:
            # retrieve memory of neighbors
            # [batch_size, n_neighbors, memory_dim]
            neighborhood_embeddings = torch.index_select(self.memory, dim=0, index=neighbors.view(-1)).reshape(neighbors.shape[0], neighbors.shape[1], self.memory_dim)
        else:
            rec_ids = neighbors.reshape(-1, 1)
            # set same timestamp for all neighbors of a particular node
            rec_timestamps = np.repeat(timestamps.cpu(), self.n_neighbors + 1).reshape(-1, 1).to(self.device)
            neighborhood_embeddings = self.compute_memory_embedding(rec_ids, rec_timestamps, layer - 1).reshape(neighbors.shape[0], neighbors.shape[1], self.memory_dim)

        # Proceed for the residual symmetric variant
        if self.variant == 'rs':

            # adjacency weights
            # [batch_size, n_neighbors]
            neighbor_weights = torch.pow(degrees, -1/2).reshape(neighbors.shape)
            node_weights = degrees[:, 0].reshape(node_ids.shape)

            adj_weights = node_weights * neighbor_weights

            # compute weighted sum of neighborhood
            # [batch_size, memory_dim]
            neighborhood_sum = torch.einsum('ij,ijk->ik', adj_weights, neighborhood_embeddings)

            # residual root node embeddings
            # [batch_size, memory_dim]
            F = neighborhood_embeddings[:, 0, :]

            # symmetric weight shared across layers
            Ws = (self.W + self.W.T) / 2.

            # compute prediction with residual connection and symmetric weights
            # [batch_size, memory_dim]
            memory_embeddings = F + nn.functional.relu(neighborhood_sum @ Ws)

        elif self.variant == 'iso':

            # apply scaling factor to embeddings of root node
            neighborhood_embeddings[:, 0] = (1 + self.eps[layer]) * neighborhood_embeddings[:, 0].clone()

            # sum over neighborhoods
            neighborhood_sum = torch.sum(neighborhood_embeddings, dim=1)

            # generate memory_embeddings with the defined mlps
            memory_embeddings = self.mlps[layer](neighborhood_sum)

        return memory_embeddings

    def aggregate_storage(self, node_ids):
        '''
        Compute aggregated information from storage

        Arguments
        ---------
            node_ids : torch.Tensor [batch_size, 1]
                ID of the nodes corresponding to the observed flows

        Returns
        -------
            agg_ids : list
                Unique node IDs for which there is storage to aggregate
            agg_flows : torch.Tensor [len(agg_ids), memory_dim]
                Aggregated flows corresponding to each unique node ID
            agg_timestamps : torch.Tensor [len(agg_ids)]
                Aggregated timestamps corresponding to each unique node ID
        '''

        # compute unique node IDs
        un_ids = torch.unique(node_ids)

        agg_ids = []
        agg_flows = []
        agg_timestamps = []

        for nid in un_ids:

            nid = nid.item()

            # if there are flows to aggregate for this ID
            if len(self.flow_embedding_storage[nid]) > 0:

                # stack stored tensors into single tensor
                nid_flow_embeddings = torch.vstack(self.flow_embedding_storage[nid])
                nid_timestamps = torch.vstack(self.timestamp_storage[nid])

                # apply aggregation function to flows
                if self.agg_function == 'softmax':
                    # use the timestamps to weight the aggregation of flows embeddings
                    # giving more weight to later (more recent) observations
                    weight_timestamps = nn.functional.softmax(nid_timestamps, dim=0)
                    agg_flow_embeddings = torch.matmul(weight_timestamps.T, nid_flow_embeddings).flatten()
                elif self.agg_function == 'mean':
                    # aggregate using the mean of stored flow embeddings (bias aggregation toward most common, 'normal', flows)
                    agg_flow_embeddings = torch.mean(nid_flow_embeddings, dim=0)
                elif self.agg_function == 'cosine':
                    nid_memory_embeddings = torch.vstack(self.memory_embedding_storage[nid])

                    # ignore flows for which the cosine similarity is < 0
                    cosine_sim = nn.functional.cosine_similarity(nid_flow_embeddings, nid_memory_embeddings)
                    sim_weights = nn.functional.relu(cosine_sim)
                    weight_timestamps = sim_weights * nid_timestamps.flatten()

                    # if weights are all 0 -> clear and skip ID
                    if weight_timestamps.sum() == 0:
                        self.flow_embedding_storage[nid].clear()
                        self.timestamp_storage[nid].clear()
                        self.memory_embedding_storage[nid].clear()
                        continue

                    # normalize weights
                    agg_weights = (weight_timestamps / weight_timestamps.sum()).unsqueeze(1)
                    agg_flow_embeddings = torch.matmul(agg_weights.T, nid_flow_embeddings).flatten()

                agg_flows.append(agg_flow_embeddings)

                # add timestamp of most recent flow for last update
                agg_timestamps.append(torch.max(nid_timestamps))

                # clear messages after usage for update
                self.flow_embedding_storage[nid].clear()
                self.timestamp_storage[nid].clear()
                if self.agg_function == 'cosine':
                    self.memory_embedding_storage[nid].clear()

                # add agg_id
                agg_ids.append(nid)

        # if there are nodes to update
        if len(agg_ids) > 0:
            # stack stored tensors into single tensor
            agg_flows = torch.vstack(agg_flows)
            agg_timestamps = torch.hstack(agg_timestamps)

        return agg_ids, agg_flows, agg_timestamps

    def update(self, node_ids):
        '''
        Update memory with previously stored information for observed node IDs

        Arguments
        ---------
            node_ids : torch.Tensor [batch_size, 1]
                ID of the nodes corresponding to the observed flows

        Returns
        -------
            None
        '''

        # compute aggregated information from storage
        agg_ids, agg_flows, agg_timestamps = self.aggregate_storage(node_ids)

        # if there are nodes to update
        if len(agg_ids) > 0:

            # update memory
            old_memory = self.memory[agg_ids, :]
            updated_memory = self.memory_updater(agg_flows, old_memory)
            self.memory[agg_ids, :] = updated_memory

            # update timestamps
            self.last_update[agg_ids] = agg_timestamps

    def store(self, node_ids, flow_embeddings, timestamps, memory_embeddings):
        '''
        Store embeddings and timestamps for next memory update

        Arguments
        ---------
            node_ids : torch.Tensor [batch_size, 1]
                ID of the nodes corresponding to the observed flows
            flow_embeddings : torch.Tensor [batch_size, memory_dim]
                Embeddings of the observed flow features
            timestamps : torch.Tensor [batch_size, 1]
                Timestamps corresponding to the observed flows
            memory_embeddings : torch.Tensor [batch_size, memory_dim]
                Embeddings from memory of observed node IDs

        Returns
        -------
            None
        '''

        for i, nid in enumerate(node_ids.flatten()):
            nid = nid.item()
            self.flow_embedding_storage[nid].append(flow_embeddings[i])
            self.timestamp_storage[nid].append(timestamps[i])
            if self.agg_function == 'cosine':
                self.memory_embedding_storage[nid].append(memory_embeddings[i])

    def detach(self):
        '''
        Detach memory and storage to avoid backpropagation until beginning
        '''

        self.memory.detach_()

        # Detach all stored messages
        for nid in self.flow_embedding_storage.keys():
            det_flows = []
            det_timestamps = []
            if self.agg_function == 'cosine':
                det_memory = []

            for i in range(len(self.flow_embedding_storage[nid])):
                det_flows.append(self.flow_embedding_storage[nid][i].detach())
                det_timestamps.append(self.timestamp_storage[nid][i].detach())
                if self.agg_function == 'cosine':
                    det_memory.append(self.memory_embedding_storage[nid][i].detach())

            self.flow_embedding_storage[nid] = det_flows
            self.timestamp_storage[nid] = det_timestamps
            if self.agg_function == 'cosine':
                self.memory_embedding_storage[nid] = det_memory

