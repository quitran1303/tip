import torch
import torch.nn as nn
from GNNLayer import GNNLayer

# Spatial GNN
class GNN(nn.Module):
    def __init__(self, num_layers ,input_size ,out_size, adj_lists, device):
        super(GNN, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.out_size = out_size
        self.adj_lists = []
        self.device = device

        _ones = torch.ones(adj_lists.shape).to(device)
        _zeros = torch.zeros(adj_lists.shape).to(device)

        setattr(self, 'layer_adj1', adj_lists)

        for index in range(2, num_layers +1):
            cur_adj = torch.pow(adj_lists ,index)
            cur_adj = torch.where(cur_adj >0, _ones, _zeros)

            prev_adj = torch.pow(adj_lists ,index -1)
            prev_adj = torch.where(prev_adj >0, _ones, _zeros)

            layer_adj = cur_adj - prev_adj
            setattr(self, 'layer_adj' +str(index), layer_adj)

        self.GNN_Layer = GNNLayer(out_size, num_layers)

    def forward(self, nodes_batch ,ts):

        pre_hidden_embs = self.raw_features
        nb = nodes_batch

        aggregated_feats = []
        for index in range(1, self.num_layers +1):
            neigh_feats = self.aggregate(nb, pre_hidden_embs ,index)
            aggregated_feats.append(neigh_feats)

        aggregated_feats = torch.cat(aggregated_feats ,dim=1)

        cur_hidden_embs = self.GNN_Layer(pre_hidden_embs ,aggregated_feats, self.pre_latent_feats)

        pre_hidden_embs = cur_hidden_embs

        return pre_hidden_embs

    def aggregate(self, nodes ,pre_hidden_embs ,layer):

        embed_matrix = pre_hidden_embs
        mask = getattr(self, 'layer_adj' +str(layer))

        num_neigh = mask.sum(1, keepdim=True)
        _ones = torch.ones(num_neigh.shape).to(self.device)
        num_neigh = torch.where(num_neigh >0 ,num_neigh ,_ones)
        mask = mask.div(num_neigh)

        aggregate_feats = mask.mm(embed_matrix)

        return aggregate_feats

