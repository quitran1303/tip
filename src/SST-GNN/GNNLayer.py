import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):

    def __init__(self, out_size,num_layers):
        super(GNNLayer, self).__init__()

        self.out_size = out_size

        dim = (num_layers+2) * self.out_size
        self.weight = nn.Parameter(torch.FloatTensor(out_size, dim))

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feats, aggregate_feats, his_feats, neighs=None):
        combined = torch.cat([self_feats, aggregate_feats, his_feats], dim=1)
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined