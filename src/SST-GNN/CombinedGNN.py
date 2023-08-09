import torch
import torch.nn as nn
from GNN import  GNN
import torch.nn.functional as F

class CombinedGNN(nn.Module):
    def __init__(self,input_size,out_size, adj_lists,
                 device,train_data,train_pos,test_data,test_pos,
                 st,GNN_layers,num_timestamps,day):
        super(CombinedGNN, self).__init__()

        self.train_data = train_data
        self.train_pos = train_pos
        self.test_data = test_data
        self.test_pos = test_pos
        self.st = st
        self.num_timestamps = num_timestamps
        self.out_size = out_size
        self.tot_nodes = adj_lists.shape[0]
        self.device = device
        self.adj_lists = adj_lists


        self.day = day

        for timestamp in range(0, self.num_timestamps):

            setattr(self, 'his_model' + str(timestamp),GNN(GNN_layers,input_size-1,
            out_size-1,adj_lists,device))

            setattr(self, 'cur_model' + str(timestamp),GNN(GNN_layers,1,
            1,adj_lists,device))


        self.his_weight = nn.Parameter(torch.FloatTensor(out_size-1, self.num_timestamps*out_size-1))
        self.cur_weight = nn.Parameter(torch.FloatTensor(1, self.num_timestamps*1))

        dim = self.num_timestamps*out_size
        self.final_weight = nn.Parameter(torch.FloatTensor(dim,dim))

        self.init_params()

    def init_params(self):
      for param in self.parameters():

          if(len(param.shape)>1):
            nn.init.xavier_uniform_(param)


    def forward(self,nodes_batch,isTrain):

        his_timestamp_embds = torch.zeros((nodes_batch.shape[0],self.out_size-1)).to(self.device)
        cur_timestamp_embds = torch.zeros((nodes_batch.shape[0],1)).to(self.device)


        historical_embds = []
        current_embds = []

        for timestamp in range(0, self.num_timestamps):


            historicalModel = getattr(self, 'his_model' + str(timestamp))
            historicalModel.adj_lists = self.adj_lists
            setattr(historicalModel, 'timestamp_no', timestamp)
            setattr(historicalModel, 'pre_latent_feats', his_timestamp_embds)

            if isTrain:
                his_raw_features = self.train_data[self.st+timestamp]
                his_pos = self.train_pos[self.st+timestamp]
            else:
                his_raw_features = self.test_data[self.st+timestamp]
                his_pos = self.test_pos[self.st+timestamp]

            his_raw_features = his_raw_features[:,:self.day-1]
            his_pos = his_pos[:,:self.day-1]
            setattr(historicalModel,'raw_features',his_raw_features)


            his_timestamp_embds = historicalModel(nodes_batch,timestamp) + his_pos

            historical_embds.append(his_timestamp_embds)
            upto_current_timestamp = torch.cat(historical_embds,dim=1)
            weight = self.his_weight[:,:(timestamp+1)*(self.out_size-1)]
            his_timestamp_embds = F.relu(weight.mm(upto_current_timestamp.t())).t()

            currentModel = getattr(self, 'cur_model' + str(timestamp))
            currentModel.adj_lists = self.adj_lists
            setattr(currentModel, 'timestamp_no', timestamp)
            setattr(currentModel, 'pre_latent_feats', cur_timestamp_embds)

            if isTrain:
                cur_raw_features = self.train_data[self.st+timestamp]
                cur_pos = self.train_pos[self.st+timestamp]
            else:
                cur_raw_features = self.test_data[self.st+timestamp]
                cur_pos = self.test_pos[self.st+timestamp]

            cur_raw_features = cur_raw_features[:,self.day-1:self.day]
            cur_pos = cur_pos[:,self.day-1:self.day]
            setattr(currentModel,'raw_features',cur_raw_features)

            cur_timestamp_embds = currentModel(nodes_batch,timestamp) + cur_pos

            current_embds.append(cur_timestamp_embds)
            upto_current_timestamp = torch.cat(current_embds,dim=1)
            weight = self.cur_weight[:,:(timestamp+1)*1]
            cur_timestamp_embds = F.relu(weight.mm(upto_current_timestamp.t())).t()


        his_final_embds = torch.cat(historical_embds,dim=1)
        cur_final_embds = torch.cat(current_embds,dim=1)

        final_embds = [his_final_embds,cur_final_embds]
        final_embds = torch.cat(final_embds,dim=1)
        final_embds = F.relu(self.final_weight.mm(final_embds.t()).t())

        return final_embds