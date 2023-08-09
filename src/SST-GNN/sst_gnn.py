
#Model: SST-GNN Model

# Packages
import os
import torch
import argparse
import pyhocon
import random
import numpy as np
from DataLoader import DataLoader
from TrafficModel import  TrafficModel


# Main Function

parser = argparse.ArgumentParser(description='pytorch version of Traffic Forecasting GNN')
parser.add_argument('-f')  


parser.add_argument('--dataset', type=str, default='EBSP')
#parser.add_argument('--dataset', type=str, default='PeMSD7')
parser.add_argument('--GNN_layers', type=int, default=3)
parser.add_argument('--num_timestamps', type=int, default=60)
parser.add_argument('--pred_len', type=int, default=9)
parser.add_argument('--epochs', type=int, default=10) #200 default
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cuda', action='store_true',help='use CUDA')
parser.add_argument('--trained_model', action='store_true')
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--input_size', type=int, default=8)
parser.add_argument('--out_size', type=int, default=8)



args = parser.parse_args()
args.cuda = False
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)

if __name__ == '__main__':

    print('Traffic Forecasting GNN with Historical and Current Model')

    #set user given seed to every random generator
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    PATH = os.getcwd() + "/"
    config_file = PATH + "experiments.conf"
 
    config = pyhocon.ConfigFactory.parse_file(config_file)
    ds = args.dataset
    pred_len = args.pred_len
    data_loader = DataLoader(config,ds,pred_len)
    train_data,train_pos,test_data,test_pos,adj = data_loader.load_data()

    num_timestamps = args.num_timestamps 
    GNN_layers = args.GNN_layers
    input_size = args.input_size
    out_size = args.input_size
    epochs = args.epochs
    save_flag = args.save_model

b_debug = False
t_debug = False

hModel = TrafficModel (train_data,train_pos,test_data,test_pos,adj,config, ds, input_size, out_size,GNN_layers,
                epochs, device,num_timestamps,pred_len,save_flag,PATH,b_debug,t_debug)
if not args.trained_model: #train model and evaluate
    hModel.run_model()
else:
    print("Running Trained Model...")
    hModel.run_Trained_Model() #run trained model


