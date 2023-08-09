import torch
import math
import numpy as np
import torch.nn as nn
from datetime import datetime
import os.path

def evaluate(test_nodes,labels, graphSage, regression, device,test_loss):


    models = [graphSage, regression]

    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                params.append(param)


    val_nodes = test_nodes
    embs = graphSage(val_nodes,False)
    predicts = regression(embs)
    loss_sup = torch.nn.MSELoss()(predicts, labels)
    loss_sup /= len(val_nodes)
    test_loss += loss_sup.item()

    for param in params:
        param.requires_grad = True

    return predicts,test_loss


def RMSELoss(yhat,y):
    yhat = torch.FloatTensor(yhat)
    y = torch.FloatTensor(y)
    return torch.sqrt(torch.mean((yhat-y)**2)).item()


def mean_absolute_percentage_error(y_true, y_pred):
  print("mean_absolute_percentage_error")

  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  
  
  print("y_true:", len(y_true))
  print("y_pred: ", len(y_pred))

  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def save_files(y_true, y_pred, direction, pred_len):
    save_path = './saved_pred'
    date = datetime.now().strftime('%Y%m%d%H%M')
    true_file = f'y_true_{date}.{direction}.{pred_len}_mins.csv'
    pred_file = f'y_pred_{date}.{direction}.{pred_len}_mins.csv'
    np.savetxt(os.path.join(save_path,true_file), y_true, delimiter=",")
    np.savetxt(os.path.join(save_path,pred_file), y_pred, delimiter=",")
    print("SAVE - True file : ",os.path.join(save_path,true_file), ", prediction file: ", os.path.join(save_path,pred_file))
    return True


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 3, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% - {iteration}/{total} {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def apply_model(train_nodes, CombinedGNN, regression, data_timestamp,
                node_batch_sz, device,pred_len,train_data,num_timestamps,day,avg_loss,lr):


    models = [CombinedGNN, regression]
    params = []
    for model in models:
      for param in model.parameters():
          if param.requires_grad:
              params.append(param)



    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0)

    optimizer.zero_grad()  # set gradients in zero...
    for model in models:
      model.zero_grad()  # set gradients in zero

    node_batches = math.ceil(len(train_nodes) / node_batch_sz)

    loss = torch.tensor(0.).to(device)
    #window slide
    CombinedGNN.st = data_timestamp
    #test_label
    raw_features = train_data[CombinedGNN.st+num_timestamps-1]
    labels = raw_features[:,day:]
    for index in range(node_batches):

      nodes_batch = train_nodes[index * node_batch_sz:(index + 1) * node_batch_sz]
      nodes_batch = nodes_batch.view(nodes_batch.shape[0],1)
      labels_batch = labels[nodes_batch]
      labels_batch = labels_batch.view(len(labels_batch),pred_len)
      embs_batch = CombinedGNN(nodes_batch,True)  # Finds embeddings for all the ndoes in nodes_batch

      logists = regression(embs_batch)


      loss_sup = torch.nn.MSELoss()(logists, labels_batch)

      loss_sup /= len(nodes_batch)
      loss += loss_sup



    avg_loss += loss.item()

    loss.backward()
    for model in models:
      nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()

    optimizer.zero_grad()
    for model in models:
      model.zero_grad()

    return CombinedGNN, regression,avg_loss