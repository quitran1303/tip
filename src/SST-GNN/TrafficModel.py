import torch
from sklearn.metrics import mean_absolute_error
import numpy as np
import datetime
from Regression import Regression
from CombinedGNN import CombinedGNN
from Utility import evaluate, apply_model, save_files, printProgressBar
from sklearn.metrics import mean_absolute_percentage_error
import sys


"""# Traffic Model"""

class TrafficModel:

    def __init__(self, train_data, train_pos, test_data, test_pos, adj,
                 config, ds, input_size, out_size, GNN_layers,
                 epochs, device, num_timestamps, pred_len, save_flag, PATH, b_debug, t_debug):

        super(TrafficModel, self).__init__()

        self.train_data, self.train_pos, self.test_data, self.test_pos, self.adj = train_data, train_pos, test_data, test_pos, adj
        self.all_nodes = [i for i in range(self.adj.shape[0])]

        self.ds = ds
        self.input_size = input_size
        self.out_size = out_size
        self.GNN_layers = GNN_layers
        self.day = input_size
        self.device = device
        self.epochs = epochs
        self.regression = Regression(input_size * num_timestamps, pred_len)
        self.num_timestamps = num_timestamps
        self.pred_len = pred_len

        self.node_bsz = 512
        self.PATH = PATH
        self.save_flag = save_flag

        self.train_data = torch.FloatTensor(self.train_data).to(device)
        self.test_data = torch.FloatTensor(self.test_data).to(device)
        self.train_pos = torch.FloatTensor(self.train_pos).to(device)
        self.test_pos = torch.FloatTensor(self.test_pos).to(device)
        self.all_nodes = torch.LongTensor(self.all_nodes).to(device)
        self.adj = torch.FloatTensor(self.adj).to(device)

        self.b_debug = b_debug
        self.t_debug = t_debug

    def run_model(self):

        print("input size: ", self.input_size, ", output size: ", self.out_size, ", self adj len:", len(self.adj), ", self.train_pos:", len(self.train_pos), ", self.test_pos:", len(self.test_pos),
              "GNN layers: ", self.GNN_layers, ", num timestamps: ", self.num_timestamps)
        timeStampModel = CombinedGNN(self.input_size, self.out_size, self.adj,
                                     self.device, self.train_data, self.train_pos, self.test_data, self.test_pos, 1,
                                     self.GNN_layers, self.num_timestamps, self.day)



        timeStampModel.to(self.device)

        regression = self.regression
        regression.to(self.device)

        min_RMSE = float("Inf")
        min_MAE = float("Inf")
        min_MAPE = float("Inf")
        best_test = float("Inf")

        lr = 0.001
        # if self.ds == "PeMSD7":
        # lr = 0.001
        # elif self.ds == "PeMSD8":
        #   lr = 0.0001

        train_loss = torch.tensor(0.).to(self.device)

        for epoch in range(1, self.epochs):

            print("Epoch: ", epoch, " running...")

            start_train_time = datetime.datetime.now()
            print("Start training at :", start_train_time, ", training size:", len(self.train_data))
            tot_timestamp = len(self.train_data)
            if self.t_debug:
                tot_timestamp = 120
            idx = np.random.permutation(tot_timestamp + 1 - self.num_timestamps)

            train_count = 0
            print("date: ", datetime.datetime.now(), ", train_count: ", len(idx))
            printProgressBar(0, len(idx), prefix='Progress:', suffix='Complete', length=50)
            for data_timestamp in idx:
                train_count += 1
                printProgressBar(train_count, len(idx), prefix='Progress:', suffix='Complete', length=50)
                timeStampModel, regression, train_loss = apply_model(self.all_nodes, timeStampModel,
                                                                     regression, data_timestamp, self.node_bsz,
                                                                     self.device,
                                                                     self.pred_len, self.train_data,
                                                                     self.num_timestamps, self.day, train_loss, lr)

                if self.b_debug:
                    break

            train_loss /= len(idx)
            # if self.ds=="PeMSD7":
            if epoch <= 24 and epoch % 8 == 0:
                lr *= 0.5
            else:
                lr = 0.0001

            print("Train avg loss: ", train_loss)

            end_train_time = datetime.datetime.now()
            print("Training done at: ", datetime.datetime.now(), ", take : ", end_train_time - start_train_time)

            start_test_time = datetime.datetime.now()
            print("Testing Start at: ", start_test_time, ", testing size:", len(self.test_data))
            pred = []
            label = []
            tot_timestamp = len(self.test_data)
            if self.t_debug:
                tot_timestamp = 120
            idx = np.random.permutation(tot_timestamp + 1 - self.num_timestamps)
            test_loss = torch.tensor(0.).to(self.device)

            test_count = 0
            print("date: ", datetime.datetime.now(), "test_count: ", len(idx))
            # Initial call to print 0% progress
            printProgressBar(0, len(idx), prefix='Progress:', suffix='Complete', length=50)
            for data_timestamp in idx:
                test_count += 1
                printProgressBar(test_count, len(idx), prefix='Progress:', suffix='Complete', length=50)
                # window slide
                timeStampModel.st = data_timestamp

                # test_label
                raw_features = self.test_data[timeStampModel.st + self.num_timestamps - 1]
                test_label = raw_features[:, self.day:]

                # evaluate
                temp_predicts, test_loss = evaluate(self.all_nodes, test_label, timeStampModel, regression,
                                                    self.device, test_loss)

                label = label + test_label.detach().tolist()
                pred = pred + temp_predicts.detach().tolist()

                if self.b_debug:
                    break

            end_test_time = datetime.datetime.now()
            print("Testing done at: ", end_test_time, ", take: ", end_test_time - start_test_time)

            test_loss /= len(idx)
            print("Average Test Loss: ", test_loss)
            print("label size:", len(label), ", pred size: ", len(pred))

            RMSE = torch.nn.MSELoss()(torch.FloatTensor(pred), torch.FloatTensor(label))
            RMSE = torch.sqrt(RMSE).item()
            MAE = mean_absolute_error(pred, label)
            MAPE = mean_absolute_percentage_error(label, pred)

            save_files(label, pred, self.ds, self.pred_len)

            print("Test loss:", test_loss, ", best test: ", best_test)
            if test_loss <= best_test:
                best_test = test_loss
                pred_after = self.pred_len * 1
                min_RMSE = RMSE
                min_MAE = MAE
                min_MAPE = MAPE
                print("save flag:", self.save_flag)
                if self.save_flag:
                    print("save path timestampmodel: ",
                          self.PATH + "/" + self.ds + "/bestTmodel_" + str(pred_after) + "minutes.pth")
                    torch.save(timeStampModel,
                               self.PATH + "/" + self.ds + "/bestTmodel_" + str(pred_after) + "minutes.pth")
                    print("save path regression: ",
                          self.PATH + "/" + self.ds + "/bestRegression_" + str(pred_after) + "minutes.pth")
                    torch.save(regression,
                               self.PATH + "/" + self.ds + "/bestRegression_" + str(pred_after) + "minutes.pth")

            print("Epoch:", epoch)
            print("RMSE: ", RMSE)
            print("MAE: ", MAE)
            print("MAPE: ", MAPE)
            print("===============================================")

            print("Min RMSE: ", min_RMSE)
            print("Min MAE: ", min_MAE)
            print("Min MAPE: ", min_MAPE)

            print("===============================================")

        return

    def run_Trained_Model(self):
        pred_after = self.pred_len
        timeStampModel = torch.load(
            self.PATH + "/saved_model/" + self.ds + "/bestTmodel_" + str(pred_after) + "minutes.pth")
        regression = torch.load(
            self.PATH + "/saved_model/" + self.ds + "/bestRegression_" + str(pred_after) + "minutes.pth")
        pred = []
        label = []
        tot_timestamp = len(self.test_data)
        idx = np.random.permutation(tot_timestamp + 1 - self.num_timestamps)
        test_loss = torch.tensor(0.).to(self.device)
        printProgressBar(0, len(idx), prefix='Progress:', suffix='Complete', length=50)
        count = 0
        for data_timestamp in idx:
            count += 1
            printProgressBar(count, len(idx), prefix='Progress:', suffix='Complete', length=50)
            # window slide
            timeStampModel.st = data_timestamp

            # test_label
            raw_features = self.test_data[timeStampModel.st + self.num_timestamps - 1]
            test_label = raw_features[:, self.day:]

            # evaluate
            temp_predicts, test_loss = evaluate(self.all_nodes, test_label, timeStampModel, regression,
                                                self.device, test_loss)

            label = label + test_label.detach().tolist()
            pred = pred + temp_predicts.detach().tolist()

        test_loss /= len(idx)
        print("Average Test Loss: ", test_loss)
        print("Label shape", label[1])

        #save_files(label, pred, self.pred_len)

        RMSE = torch.nn.MSELoss()(torch.FloatTensor(pred), torch.FloatTensor(label))
        RMSE = torch.sqrt(RMSE).item()
        MAE = mean_absolute_error(pred, label)
        
        MAPE = mean_absolute_percentage_error(label, pred)
        
        save_files(label, pred, self.ds, self.pred_len)

        print("RMSE: ", RMSE)
        print("MAE: ", MAE)
        print("MAPE: ", MAPE)
        print("===============================================")
