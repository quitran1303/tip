import torch
import numpy as np
from collections import defaultdict
import pandas as pd
import sys

# Data Center

class DataCenter(object):

    def __init__(self, config):
        super(DataCenter, self).__init__()
        self.config = config

    def getPositionEmbedding(self, pos):
        input = np.arange(0, pos + 1, 1)
        a = input * 360
        day = a / 1440
        week = a / 10080
        month = a / 302400
        day = np.deg2rad(day)
        week = np.deg2rad(week)
        day = np.sin(day)
        week = np.sin(week)
        combined = day + week
        return combined

    def load_data(self, ds, st_day, en_day, hr_sample, day, pred_len):

        content_file = self.config['file_path.' + ds + '_content']

        if ds == "PeMSD8" or ds == "PeMSD4":
            timestamp_data = np.load(content_file)
            timestamp_data = timestamp_data[:, :, 2]
        else:
            timestamp_data = []

            with open(content_file) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split(",")
                    info = [float(x) for x in info]

                    timestamp_data.append(info)

        timestamp_data = np.asarray(timestamp_data)
        timestamp_data = timestamp_data.transpose()
        tot_node = timestamp_data.shape[0]

        pos = float(timestamp_data.shape[1])

        print(sys._getframe().f_lineno, "tot_node: ", tot_node, ", POS: ", pos)
        pos_embd = self.getPositionEmbedding(pos)

        st_day -= 1

        timestamp = 24 * hr_sample     #hr_sample = 60

        ts_data = []
        ps_data = []
        for idx in range(st_day, en_day + 1 - day, 1):

            st_point = idx * timestamp
            en_point = (idx + 1) * timestamp

            last_hour = False
            for st in range(st_point, en_point):
                ts = []
                for nd in range(tot_node):
                    a = timestamp_data[nd][st: st + (day * timestamp):timestamp]

                    assert len(a) == day
                    if (st + (day - 1) * timestamp + pred_len) >= len(timestamp_data[nd]):
                        last_hour = True
                        break

                    for pred in range(1, pred_len + 1):
                        gt = timestamp_data[nd][st + (day - 1) * timestamp + pred]
                        a = np.append(a, gt)

                    assert len(a) == (day + pred_len)
                    ts.append(a)

                if last_hour:
                    break
                ts = np.asarray(ts)
                pos_a = pos_embd[st: st + (day * timestamp):timestamp]
                pos_a = np.expand_dims(pos_a, axis=0)
                pos_a = np.repeat(pos_a, tot_node, axis=0)
                ps_data.append(pos_a)
                ts_data.append(ts)

        return ts_data, ps_data

    def load_adj(self, ds):
        print("adj file: ", self.config['file_path.' + ds + '_cites'])
        W = self.load_PeMSD(self.config['file_path.' + ds + '_cites'])
        adj_lists = defaultdict(set)
        for row in range(len(W)):
            adj_lists[row] = set()
            for col in range(len(W)):
                if float(W[row][col]) > 0:
                    adj_lists[row].add(col)
                    adj_lists[col].add(row)

        adj = torch.zeros((len(adj_lists), len(adj_lists)))
        for u in adj_lists:
            for v in adj_lists[u]:
                adj[u][v] = 1
                adj[v][u] = 1
        return adj

    def load_PeMSD(self, file_path, sigma2=0.1, epsilon=0.5, scaling=True):

        try:
            W = pd.read_csv(file_path, header=None).values
        except FileNotFoundError:
            print('ERROR: No File Found.')

        n = W.shape[0]
        W = W / 10000.
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)

        return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
