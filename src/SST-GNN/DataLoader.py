from DataCenter import DataCenter

"""# Data Loader"""

class DataLoader:

  def __init__(self, config,ds,pred_len):

    super(DataLoader, self).__init__()

    self.ds = ds
    self.dataCenter = DataCenter(config)

    if ds == "EBFL":
      train_st = 1
      train_en = 47

      test_st = 48
      test_en = 87

    if ds == "WBFL":
      train_st = 1
      train_en = 47

      test_st = 48
      test_en = 87

    if ds == "EBSP":
      train_st = 1
      train_en = 69

      test_st = 70
      test_en = 87

    if ds == "WBSP":
      train_st = 1
      train_en = 69

      test_st = 70
      test_en = 87

    if ds == "PeMSD7":
      train_st = 1
      train_en = 69

      test_st = 70
      test_en = 87

    elif ds == "PeMSD8":
      train_st = 1
      train_en = 50

      test_st = 51
      test_en = 62

    elif ds == "PeMSD4" :
      train_st = 1
      train_en = 47

      test_st = 48
      test_en = 58

    self.train_st = train_st
    self.train_en = train_en
    self.test_st = test_st
    self.test_en = test_en

    self.hr_sample = 60
    self.day = 8
    self.pred_len = pred_len

  def load_data(self):
    print("Loading Data...")
    train_data,train_pos = self.dataCenter.load_data(self.ds,self.train_st,self.train_en,self.hr_sample,self.day,self.pred_len)
    test_data,test_pos = self.dataCenter.load_data(self.ds,self.test_st,self.test_en,self.hr_sample,self.day,self.pred_len)
    adj = self.dataCenter.load_adj(self.ds)
    print("Data Loaded")
    print("Dataset: ", self.ds)
    print("Total Nodes: ",adj.shape[0])
    print("Train timestamps: ",len(train_data))
    print("Test timestamps: ",len(test_data))
    print("Predicting After: ",self.pred_len*1,"minutes")

    return train_data,train_pos,test_data,test_pos,adj
