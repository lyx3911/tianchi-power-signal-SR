import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy import interpolate as inter
from tqdm import tqdm
import scipy.signal as signal


class TimeSeriesDataset(Dataset):
    def __init__(self, path_low, path_high, path_high_high, window_size=10000, time_step=1000, scale=10, is_train=True):
        self.data_low = self.read_file(path_low)
        self.data_high = self.read_file(path_high)
        self.data_hhigh = self.read_file(path_high_high)
        self.scale = scale

        self.window_size = window_size
        self.time_step = time_step

        # self.data_low = self.inter_data(self.data_low)
        self.data_len = (len(self.data_low['bus1'])-window_size) // time_step

        print("dataset info: ")
        print("low data:", path_low)
        print("high data:", path_high)
        print("window size:", window_size)
        print("time step:", time_step)
        print("low data numbers:", len(self.data_low['bus1']))
        print("high data numbers:", len(self.data_high['bus1']))
        print("dataset len:", self.data_len)

    
    def read_file(self, file_name):
        data_dict = {'bus1': [], 
                    'bus2': [],
                    'bus3': []}
        data = pd.read_csv(file_name, header=None)
        data_dict['bus1'] = data.values[:, 0].astype(float).tolist()
        data_dict['bus2'] = data.values[:, 1].astype(float).tolist()
        data_dict['bus3'] = data.values[:, 2].astype(float).tolist()

        return data_dict        
    
    def inter_data(self, data_low, mode='linear'):
        data_high = {}
        for key in data_low.keys():
            bus_low = data_low[key]
            x_low = np.linspace(0, len(bus_low)-1, len(bus_low))
            x_high = np.linspace(0, len(bus_low)-1, len(bus_low)*self.scale)
            # print(x_low)
            # print(x_high)
            f = inter.interp1d(x_low, bus_low, kind=mode)
            bus_high = f(x_high)
            data_high[key] = bus_high
        return data_high
    
    def __getitem__(self, index):
        low_data = torch.stack([
            torch.tensor(self.data_low['bus1'][index*self.time_step:(index*self.time_step+self.window_size)]),
            torch.tensor(self.data_low['bus2'][index*self.time_step:(index*self.time_step+self.window_size)]),
            torch.tensor(self.data_low['bus3'][index*self.time_step:(index*self.time_step+self.window_size)]),
        ])

        high_data = torch.stack([
            torch.tensor(self.data_high['bus1'][index*self.time_step*5:(index*self.time_step+self.window_size)*5]),
            torch.tensor(self.data_high['bus2'][index*self.time_step*5:(index*self.time_step+self.window_size)*5]),
            torch.tensor(self.data_high['bus3'][index*self.time_step*5:(index*self.time_step+self.window_size)*5]),
        ]) 


        high_high_data = torch.stack([
            torch.tensor(self.data_hhigh['bus1'][index*self.time_step*10:(index*self.time_step+self.window_size)*10]),
            torch.tensor(self.data_hhigh['bus2'][index*self.time_step*10:(index*self.time_step+self.window_size)*10]),
            torch.tensor(self.data_hhigh['bus3'][index*self.time_step*10:(index*self.time_step+self.window_size)*10]),
        ]) 
        
        return low_data.float()/1000000.0, high_data.float()/1000000.0, high_high_data.float()/1000000.0
    
    def __len__(self):
        return (len(self.data_low['bus1'])-self.window_size) // self.time_step

def build_dataloader(low_path, high_path, high_high_path, window_size=10000, time_step=1000, batch_size=16, shuffle=False, is_train=True):
    dataset = TimeSeriesDataset(low_path, high_path, high_high_path, window_size=window_size, time_step=time_step, scale=10, is_train=is_train)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

class InferDataset():
    """
        滑窗推理时用
    """
    def __init__(self, path_low, path_high=None, scale=10, window_size=10000, time_step=1000):
        self.data_low = self.read_file(path_low)
        if path_high:
            self.data_high = self.read_file(path_high)
        else:
            self.data_high = None
        self.window_size = window_size
        self.time_step = time_step
        self.scale = scale

        # self.data_low = self.inter_data(self.data_low)
        self.time_len = len(self.data_low['bus1'])

        if (len(self.data_low['bus1'])-window_size) % time_step == 0:
            self.data_len = (len(self.data_low['bus1'])-window_size) // time_step
        else:
            self.data_len = (len(self.data_low['bus1'])-window_size) // time_step + 1

        self.pred_data = {}
        self.countlist = []
        self.index = 0
        self.begin = 0
        self.end = 0
        self.init_pred()


        print("dataset info: ")
        print("low data:", path_low)
        print("high data:", path_high)
        print("window size:", window_size)
        print("time step:", time_step)
        print("low data numbers:", len(self.data_low['bus1']))
        if path_high:
            print("high data numbers:", len(self.data_high['bus1']))
        print("dataset len:", self.data_len)
        
    def init_pred(self):
        self.countlist = np.array([0.0]*self.time_len*self.scale)
        self.pred_data = {'bus1': np.array([0.0]*self.time_len*self.scale),
                            'bus2': np.array([0.0]*self.time_len*self.scale),
                            'bus3': np.array([0.0]*self.time_len*self.scale)}
        self.index = 0

    def read_file(self, file_name):
        data_dict = {'bus1': [], 
                    'bus2': [],
                    'bus3': []}
        data = pd.read_csv(file_name, header=None)
        data_dict['bus1'] = data.values[:, 0].astype(float).tolist()
        data_dict['bus2'] = data.values[:, 1].astype(float).tolist()
        data_dict['bus3'] = data.values[:, 2].astype(float).tolist()

        return data_dict

    def __len__(self):
        return self.data_len

    def gt(self):
        return self.data_high
    
    def __iter__(self):
        return self
    
    def get_window_data(self, begin, end):
        low_data = torch.stack([
            torch.tensor(self.data_low['bus1'][begin:end]),
            torch.tensor(self.data_low['bus2'][begin:end]),
            torch.tensor(self.data_low['bus3'][begin:end]),
        ])
        return low_data.float() / 1000000.0

    def __next__(self):
        if self.index < self.data_len-1:
            self.begin = self.index*self.time_step
            self.end = self.index*self.time_step+self.window_size
            self.index += 1
            # print(self.begin, self.end, self.time_len)
            return self.get_window_data(self.begin, self.end)
        elif self.index ==  self.data_len-1:
            self.end = self.time_len
            self.begin = self.time_len - self.window_size
            self.index += 1
            # print(self.begin, self.end, self.time_len)
            return self.get_window_data(self.begin, self.end)
        else:
            raise StopIteration
    
    def update(self, pred_high):
        self.countlist[self.begin*self.scale:self.end*self.scale] += 1
        self.pred_data['bus1'][self.begin*self.scale:self.end*self.scale] += pred_high[0]
        self.pred_data['bus2'][self.begin*self.scale:self.end*self.scale] += pred_high[1]
        self.pred_data['bus3'][self.begin*self.scale:self.end*self.scale] += pred_high[2]


    def prediction(self): 
        for key in self.pred_data.keys():
            self.pred_data[key] = self.pred_data[key] / self.countlist * 1000000.0
        return self.pred_data


if __name__ == "__main__":
    infer_dataset = InferDataset("data/Valid_1Hz.csv", "data/Valid_5Hz.csv")
    for data in tqdm(infer_dataset):
        print(data.shape)
        pass