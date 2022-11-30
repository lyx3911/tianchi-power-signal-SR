import numpy as np
import torch.nn as nn
import torch
import pandas as pd

def MAAPE(pred, label):
    return np.mean(np.arctan(np.abs((label-pred)/(label))))

def MAAPE_Error(pred, label):
    error = 0
    for key in pred.keys():
        error = error+MAAPE(pred[key], label[key])
    return error / len(pred.keys())

def save_result(data, save_path):
    frame = pd.DataFrame({'bus1': data['bus1'], 'bus2': data['bus2'], 'bus3': data['bus3'] })
    # 保留两位小数，不然大小会超
    formater = "{0:.02f}".format
    frame = frame.applymap(formater)
    print("format finished")
    frame.to_csv(save_path, index=False, header=False)
    print("save finished")

class MAAPELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, label, eps=1e-8):
        return torch.mean(torch.arctan(torch.abs((label-pred)/(label+eps) )))

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count