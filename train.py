import torch
import torch.optim as optim
from tqdm import tqdm
from datasets import build_dataloader, InferDataset
from models.SR import SRNet
from utils import MAAPE_Error, AverageMeter, MAAPELoss, save_result

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    maape = AverageMeter()
    hh_maape = AverageMeter()

    maape_criterion = MAAPELoss()

    index = 0
    for low_data, high_data, high_high_data in tqdm(train_loader):
        low_data, high_data, high_high_data =  low_data.cuda(), high_data.cuda(), high_high_data.cuda()
        # print(low_data.shape, high_data.shape)
        pred_high, pred_high_high = model(low_data)
        
        hh_loss = criterion(pred_high_high, high_high_data)
        h_loss = criterion(pred_high, high_data)

        hh_maape_loss = maape_criterion(pred_high_high, high_high_data)
        h_maape_loss = maape_criterion(pred_high, high_data)

        maape.update(h_maape_loss.item())
        hh_maape.update(hh_maape_loss.item())

        loss = 0.5*hh_loss + 0.5*h_loss + 0.5*hh_maape_loss + 0.5*h_maape_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        index = index + 1

    return maape.avg, hh_maape.avg

def eval(model, valid_loader):
    # l1loss = AverageMeter()
    maape = AverageMeter()
    hh_maape = AverageMeter()
    model.eval()

    criterion1 = torch.nn.SmoothL1Loss()
    criterion2 = MAAPELoss()
    
    for low_data, high_data, high_high_data in tqdm(valid_loader):
        low_data, high_data, high_high_data = low_data.cuda(), high_data.cuda(), high_high_data.cuda()
        with torch.no_grad():
            pred_high, pred_high_high = model(low_data)
        h_loss = criterion2(pred_high, high_data)
        hh_loss = criterion2(pred_high_high, high_high_data)
        maape.update(h_loss.item())
        hh_maape.update(hh_loss.item())
    return maape.avg, hh_maape.avg

def infer_all(model):
    model.eval()
    infer_dataset = InferDataset("data2/Valid_1Hz.csv", "data2/Valid_10Hz.csv", window_size=window_size, time_step=time_step)
    for low_data in tqdm(infer_dataset):
        low_data = low_data.cuda().unsqueeze(0)
        with torch.no_grad():
            _, pred_high = model(low_data)
        infer_dataset.update(pred_high.cpu().squeeze(0).numpy())
    result = infer_dataset.prediction()
    maape = MAAPE_Error(result, infer_dataset.gt())
    print("val maape:", maape)
    save_result(result, "val.csv")

    infer_dataset = InferDataset("data2/Test_low.csv", window_size=window_size, time_step=time_step)
    for low_data in tqdm(infer_dataset):
        low_data = low_data.cuda().unsqueeze(0)
        with torch.no_grad():
            _, pred_high = model(low_data)
        infer_dataset.update(pred_high.cpu().squeeze(0).numpy())
    
    save_result(infer_dataset.prediction(), "submission.csv")

    return maape

if __name__ == "__main__":
    window_size = 2000
    time_step = 200
    train_loader = build_dataloader("data2/Train_1Hz.csv", "data/Train_5Hz.csv", "data2/Train_10Hz.csv", window_size=window_size, time_step=time_step, batch_size=64, shuffle=True, is_train=True)
    valid_loader = build_dataloader("data2/Valid_1Hz.csv", "data/Valid_5Hz.csv", "data2/Valid_10Hz.csv", window_size=window_size, time_step=time_step, batch_size=128, shuffle=False, is_train=False)
    model = SRNet(in_len=window_size, out_len1=window_size*5, out_len2=window_size*10)

    model.cuda()

    criterion = MAAPELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    best_ape = 1.0

    for epoch in range(500):
        maape, hh_maape = train_epoch(model, train_loader, criterion, optimizer)
        eval_maape, eval_hh_maape = eval(model, valid_loader)
        print("epoch: {}, train 5Hz: {}, train 10Hz: {}, valid 5Hz: {}, valid 10Hz: {}".format(epoch, maape, hh_maape, eval_maape, eval_hh_maape))
        
        if epoch > 30:
            if eval_hh_maape < best_ape:
                best_ape = eval_hh_maape            
                infer_all(model)
                torch.save(model.state_dict(), 'state_dict-2.pth')