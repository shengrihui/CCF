# -*- coding: utf-8 -*-
"""
Created on  2021/10/26 20:16

@author: shengrihui
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import csv
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from MyCosineAnnealingWarmRestarts import MyCosineAnnealingWarmRestarts as MyCos


class Config():
    normalization = {
        'lat': {'min': 30.335862, 'max': 30.583262},
        'lon': {'min': 121.652935, 'max': 121.881350},
        'Cog': {'min': 0.000000, 'max': 359.000000},
        'Sog': {'min': 1.100000, 'max': 36.000000}
    }

    input_size = 4  # 输入维度4，精度纬度速度角度
    out_size = 2  # 输出维度2，精度纬度
    batch_size = 4
    hidden_size = 30
    num_layers = 3
    input_len = 12  # 输入长度
    out_len = 6  # 预测长度10
    data_len = out_len + input_len

    lr = 0.01
    momentum = 0.9
    start_epoch = 200
    epochs = 100
    save_dir = '11'
    
    model_path = f'{save_dir}/model/'
    best_model = f'{model_path}/model_34.pt'
    logs = f'{save_dir}/logs/'
    result_path = f'{save_dir}/result/'
    test_best = 0.9903486290068979
    pred_best = 0.9777078582010242


config = Config()
if not os.path.exists(config.save_dir):
    os.mkdir(config.save_dir)
    os.makedirs(config.logs)
    os.makedirs(config.model_path)
    os.makedirs(config.result_path)


# 数据归一化和返回数据
class Data_Normalization():
    @staticmethod
    def normalization(data, data_name):
        m1 = config.normalization[data_name]['min']
        m2 = config.normalization[data_name]['max']
        return (data - m1) / (m2 - m1)
    
    @staticmethod
    def n2data(out, data_name):
        m1 = config.normalization[data_name]['min']
        m2 = config.normalization[data_name]['max']
        return out * (m2 - m1) + m1


class CCFData(Dataset):
    _data = []
    _pred = []
    
    @classmethod
    def read_csv(cls):
        train_csv = pd.read_csv('data/train.csv')
        mmsi = train_csv['mmsi']
        lat = train_csv['lat']
        lon = train_csv['lon']
        sog = train_csv['Sog']
        cog = train_csv['Cog']
        time = train_csv['timestamp']
        mmsi_dict = defaultdict(list)
        for i in range(len(mmsi)):
            mmsi_dict[mmsi[i]].append(
                [lat[i], lon[i], sog[i], cog[i], time[i]])
        
        def spllit_each_mmsi(data, m=0):
            data_name = ['lat', 'lon', 'Sog', 'Cog']
            ret = []
            i = 0
            while i < len(data):
                try:
                    if abs(data[i + config.data_len + 1][4] - data[i + config.data_len][4]) > 100:
                        i += config.data_len
                        continue
                except:
                    if i + config.data_len > len(data):
                        break
                cell_in = [[Data_Normalization.normalization(data[k][j], data_name[j]) for j in range(4)] for k in
                           range(i, i + config.input_len)]
                cell_out = [[Data_Normalization.normalization(data[j][0], 'lat'),
                             Data_Normalization.normalization(data[j][1], 'lon')]
                            for j in range(i + config.input_len, i + config.data_len)]
                
                if m < 27:
                    ret.append([cell_in, cell_out])
                elif m >= 27:
                    ret.append([cell_in, cell_out, m])
                i += 1
            return ret
        
        for m in mmsi_dict:
            if m < 27:
                cls._data.extend(spllit_each_mmsi(mmsi_dict[m]))
            elif m >= 27:
                cls._data.extend(spllit_each_mmsi(
                    mmsi_dict[m][:-config.data_len]))
                cls._pred.extend(spllit_each_mmsi(
                    mmsi_dict[m][-config.data_len:], m))
    
    def __init__(self, mode='train'):
        super(CCFData, self).__init__()
        if self._data == []:
            self.read_csv()
        self.n = len(self._data)
        self.mode = mode
        if mode == 'train':
            self.data = self._data[:int(0.8 * self.n)]
        elif mode == 'test':
            self.data = self._data[int(0.8 * self.n):]
        elif mode == 'pred':
            self.data = self._pred
    
    def __getitem__(self, idx):
        if self.mode != 'pred':
            return torch.tensor(self.data[idx][0]).double(), torch.tensor(self.data[idx][1]).double()
        else:
            return torch.tensor(self.data[idx][0]).double(), torch.tensor(self.data[idx][1]).double(), self.data[idx][2]
    
    def __len__(self):
        return len(self.data)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, input_len, out_len, out_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_len = input_len
        self.out_len = out_len
        self.out_size = out_size
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(self.input_len * self.hidden_size, 60),
            nn.Sigmoid(),
            nn.Linear(60, self.out_len * self.out_size)
        )
    
    # def forward(self, x, hidden):
    # out, (h_n, c_n) = self.lstm(x, hidden)
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = torch.tanh(out)
        bs = out.shape[0]
        out = out.reshape(bs, -1)
        out = self.fc(out)
        out = out.reshape(bs, self.out_len, -1)
        return out
    
    # def init_h_c(self, bs=config.batch_size):
    #     h = torch.zeros(self.num_layers, bs, self.hidden_size)
    #     c = torch.zeros(self.num_layers, bs, self.hidden_size)
    #     return h, c


model = Model(
    input_size=config.input_size,
    hidden_size=config.hidden_size,
    num_layers=config.num_layers,
    input_len=config.input_len,
    out_len=config.out_len,
    out_size=config.out_size
)


class My_loss(nn.Module):
    def __init__(self):
        super(My_loss, self).__init__()
        self.s = []
        self.score = 0
        self.num = 0
        self.total_loss = 0
    
    def forward(self, y_hat, y):
        for y1, y2 in zip(y_hat, y):
            loss = F.mse_loss(y, y_hat)
            self.num += 1
            self.total_loss += loss.item()
            if self.num % 60 == 0:
                s = 1 / (self.total_loss + 1)
                self.total_loss = 0
                self.s.append(s)
        if self.s != []:
            self.score = sum(self.s) / len(self.s)
            return self.score
        else:
            return 0


criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, 'max', 0.1, 10)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,35,60], gamma=0.4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3,T_mult=2,eta_min=0)
scheduler = MyCos(optimizer, T_0=5, T_mult=2,factor=0.9, eta_min=0.000001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

model.double()
model.to(device)
criterion.to(device)
if os.path.exists(config.best_model):
    model.load_state_dict(torch.load(config.best_model))


def train(epoch):
    train_loader = DataLoader(dataset=train_data,
                              shuffle=True,
                              num_workers=1,
                              drop_last=True,
                              batch_size=config.batch_size)
    tbar = tqdm(train_loader)
    total_loss = 0
    for idx, (data_in, data_out) in enumerate(tbar):
        data_in, data_out = data_in.to(device), data_out.to(device)
        optimizer.zero_grad()
        out = model(data_in)
        loss = criterion(out, data_out)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        tbar.set_description('Epoch %d : loss:%.6f total_loss:%.6f' % (
            epoch, loss.item(), total_loss))
    writer.add_scalar('train_loss', total_loss, epoch)


def test(epoch):
    my_loss = My_loss()
    my_loss.to(device)
    test_loader = DataLoader(dataset=test_data,
                             shuffle=False,
                             drop_last=False,
                             num_workers=1,
                             batch_size=config.batch_size)
    with torch.no_grad():
        tbar = tqdm(test_loader)
        for idx, (data_in, data_out) in enumerate(tbar):
            data_in, data_out = data_in.to(device), data_out.to(device)
            out = model(data_in)
            my_score = my_loss(data_out, out)
            loss = criterion(out, data_out)
            tbar.set_description('Val: loss:%.6f score:%.6f' %
                                 (loss.item(), my_score))
    writer.add_scalar('test_score', my_score, epoch)
    return my_score


def pred(best_pred, epoch):
    def write_csv(out, path):
        with open(path, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(out)
    
    pred_loader = DataLoader(dataset=pred_data,
                             shuffle=False,
                             drop_last=False,
                             num_workers=1,
                             batch_size=1)
    out_csv = [['mmsi', 'lat', 'lon']]
    total_loss = 0
    with torch.no_grad():
        tbar = tqdm(pred_loader)
        for idx, (data_in, data_out, mmsi) in enumerate(tbar):
            data_in, data_out = data_in.to(device), data_out.to(device)
            out = model(data_in)
            out = torch.squeeze(out)
            loss = criterion(out, data_out)
            total_loss += loss.item()
            mse = total_loss / config.out_len
            score = 1 / (mse + 1)
            tbar.set_description('Pred score:%f...' % score)
            for lat, lon in out:
                lat = str(Data_Normalization.n2data(lat.item(), 'lat'))
                lon = str(Data_Normalization.n2data(lon.item(), 'lon'))
                out_csv.append([str(mmsi.item()), lat, lon])
    # print('pred score：', score)
    writer.add_scalar('pred_score', score, epoch)
    write_csv(out_csv, f'{config.result_path}/result_{epoch}_{score}.csv')
    if score > best_pred:
        print('save result..')
        write_csv(out, f'{config.result_path}/result{epoch}.csv')
    return score


if __name__ == '__main__':
    train_data = CCFData('train')
    test_data = CCFData('test')
    pred_data = CCFData('pred')
    writer = SummaryWriter(config.logs)
    

    test_best = config.test_best
    pred_best = config.pred_best
    p_score = pred_best
    for epoch in range(config.start_epoch, config.start_epoch + config.epochs):
        # break
        writer.add_scalar('lr', optimizer.state_dict()[
            'param_groups'][0]['lr'], epoch)
        train(epoch)
        
        is_save =False
        test_score = test(epoch)
        if test_score > test_best:
            test_best = test_score
            is_save=True
       
        p = pred(p_score, epoch)
        if p > p_score:
            p_score = p
            is_save=True
        
        if is_save:
            torch.save(model.state_dict(), f'{config.model_path}/model_{epoch}.pt')
            print("Save model...")

        # scheduler.step(p_score)
        scheduler.step()
    print('test_best', test_best)
    print('best_pred', p_score)
    
    torch.save(model.state_dict(), f'{config.model_path}/model_last.pt')
    print("Save model...")
