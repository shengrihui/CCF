# -*- coding: utf-8 -*-
"""
Created on  2021/10/26 20:16

@author: shengrihui
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import csv
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Config():
    # 输入维度4，精度纬度速度角度
    input_size = 4
    # 输出维度2，精度纬度
    out_size = 2
    batch_size = 4
    hidden_size = 30
    num_layers = 3
    out_len = 6  # 预测长度10
    input_len = 40  # 输入长度
    data_len = out_len + input_len
    
    lr = 0.000001
    start_epoch = 80
    epochs = 10
    save_dir = '02'
    pt_path = f'{save_dir}/model.pt'
    logs = f'{save_dir}/logs/'
    best = 0.9947640643509494


config = Config()
if not os.path.exists(config.save_dir):
    os.mkdir(config.save_dir)
    os.makedirs(config.logs)


class CCFData(Dataset):
    _data = []
    _pred = []
    
    @classmethod
    def read_csv(cls):
        train_csv = pd.read_csv('train.csv')
        mmsi = train_csv['mmsi']
        lat = train_csv['lat']
        lon = train_csv['lon']
        sog = train_csv['Sog']
        cog = train_csv['Cog']
        time = train_csv['timestamp']
        mmsi_dict = defaultdict(list)
        for i in range(len(mmsi)):
            mmsi_dict[mmsi[i]].append([lat[i], lon[i], sog[i], cog[i], time[i]])
        
        def spllit_each_mmsi(data, m=0):
            ret = []
            for i in range(len(data) - config.data_len + 1):
                try:
                    if abs(data[i + config.data_len + 1][4] - data[i + config.data_len + 1][4]) > 100:
                        continue
                except:
                    pass
                cell_in = [[data[k][j] for j in range(4)] for k in range(i, i + config.input_len)]
                cell_out = [[data[j][0], data[j][1]] for j in range(i + config.input_len, i + config.data_len)]
                if m < 27:
                    ret.append([cell_in, cell_out])
                elif m >= 27:
                    ret.append([cell_in, cell_out, m])
            return ret
        
        for m in mmsi_dict:
            if m < 27:
                cls._data.extend(spllit_each_mmsi(mmsi_dict[m]))
            elif m >= 27:
                cls._data.extend(spllit_each_mmsi(mmsi_dict[m][:-46]))
                cls._pred.extend(spllit_each_mmsi(mmsi_dict[m][-46:], m))
    
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
            nn.Linear(self.input_len * self.hidden_size, self.input_len),
            nn.Sigmoid(),
            nn.Linear(self.input_len, self.out_len * self.out_size)
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

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model.double()
model.to(device)
criterion.to(device)
if os.path.exists(config.pt_path):
    model.load_state_dict(torch.load(config.pt_path))


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
        # h, c = model.init_h_c()
        # h, c = h.to(device), c.to(device)
        optimizer.zero_grad()
        # out = model(data_in, (h, c))
        out = model(data_in)
        loss = criterion(out, data_out)
        total_loss += loss.item()
        average_loss = total_loss / ((idx + 1) * config.batch_size)
        loss.backward()
        optimizer.step()
        tbar.set_description('Epoch %d : loss:%.6f average_loss:%.6f' % (epoch, loss.item(), average_loss))
    writer.add_scalar('train_loss', average_loss, epoch)


def test(epoch):
    score = 0
    total_loss = 0
    test_loader = DataLoader(dataset=test_data,
                             shuffle=False,
                             drop_last=False,
                             num_workers=1,
                             batch_size=config.batch_size)
    with torch.no_grad():
        tbar = tqdm(test_loader)
        for idx, (data_in, data_out) in enumerate(tbar):
            data_in, data_out = data_in.to(device), data_out.to(device)
            # h, c = model.init_h_c()
            # h, c = h.to(device), c.to(device)
            # out = model(data_in, (h, c))
            out = model(data_in)
            loss = criterion(out, data_out)
            total_loss += loss.item()
            mse = total_loss / ((idx + 1) * config.batch_size * config.out_len)
            score = 1 / (mse + 1)
            tbar.set_description('Val: loss:%.6f score:%.6f' % (loss.item(), score))
    writer.add_scalar('test_score', score, epoch)
    return score


def writer_csv(out):
    with open('result.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(out)


def pred():
    pred_loader = DataLoader(dataset=pred_data,
                             shuffle=False,
                             drop_last=False,
                             num_workers=1,
                             batch_size=1)
    out_csv = [['mmsi', 'lat', 'lon']]
    total_loss = 0
    result = []
    with torch.no_grad():
        tbar = tqdm(pred_loader)
        for idx, (data_in, data_out, mmsi) in enumerate(tbar):
            data_in, data_out = data_in.to(device), data_out.to(device)
            out = model(data_in)
            tbar.set_description('Pred...')
            out = torch.squeeze(out)
            loss = criterion(out, data_out)
            total_loss += loss.item()
            for lat, lon in out:
                result.append([lat.item(), lon.item()])
                lat = str(lat.item())
                lon = str(lon.item())
                out_csv.append([str(mmsi.item()), lat, lon])
    
    ####################################################
    ####################################################
    ####################################################
    mse = total_loss / (len(pred_data) * config.out_len)
    score = 1 / (mse + 1)
    print(score)
    writer_csv(out_csv)
    ####################################################
    ####################################################
    ####################################################
    # writer.add_scalar('pred', average_loss, epoch)
    return result


if __name__ == '__main__':
    train_data = CCFData('train')
    test_data = CCFData('test')
    pred_data = CCFData('pred')
    writer = SummaryWriter(config.logs)

    best = config.best
    for epoch in range(config.start_epoch, config.start_epoch + config.epochs):
        # break
        train(epoch)
        if (epoch - config.start_epoch) % 1 == 0:
            score = test(epoch)
            if score > best:
                best = score
                torch.save(model.state_dict(), config.pt_path)
                print("Save model...")
    pred()
    print('best', best)

    torch.save(model.state_dict(), config.pt_path)
    print("Save model...")

