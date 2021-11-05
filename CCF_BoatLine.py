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
from tqdm import tqdm


class Config():
    # 输入维度4，精度纬度速度角度
    input_size = 4
    # 输出维度2，精度纬度
    out_size = 2
    batch_size = 4
    hidden_size = 30
    num_layers = 3
    # 预测长度10
    out_len = 6
    # 输入长度
    input_len = 40
    lr = 0.01
    start_epoch = 0
    epochs = 15
    pt_path = 'model.pt'
    best = 0


config = Config()


class CCFData(Dataset):
    _data = []
    _pred = []
    
    @classmethod
    def read_csv(cls):
        with open('train.csv', 'r') as f:
            reader = csv.reader(f)
            csv_data = list(reader)  # 将数据空间转换成嵌套列表
            # 遍历列表（跳过表头），将列表里的数据转换成浮点数，便于后续处理
            l = config.out_len + config.input_len
            i = 1
            print(len(csv_data))
            while i < len(csv_data):
                if i + l > len(csv_data):
                    break
                if int(csv_data[i + l][5]) - int(csv_data[i][5]) < 100:  #:and \
                    # csv_data[i+l][0]==csv[i][0]
                    data_cell_in = [[float(csv_data[i + j][k]) for k in range(1, 5)] for j in
                                    range(config.input_len )]
                    data_cell_out = [[float(csv_data[i + j][0]), float(csv_data[i + j][1])] for j in
                                     range(config.input_len + 1, l + 1)]
                    
                    cls._data.append([data_cell_in, data_cell_out])
                    i += 1
                else:
                    if int(csv_data[i][0]) - int(csv_data[i - 1][0]) == 1 and \
                            int(csv_data[i][0]) >= 27:
                        data_cell_in = [[float(csv_data[i + j][k]) for k in range(1, 5)] for j in
                                        range(config.input_len + 1)]
                        data_cell_out = [[float(csv_data[i + j][1]), float(csv_data[i + j][2])] for j in
                                         range(config.input_len + 1, l + 1)]
                        cls._pred.append([data_cell_in, data_cell_out])
                        i += l
                    else:
                        i += 1
    
    def __init__(self, mode='train'):
        super(CCFData, self).__init__()
        # if self._data == []:
        #     self.read_csv()
        # self.n = len(self._data)
        # if mode == 'train':
        #     self.data = self._data[:int(0.8 * self.n)]
        # elif mode == 'eval':
        #     self.data = self._data[int(0.8 * self.n):]
        # elif mode == 'pred':
        #     self.data = self._pred
        self.data=[]
        l = config.out_len + config.input_len
        with open("train.csv",'r') as f:
            reader = csv.reader(f)
            csv_data = list(reader)
            for i in range(1,60):
                data_cell_in = [[float(csv_data[i + j][k]) for k in range(1, 5)] for j in
                                range(config.input_len )]
                data_cell_out = [[float(csv_data[i + j][0]), float(csv_data[i + j][1])] for j in
                                 range(config.input_len + 1, l + 1)]
    
                self.data.append([data_cell_in, data_cell_out])
        
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])
    
    def __len__(self):
        return len(self.data)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, input_len, out_len, out_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_len = input_len
        self.out_len = out_len
        self.out_size = out_size
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(self.input_len * self.hidden_size, self.input_len),
            nn.Sigmoid(),
            nn.Linear(self.input_len, self.out_len * self.out_size)
        )
    
    #def forward(self, x, hidden):
        #out, (h_n, c_n) = self.lstm(x, hidden)
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = torch.tanh(out)
        bs = out.shape[0]
        out = out.reshape(bs, -1)
        out = self.fc(out)
        out = out.reshape(bs,self.out_len, -1)
        return out
    
    # def init_h_c(self, bs=config.batch_size):
    #     h = torch.zeros(self.num_layers, bs, self.hidden_size)
    #     c = torch.zeros(self.num_layers, bs, self.hidden_size)
    #     return h, c


model = Model(
    input_size=config.input_size,
    hidden_size=config.hidden_size,
    num_layers=config.num_layers,
    batch_size=config.batch_size,
    input_len=config.input_len,
    out_len=config.out_len,
    out_size=config.out_size
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
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
    
    for idx, (data_in, data_out) in enumerate(tbar):
        data_in, data_out = data_in.to(device), data_out.to(device)
        # h, c = model.init_h_c()
        # h, c = h.to(device), c.to(device)
        optimizer.zero_grad()
        #out = model(data_in, (h, c))
        out = model(data_in)
        loss = criterion(out, data_out)
        loss.backward()
        optimizer.step()
        tbar.set_description('Epoch:%d  loss:%.6f ' % (epoch, loss.item()))

def test():
    score = 0
    total_score = 0
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
            #out = model(data_in, (h, c))
            out = model(data_in)
            loss = criterion(out, data_out)
            s = 1 - loss.item()
            total_score += s
            score = total_score / ((idx+1) * config.batch_size)
            tbar.set_description('Val: loss:%.6f score:%.6f' % (loss.item(), score))
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
    out_csv=[['mmsi','lat','lon']]
    with torch.no_grad():
        tbar = tqdm(pred_loader)
        for idx, (data_in, data_out) in enumerate(tbar):
            mmsi='28'
            data_in, data_out = data_in.to(device), data_out.to(device)
            out = model(data_in)
            tbar.set_description('Pred...')
            out=torch.squeeze(out)
            for lat,lon in out:
                lat=str(lat.item())
                lon = str(lon.item())
                out_csv.append([mmsi,lat,lon])
    writer_csv(out_csv)
            
    

if __name__ == '__main__':
    # a = torch.rand(config.input_len, config.batch_size, config.input_size)
    # y = torch.ones(config.out_len, config.batch_size, config.out_size)
    # a = a.to(device)
    # a = torch.rand(config.input_len, 1, config.input_size)
    #
    # h, c = model.init_h_c(1
    #
    #                       )
    # h, c = h.to(device), c.to(device)
    # out = model(a, (h, c))
    # print(out.shape)
    # loss = criterion(out, y)
    
    train_data=CCFData()
    # a=len(train_data)
    test_data=CCFData()
    # b=len(test_data)
    pred_data=CCFData()
    # c=len(pred_data)
    # print(a,b,c,a+b+c)
    
    for epoch in range(config.start_epoch,config.start_epoch+config.epochs):
        break
        train(epoch)
        if (epoch-config.start_epoch)%3==0:
            best=test()
            if best>config.best:
                torch.save(model.state_dict(), config.pt_path)
    pred()