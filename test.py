# -*- coding: utf-8 -*-
"""
Created on  2021/10/13 21:41

@author: shengrihui
"""

import pandas as pd
import os
from collections import defaultdict
from pprint import pprint
import json
from pyecharts.charts import Geo, Timeline
from pyecharts import options as opts
import time
import matplotlib.pyplot as plt

def read_data(mode='train'):
    """
    读取csv数据
    :param mode:默认读train.csv
    :return:
    {  mmsi:{ timestamp:[lon,lat,Sog,Cog]}  }
    """
    data_csv = pd.read_csv('%s.csv' %mode)
    data_dict = defaultdict(dict)
    timestamp = data_csv['timestamp']
    lat = data_csv['lat']
    lon = data_csv['lon']
    mmsi = data_csv['mmsi']
    Sog=data_csv['Sog']
    Cog=data_csv['Cog']
    
    for i in timestamp.index:
        data_dict[mmsi[i]][timestamp[i]] = [lon[i], lat[i],Sog[i],Cog[i]]
    #pprint(data_dict)
    return data_dict

class HTML():
    def __init__(self,mode='train'):
        self.data_dict=datas[mode]
        self.mode=mode
        self.data_pair=self.make_json()
    def make_json(self):
        """
        生成绘制Geo时候所需要的json文件
        :param mode:
        :return: 绘制GEO时需要的data_pair数据
        """
        data_pair = defaultdict(list)
        for m, time_value in self.data_dict.items():
            js = {}
            for t, v in time_value.items():
                js[str(t)] = [v[0],v[1]]
                data_pair[m].append([str(t),  [v[0],v[1]]])
            with open(f'json/{self.mode}_{m}.json', 'w', encoding='utf-8') as f:
                f.write(json.dumps(js))
        return data_pair

    def Geo_(self,i):
        geo = (
            Geo(init_opts=opts.InitOpts(width="1600px", height="900px",
                                        ))
                .add_schema(maptype="china",
                            label_opts=opts.LabelOpts(is_show=False))
                
                .add_coordinate_json(f'json/{self.mode}_{i}.json')
                
                .add(series_name="line", data_pair=self.data_pair[i])
            
            # .set_global_opts(title_opts=opts.TitleOpts(title="美国{}年震级统计".format(self.data_year)),
            #                  visualmap_opts=opts.VisualMapOpts(max_=max_)
            #                  )
            
            # .render(f"1.html")
        )
        return geo
    

    def html(self):
        tl = Timeline(init_opts=opts.InitOpts(width="1600px", height="900px"))
        for i in self.data_dict.keys():
             try:
                geo = self.Geo_(i)
                tl.add(geo, "{}".format(i))
                print(i, "完成")
             except:
                print(i, "未完成")
        tl.render(f"html/{self.mode}_line.html")

def vis_html():
    global train_csv,test_csv,train2_csv,datas
    train_csv = read_data('train')
    test_csv = read_data('test')
    train2_csv = read_data('train2')
    datas = {'train': train_csv, 'test': test_csv, 'train2': train2_csv}
    
    for m in datas:
        data_html=HTML(m)
        data_html.html()

#用matpltlib散点图可视化，不太成功
# def plt_vis():
#     mmsi_list=train_csv.keys()
#     for mm in mmsi_list:
#         print(mm)
#         train_mm=train_csv[mm]
#         train2_mm=train2_csv[mm]
#         x=[]
#         y=[]
#         x2=[]
#         y2=[]
#         for t,(lon,lat,s,c) in train_mm.items():
#             x.append(lon)
#             y.append(lat)
#         for t,(lon,lat,s,c) in train2_mm.items():
#             x2.append(lon)
#             y2.append(lat)
#         #plt.figure(figsize=(80, 80), dpi=80)
#         #plt.subplot(3, 1, 1)
#         plt.figure()
#         plt.scatter(x,y,s=2,c='blue')
#         plt.title('train')
#         plt.show()
#         plt.savefig('vis/train_%s.png'%mm)
#         plt.figure()
#         # plt.subplot(3,1, 2)
#         plt.scatter(x2, y2, s=2, c='red')
#         plt.title('train2')
#         plt.subplot(3,1, 3)
#         plt.show()
#         plt.savefig('vis/train2_%s.png'%mm)
#         plt.figure()
#         plt.scatter(x,y,s=0.5,c='blue',alpha=0.4,label='train')
#         plt.scatter(x2, y2, s=0.5, c='red',alpha=0.4,label='train2')
#         plt.title('both')
#         plt.legend()
#         plt.show()
#         plt.savefig('vis/%s.png' % mm)
#         #break
#         #exit(0)


def time2str(t):
    timeArray = time.localtime(t)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime

def time_data(mode='train'):
    data_csf=pd.read_csv('%s.csv'%mode)
    data_time=data_csf['timestamp']
    data_mmsi=data_csf['mmsi']
    r,c=data_csf.shape
    start=0
    time_dict=defaultdict(list)
    while start<r:
        mm=data_mmsi[start]
        end=start+1
        while end<r:
            if end==r-1 or \
                    data_mmsi[end]!=data_mmsi[start] or \
                    int(data_time[end])-int(data_time[end-1])>1000:
                t0=time2str(int(data_time[start]))
                t2 = time2str(int(data_time[end]))
                t1 = time2str(int(data_time[end-1]))
                #print(mm,t0,t1,end-start,end-1,start)
                time_dict[mm].append([t0,t1,end-start])
                break
            end+=1
        #print()
        start=end
    return time_dict

def compare_train_test_time():
    train_time=time_data()
    print('test')
    test_time=time_data('test')
    for i in range(87,36,-1):
        print(i)
        print('train')
        for j in train_time[i]:
            print(j)
        print('test')
        for j in test_time[i]:
            print(j)
    
def split_train_last6():
    import csv
    with open('train.csv', 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
    out=[['mmsi','lat','lon']]
    pre=0
    for idx in range(1,len(result)):
        mm=int(result[idx][0])
        if mm>27 and mm!=pre :
            pre=mm
            for i in range(idx-6,idx):
                out.append(result[i][:3])
    for i in range(len(result) - 6, len(result)):
        out.append(result[i][:3])

    with open('result.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(out)


def prepare_dataset():
    import pandas as pd
    from itertools import chain
    from collections import defaultdict
    from pprint import pprint
    train_csv = pd.read_csv('train.csv')
    mmsi = train_csv['mmsi']
    lat = train_csv['lat']
    lon = train_csv['lon']
    sog = train_csv['Sog']
    cog = train_csv['Cog']
    mmsi_dict = defaultdict(list)
    for i in range(len(mmsi)):
        mmsi_dict[mmsi[i]].append([lat[i], lon[i], sog[i], cog[i]])
    _data = []
    _pred = []
    
    def spllit_each_mmsi(data,m=0):
        ret = []
        for i in range(len(data) - 45):
            cell_in = data[i:i + 40]
            cell_out = [[data[j][0], data[j][1]] for j in range(i + 40, i + 46)]
            if m<27:
                ret.append([cell_in, cell_out])
            elif m>=27:
                ret.append([cell_in, cell_out,m])
        return ret
    
    for m in mmsi_dict:
        if m < 27:
            _data.extend(spllit_each_mmsi(mmsi_dict[m]))
        elif m >= 27:
            _data.extend(spllit_each_mmsi(mmsi_dict[m][:-46]))
            _pred.extend(spllit_each_mmsi(mmsi_dict[m][-46:],m))
    
    #pprint(_data[0])
    pprint(_pred[-1][1])
    print(lat[-6:])
    
    #exit(0)
    # data_list_4 = list(chain.from_iterable(zip(lat, lon, sog, cog)))
    # data1 = [data_list_4[i:i + n] for i in range(0, len(data_list), 4)]

    # def sepmmsi():
    #     for i in mmsidata:
    #         while i == 1:
    #             list_i = list.append(lat[i])
    #             list_i = list.append(lon[i])
    #             list_i = list.append(sog[i])
    #             list_i = list.append(cog[i])
    #         i = i + 1
    
    # def data46():
    #     for m in list_i:
    #         pass
    #
    # def seqdata():
    #     for i in data46.index:
    #         if i > 40:
    #             list_i.pop(sog)
    #             list_i.pop(cog)
    #     return list_i


if __name__ == '__main__':
    #vis_html()
    #compare_train_test_time()
    #split_train_last6()
    prepare_dataset()
    
    
    

