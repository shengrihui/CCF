import pandas as pd
import csv

# ['timestamp','lon', 'lat', 'speed']

# 读取csv文件
with open('train.csv', 'r') as f:
    reader = csv.reader(f)
    result = list(reader)  # 将数据空间转换成嵌套列表
    L = len(result)
    # 遍历列表（跳过表头），将列表里的数据转换成浮点数，便于后续处理
    for i in range(1, L):
        result[i][0] = int(result[i][0])
        result[i][1:5] = list(map(float, result[i][1:5]))
        result[i][5] = int(result[i][5])

# 遍历整个数据空间（跳过表头），判断是否存在空缺，线性插值
# for i in range(2, len(result)):
#     if result[i][5] - result[i-1][5] > 1 and result[i]==result[i-1]:
#         gap = int(result[i][0] - result[i-1][0])
#         interval_s=[  ( result[i][idx] -result[i-1][idx])/gap for idx in range(1,5+1)  ]
#         # interval_lat = (result[i][1]-result[i-1][1])/gap
#         # interval_lon = (result[i][2]-result[i-1][2])/gap
#         # interval_Sog = (result[i][3]-result[i-1][3])/gap
#         # interval_Cog = (result[i][3] - result[i - 1][4]) / gap
#         for k in range(1, gap):
#             interval=[result[0]]
#             for idx in range(1,5+1):
#                 interval.append(interval_s[idx]*k+result[-1][idx])
#             result.append(interval)
#
#                 # result.insert(i-1+k, [result[i-1][0] + k, \
#                 #     interval_long*k + result[i-1][1], \
#                 #     interval_lat*k + result[i-1][2], \
#                 #     interval_speed*k + result[i-1][3]])

out = [result[0], result[1]]
from tqdm import tqdm

for i in tqdm(range(2, len(result))):
    
    if 100 > result[i][5] - result[i - 1][5] > 1 and result[i][0] == result[i - 1][0]:
        gap = int(result[i][5] - result[i - 1][5])
        interval_s = [(result[i][idx] - result[i - 1][idx]) / gap for idx in range(5 + 1)]
        for k in range(1, gap):
            interval = [interval_s[idx] * k + result[i - 1][idx] for idx in range(5 + 1)]
            interval[0] = int(interval[0])
            interval[5] = int(interval[5])
            out.append(interval)
    else:
        out.append(result[i])

# 遍历，判断插值效果
from pprint import pprint

# pprint(out)
for i in out:
    print(i)
    break

# 将插值后的数据写入csv文件
with open('train2.csv', "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 由于我转换数据类型及插值的时候跳过了表头，故不需要再写入表头
    # writer.writerow(["timestamp", "long", "lat",'speed'])   # 先写入columns_name
    writer.writerows(out)  # 写入多行用writerows
