import json, re, os
import os.path as osp
from glob import glob
from colorama import Fore
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
# from dkkd_create_graph import get_shape_info, create_cls_to_idx, class_2_idx


root = '/home/agent/Documents/graph/GNN/dataset/DKKD'

# classes, class_to_idx = create_cls_to_idx('dataset/DKKD/labels.txt')

def read_and_count():
    jsons = glob(osp.join(root, '*.json'))
    boxes = []
    for js in jsons:
        # boxes.extend(get_shape_info(js))
        ...
        
    cls = [box[-1] for box in boxes]

    count = Counter(cls)
    class_name = []
    sl = []
    for c, num in count.items():
        print('{:>40s}'.format(c), Fore.RED, num, Fore.RESET)
        class_name.append(c)
        sl.append(num)


    df = pd.DataFrame({'class_name':class_name,'num':sl},
                    index=pd.RangeIndex(start=1, stop=len(class_name) + 1 ,name='index'))

    df.to_csv('dataset/DKKD/classes_unbalance.csv', encoding='utf-8')


df = pd.read_csv('dataset/DKKD/classes_unbalance.csv', encoding='utf-8')

cls = df['class_name']
num = df['num']
fig = plt.figure(figsize = (10, 5))

plt.bar(cls, num, color ='maroon', width = 0.3)
plt.xlabel("classes")
plt.ylabel("num nodes")
plt.xticks(range(len(cls)), labels=cls, rotation=-90, fontsize=10)
plt.show()