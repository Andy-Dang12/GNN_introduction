import json, re, os
import os.path as osp
from glob import glob
from colorama import Fore
from typing import List
from cv2 import sort
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from dkkd_create_graph import get_shape_info, class_2_idx, classes
import numpy as np
import torch


root = '/home/agent/Documents/graph/GNN/dataset/DKKD'

def read_and_count():
    jsons = glob(osp.join(root, '*.json'))
    boxes = []
    for js in jsons:
        boxes.extend(get_shape_info(js))
        
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

def visualize():
    df = pd.read_csv('dataset/DKKD/classes_unbalance.csv', encoding='utf-8')
    df = df.sort_values(['class_name'], key=lambda x:x)
    cls = df['class_name']
    num = df['num']
    
    fig = plt.figure(figsize = (10, 5))

    plt.bar(cls, num, color ='maroon', width = 0.3)
    # plt.xlabel("classes")
    # plt.ylabel("num nodes")
    plt.xticks(range(len(cls)), labels=cls, rotation=-90, fontsize=10)
    plt.show()
    
    
visualize()


def cacu_alpha() -> torch.Tensor:
    df = pd.read_csv('dataset/DKKD/classes_unbalance.csv', encoding='utf-8')
    classes = df['class_name']
    num = df['num']
    stat_cls = []
    for cls, n_node in zip(classes, num):
        idx = class_2_idx(cls)
        stat_cls.append((idx, n_node))
        
    stat_cls = np.array(sorted(stat_cls, key=lambda x:x[0]))
    tong = 1.*stat_cls[:,1].sum()
    alpha = stat_cls[:,1]/tong
    return torch.from_numpy(alpha)
from dgl import DGLGraph

def cacu_alpha(g:DGLGraph) -> torch.Tensor:
    
    ...
# alpha = cacu_alpha()
# print(alpha)