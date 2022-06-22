import json
import os
import os.path as osp
import re
import xml.etree.ElementTree as ET
from glob import glob
from typing import Dict, List, Tuple
from urllib.request import urlretrieve
# from xml.dom.minidom import parseString
from xml.dom.expatbuilder import parseString

import cv2
import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from colorama import Fore
from dgl.data import DGLDataset

BoxInfo = Tuple[int, int, int, int, str]
BoxCoor = Tuple[int, int, int, int]
labels = ['__ignore__', '_background_']
class_to_idx = {}

def class_2_idx(classname:str):
    return class_to_idx[classname]
    
#TODO create label and class_to_idx

def get_shape_info(jsonp:str) -> List[BoxInfo]:
    """ read json annotation file from labelme and caculus bnbox """
    with open(jsonp, 'r') as f:
        data = json.load(f)
    shapes = data['shapes']
    
    boxes = []
    for shape in shapes:
        #NOTE label
        lbl = shape['label'].strip()
        if lbl not in labels:
            labels.append(lbl)
        
        #NOTE points
        points = np.array(shape['points'])
        xmin = round(points[:, 0].min())
        ymin = round(points[:, 1].min())
        xmax = round(points[:, 0].max())
        ymax = round(points[:, 1].max())
        boxes.append((xmin, ymin, xmax, ymax, lbl))
    
    return boxes


def coordinateCvt2YOLO(size:Tuple[int, int], box:BoxCoor):
    dw = 1. / size[0]
    dh = 1. / size[1]

    x = (box[0] + box[2]) / 2.0         # (xmin + xmax / 2)
    y = (box[1] + box[3]) / 2.0         # (ymin + ymax / 2)

    w = box[2] - box[0]                 # (xmax - xmin) = w
    h = box[3] - box[1]                 # (ymax - ymin) = h

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (round(x, 5), round(y, 5), round(w, 5), round(h, 5))


def from_labelme2yolo(jsonp:str, imgp:str) -> str:
    boxes = get_shape_info(jsonp)
    image = cv2.imread(imgp, 0)
    hei, wid = image.shape
    for box in boxes:
        yolobnb = coordinateCvt2YOLO(size=(wid, hei), box=box[:4])
        idx = class_2_idx(box[4])   #NOTE label
        #TODO

class DKKDGraphDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='karate_club')
        
    def __len__(self): return 1
    def __getitem__(self, i): return self.graph
    
    def process(self):
        
        
        
        
        
        #NOTE download data
        # urlretrieve('https://data.dgl.ai/tutorial/dataset/members.csv', './members.csv')
        # urlretrieve('https://data.dgl.ai/tutorial/dataset/interactions.csv', './interactions.csv')
        nodes_data = pd.read_csv('dataset/Karate_Club/members.csv')
        edges_data = pd.read_csv('dataset/Karate_Club/interactions.csv')
        n_nodes  = nodes_data.shape[0]
        
        node_labels  = torch.tensor(nodes_data['Club'].astype('category').cat.codes.to_list())
        node_features = torch.from_numpy(nodes_data['Age'].to_numpy()).reshape(n_nodes, 1)
        edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())
        edges_src = torch.from_numpy(edges_data['Src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['Dst'].to_numpy())
        
        #NOTE create graph with nodes and edges feature
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=n_nodes)
        self.graph = dgl.to_bidirected(self.graph)      # convert to undirected
        self.graph.ndata['feat'] = node_features        #NOTE learnable
        self.graph.ndata['label'] = node_labels
        self.graph.ndata['node_features'] = nn.Parameter(torch.randn(self.graph.num_nodes(), 10))

        self.graph.edata['weight'] = edge_features

        #NOTE If your dataset is a node classification dataset, 
        #! you will need to assign masks indicating whether 
        #! a node belongs to training, validation, and test set.

        n_train = int(n_nodes * 0.6)
        n_val  = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask  = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask  = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        
        