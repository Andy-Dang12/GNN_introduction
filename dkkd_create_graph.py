import json
import os, os.path as osp
import re
from glob import glob
from itertools import chain
from time import time
from typing import Dict, List, Tuple, Union

# import xml.etree.ElementTree as ET
# from xml.dom.minidom import parseString
# from xml.dom.expatbuilder import parseString
import cv2
import numpy as np
import pandas as pd
import torch

from colorama import Fore
from dgl import DGLGraph

from phoBERT import word2vec
from vietOCR import img2word, imgs2words

Number = Union[int, float]
VOCBox = Tuple[Number, Number, Number, Number, str]
VOCCoor = Tuple[Number, Number, Number, Number]
YOLOBox = Tuple[int, float, float, float, float]
YOLOCoor = Tuple[float, float, float, float]

labels = ['__ignore__', '_background_']

def create_cls_to_idx(label_path:str) -> Tuple[list, Dict[str, int]]:
    assert label_path.endswith('.txt')
    with open(label_path, 'r') as f:
        lines = f.read().strip().splitlines()
    classes = sorted([l for l in lines if l not in labels])
    
    with open(label_path, 'w') as f:
        f.write('\n'.join(labels + classes))
        
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

classes, class_to_idx = create_cls_to_idx('dataset/DKKD/labels.txt')

def class_2_idx(classname:str):
    assert classname in class_to_idx.keys(), 'invalid classname: ' + Fore.RED + classname
    return class_to_idx[classname]


def sorted_VOCbox(boxes:List[VOCBox]) -> List[List[VOCBox]]:
    def _get_line_center_x2(box:VOCBox) -> int:
        return box[1] + box[3]
    
    #NOTE sort by ymin/ymax/ycenter from low to high
    boxes = sorted(boxes, key=lambda x:(x[1] + x[3]))
    
    lines = []
    line  = [boxes[0]]
    linecenter_x2 = _get_line_center_x2(boxes[0])
    for box in boxes[1:]:
        if 2*box[1] <= linecenter_x2:    #! thresshold
            line.append(box)
        
        else:
            lines.append(line.copy())
            line = [box]
        linecenter_x2 = _get_line_center_x2(box)
        
    lines.append(line)
    
    def _order_box_xmin(boxes:List[VOCBox]) -> List[VOCBox]:
        return sorted(boxes, key= lambda x: x[0])
    
    return [_order_box_xmin(l) for l in lines]

def get_shape_info(jsonp:str) -> List[VOCBox]:
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
        xmin = points[:, 0].min()
        ymin = points[:, 1].min()
        xmax = points[:, 0].max()
        ymax = points[:, 1].max()
        boxes.append((xmin, ymin, xmax, ymax, lbl))
    return boxes
    # return sorted_VOCbox(boxes)


def coordinateCvt2YOLO(size:Tuple[int, int], box:VOCCoor) -> YOLOCoor:
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


def from_labelme2yolo(jsonp:str, imgp:str) -> List[YOLOBox]:
    r"""
    convert a annotation from labelme to yolo 
    how to use: 
    >>> label = YOLOBox[0]
    >>> coor_feature = YOLOBox[1:]
    """
    boxes = get_shape_info(jsonp)
    image = cv2.imread(imgp, 0)
    hei, wid = image.shape
    lbls = []
    for box in boxes:
        yolobnb = coordinateCvt2YOLO(size=(wid, hei), box=box[:4])
        idx = class_2_idx(box[4])   #NOTE label
        lbls.append((idx, *yolobnb))
        
    return lbls

def save_nodes(idx, label , nodes_path:str, 
               feats:List[List[torch.Tensor]], feats_path:str
               ) -> Tuple[pd.DataFrame, np.ndarray]:
    r"""
    save node data
    index save trong file .csv
    node feature save trong file .npy
    """
    assert nodes_path.endswith('.csv'), 'save node data ở dạng .csv'
    assert feats_path.endswith('.npy'), 'save node feat ở dạng .npy'
    
    node = pd.DataFrame({'Id':chain(*idx), 'label':chain(*label)})
    node.to_csv(nodes_path, encoding='utf-8', index=False)
    
    feats = [tensor1D.numpy() for tensor1D in chain(*feats)]
    assert len(tuple(chain(*label))) == len(tuple(chain(*idx))), 'n_nodes != n_feats'
    assert len(tuple(chain(*idx))) == len(feats), 'n_nodes != n_feats'
    
    feat_2D = np.stack(feats, axis=0)
    np.save(feats_path, feat_2D)
    
    print("node label saved ", nodes_path)
    print("node feat saved ", feats_path)
    return node, feat_2D

def save_edges(edges, path:str):
    src_node, dst_node = edges
    edges_data = pd.DataFrame(
        {'src':src_node,
         'dst':dst_node}
    )
    edges_data.to_csv(path, encoding='utf-8', index=False)
    print('edge data saved ', path)

def is_edge_diff_line(box_src, box_dst, thress=1.0):
    #NOTE so sánh 2 box xem có tạo edge hay k
    xminS, _, xmaxS = box_src[:3]
    xminD, _, xmaxD = box_dst[:3]
    
    if xminS <= xminD <= xmaxS or xminS <= xmaxD <= xmaxS:
        return True 
    else:
        return False

def is_edge_same_line(box_src, box_dst, thress=1.0):
    xminS, _, xmaxS = box_src[:3]
    xminD, _, xmaxD = box_dst[:3]
    
    return

def build_graph(jsonp:str, imgp:str) -> DGLGraph:
    r"""
    from coordinate and word
    """
    image = cv2.imread(imgp, 0)
    hei, wid = image.shape

    boxes    = get_shape_info(jsonp)
    lineboxes = sorted_VOCbox(boxes)
    n_nodes  = len(boxes)
    
    #? đưa 2 function _create_edges và _create_node_feature ra ngoài build_graph 
    
    def _create_a_node_data(box:VOCBox, image:np.ndarray=image, 
                               wid:int=wid, hei:int=hei) -> Tuple[torch.Tensor, int]:
        xmin, ymin, xmax, ymax, lbl = box
        coor_feat = coordinateCvt2YOLO(
            size=(wid, hei), box=(xmin, ymin, xmax, ymax))
        coor_feat = torch.tensor(coor_feat)
        image_word = image[ymin:ymax, xmin:xmax]
        word = img2word(image_word)
        word_embedding = word2vec(word)
        idx_class = class_2_idx(lbl)
        
         #!FIXME cvt 2 same type and 1d-tensor
        return torch.cat((coor_feat, word_embedding)), idx_class
    
    def _create_nodes_data(boxes:List[Union[VOCBox, VOCCoor]], image:np.ndarray=image, 
                           wid:int=wid, hei:int=hei) -> Tuple[List[torch.Tensor], List[int]]:
        r"""
        input 1 dòng gồm nhiều box
        output data(node feature và index_class) của từng box trong dòng đó
        tận dụng predict_batch để tăng performance, mỗi box trong dòng là 1 batch
        """
        coor_feats , imgs, lbls = [], [], []
        
        for box in boxes:
            xmin, ymin, xmax, ymax, lbl = box
            coor_feat = coordinateCvt2YOLO(size=(wid, hei), 
                                           box=(xmin, ymin, xmax, ymax))
            coor_feats.append(torch.tensor(coor_feat))
            imgs.append(image[round(ymin):round(ymax), round(xmin):round(xmax)])    #*copy?
            lbls.append(class_2_idx(lbl))               #*index

        words = imgs2words(imgs, return_prob=False)
        nodes_feats = []
        for ct, word in zip(coor_feats, words):
            word_embedding = word2vec(word)
            node_feat = torch.cat((ct, word_embedding))
            nodes_feats.append(node_feat)
            
        return nodes_feats, lbls
    
    #NOTE create node data include feat and label of each node
    nodes_feats ,lbls, nodes_idx = [], [], []   # List[List]
    start_idx = 0
    for line in lineboxes:
        _nodes_feats, _lbls = _create_nodes_data(line)
        nodes_feats.append(_nodes_feats)            #NOTE list of list of vector node feat
        lbls.       append(_lbls)                   #NOTE list of list of node label

        stop_idx = start_idx + len(_lbls)
        nodes_idx.  append(
            tuple(range(start_idx, stop_idx)))      #NOTE list of tuple of node index
        start_idx = stop_idx
    
    #TEST node index , check đủ số lượng node index và ko bị duplicate
    b, c = [], 0
    for line in nodes_idx:
        b.extend(line)
        c += len(line)
    assert c           == n_nodes , 'c != n_nodes'
    assert len(set(b)) == n_nodes , 'duplicata node index'
    del b, c
    
    def _create_edges(lineboxes:List[List[VOCBox]], 
                      nodes_idx:List[Tuple[int, ...]]
                      ) -> Tuple[list, list]:
        r"""
        input: tọa độ các box và node index
        tạo các edges cho graph
        !TODO 1 trong cùng 1 line , 1 edge sẽ link 2 nodes cạnh nhau
        !TODO 2 đối với các box khác line, 1 edge sẽ link 2 node 
        có overlap trong phạm vi ymin:ymax
        """
        
        src_node = []
        dst_node = []
        
        # NOTE tạo edges giữa 2 box khác dòng/line
        for linebox,        lineIDX,        nextlinebox,    nextlineIDX in zip(
            lineboxes[:-1], nodes_idx[:-1], lineboxes[ 1:], nodes_idx[ 1:]):
            #src node                       #dst node
            for box_src, idx_src in zip(linebox, lineIDX):
                for box_dst, idx_dst in zip(nextlinebox, nextlineIDX):
                    if is_edge_diff_line(box_src, box_dst):
                        src_node.append(idx_src)
                        dst_node.append(idx_dst)
        
        # NOTE tạo edges giữa 2 box liên tiếp trong cùng 1 dòng/line
        for lineIDX in nodes_idx:
            # tạm thời ko dùng tọa độ của box trong linebox
            # nếu tạo thêm edge và cần thress thì dùng thêm linebox
            numbox = len(lineIDX)
            if numbox >= 3:
                for idx, nextidx in zip(lineIDX[:-1], lineIDX[:1]):
                    src_node.append(idx)
                    dst_node.append(nextidx)
            elif numbox == 2:
                src_node.append(lineIDX[0])
                dst_node.append(lineIDX[-1])
            # if numbox <= 1:
            #     continue
            
        return src_node, dst_node
    
        #     for box_src, idx_src, box_dst, idx_dst in zip(
        #         linebox, lineIDX, nextlinebox, nextlineIDX):
        #         if is_edge(box_src, box_dst):
        #             src_node.append(idx_src)
        #             dst_node.append(idx_dst)
        
        # for line, nextline in zip(lineboxes[:-1], lineboxes[1:]):
        #     for qbox in line: 
        #         for sbox in nextline: 
        #             if is_edge(qbox, sbox):
        #                 src_node.append(qbox[-1])
        #                 dst_node.append(sbox[-1])
        #             break
    
    def _create_edges_oneline(nodes_idx:List[Tuple[int, ...]]
                              ) -> Tuple[List[int], List[int]]:
        r"""
        tạo cạnh nối 2 node có index liên tiếp
        từ đầu dòng đến cuối dòng , cuối dòng trên nối với đầu dòng dưới
        """
        src, dst = [], []
        for nodeidx in chain(*nodes_idx):
            src.append(nodeidx)
            dst.append(nodeidx + 1)
        
        src.pop(-1)
        dst.pop(-1)
        assert len(src) == len(dst)
    
        return src, dst
    
    def _create_edges_every_box(nodes_idx:List[Tuple[int, ...]]
                                ) -> Tuple[List[int], List[int]]:
        r"""
        tạo edge nối từng box của dòng trên với từng box của dòng dưới
        """

        src, dst = [], []
        for lineIdx, nextlineIdx in zip(nodes_idx[:-1], nodes_idx[1:]):
            for box in lineIdx:
                for nbox in nextlineIdx:
                    src.append(box)
                    dst.append(nbox)

        return src, dst
    
    def _create_edges_step_box(nodes_idx:List[Tuple[int, ...]]
                                ) -> Tuple[List[int], List[int]]:
        raise NotImplementedError('chưa implement tạo edges theo pp này')
        src, dst = [], []
        for first_L, second_L, thirt_L in zip(
            nodes_idx[:-2], nodes_idx[1:-1], nodes_idx[2:]):
            ...
        
        return src, dst
    
    #NOTE code test
    assert len(lineboxes) == len(nodes_idx), 'thiếu dòng'
    for linebox, lineIDX in zip(lineboxes, nodes_idx):
        assert len(linebox) == len(lineIDX), 'độ dài dòng khác nhau, thiếu box'
        
    # src_node, dst_node = _create_edges(lineboxes, nodes_idx)
    # src_node, dst_node = _create_edges_oneline(nodes_idx)
    src_node, dst_node = _create_edges_every_box(nodes_idx)     #TODO
    # src_node, dst_node = _create_edges_step_box(nodes_idx)
    
    # NOTE save graph as csv and npz
    
    name_save = osp.join('dataset/DKKD_graph', osp.basename(js))
    save_edges((src_node, dst_node), re.sub('.json$', '.edges.csv', name_save))
    save_nodes(nodes_idx, lbls, re.sub('.json$', '.idx.csv', name_save),
               nodes_feats, re.sub('.json$', '.nfeat.npy', name_save))
    

if __name__ == '__main__':
    inp = '/home/agent/Documents/graph/GNN/dataset/DKKD'
    out = 'dataset/graph_DKKD'
    
    jsons = glob(osp.join(inp, '*.json'))
    start = time()
    for js in jsons:
        imgp = re.sub('.json$', '.jpg', js)
        print(Fore.LIGHTGREEN_EX)
        print(js, Fore.RESET)
        
        build_graph(js, imgp)
    end = time()
    print(Fore.MAGENTA, 'runtime: ', end-start) #782.7624568939209