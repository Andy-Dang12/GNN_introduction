import os, re, json
import os.path as osp
from itertools import chain
# import xml.etree.ElementTree as ET
from glob import glob
from turtle import end_fill
from typing import Dict, List, Optional, Tuple, Union
# from xml.dom.minidom import parseString
# from xml.dom.expatbuilder import parseString
from colorama import Fore
import cv2
import numpy as np
import pandas as pd
import torch, dgl
import torch.nn as nn

from dgl.data import DGLDataset
from dgl import DGLGraph

from vietOCR import img2word, imgs2words
from phoBERT import word2vec


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
    classes = [l for l in lines if l not in labels]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

classes, class_to_idx = create_cls_to_idx('dataset/DKKD/labels.txt')

def class_2_idx(classname:str):
    assert classname in class_to_idx.keys(), 'invalid classname'
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
    node feature save trong file .npz
    """
    assert nodes_path.endswith('.csv'), 'save node data ở dạng .csv'
    assert feats_path.endswith('.npz'), 'save node feat ở dạng .npz'
    
    node = pd.DataFrame({'Id':chain(*idx), 'label':chain(*label)})
    node.to_csv(nodes_path, encoding='utf-8', index=False)
    
    feats = [tensor1D.numpy() for tensor1D in chain(*feats)]
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
        #NOTE code test
        assert len(lineboxes) == len(nodes_idx), 'thiếu dòng'
        for linebox, lineIDX in zip(lineboxes, nodes_idx):
            assert len(linebox) == len(lineIDX), 'độ dài dòng khác nhau, thiếu box'
        
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
        assert len(lineboxes) == len(nodes_idx), 'len(lineboxes) != len(nodes_idx)'
        for linebox, lineIDX in zip(lineboxes, nodes_idx):
            assert len(linebox) == len(lineIDX), 'len(linebox) != len(lineIDX)'
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
    
    src_node, dst_node = _create_edges(lineboxes, nodes_idx)
    # NOTE save graph as csv and npz
    
    name_save = osp.join('dataset/graph_DKKD', osp.basename(js))
    save_nodes(nodes_idx, lbls, re.sub('.json$', '.idx.csv', name_save),
               nodes_feats, re.sub('.json$', '.nfeat.npz', name_save))
    save_edges((src_node, dst_node), re.sub('.json$', '.edges.csv', name_save))
    
    # # NOTE DGL Graph Construction
    # g = dgl.graph((src_node, dst_node), num_nodes=n_nodes)
    # g = dgl.to_bidirected(g)
    
    # g.ndata['feat'] = ... #! FIXME
    # g.ndata['label'] = ...
    # g.ndata['train_mask'] = torch.ones(n_nodes, dtype=torch.bool)   #tất cả để train/val/test
    # g.ndata['val_mask'] = torch.ones(n_nodes, dtype=torch.bool)
    # g.ndata['test_mask'] = torch.ones(n_nodes, dtype=torch.bool)
    

class DKKDGraphDataset(DGLDataset):
    def __init__(self, name:str, root:str):
        super().__init__(name='DKKD')
        self.jsons = glob(osp.join(root, '*.json'))
        
    def __len__(self): 
        return len(self.jsons)
    
    def __getitem__(self, idx:int) -> DGLGraph: 
        js = self.jsons[idx]
        imgp = re.sub('.json$', '.jpg', js)
        assert osp.isfile(imgp), '{img} is not found'.format(img=imgp)
        
        return build_graph(js, imgp)
    
    def process(self):
        #NOTE dataset gồm nhiều graph, mỗi ảnh sẽ tạo thành 1 graph

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
        

if __name__ == '__main__':
    from colorama import Fore
    inp = '/home/agent/Documents/graph/GNN/dataset/DKKD'
    out = 'dataset/graph_DKKD'
    
    jsons = glob(osp.join(inp, '*.json'))
    for js in jsons:
        imgp = re.sub('.json$', '.jpg', js)
        print(Fore.LIGHTGREEN_EX)
        print(js, Fore.RESET)
        
        build_graph(js, imgp)