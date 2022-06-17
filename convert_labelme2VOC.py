import os
import json
from glob import glob
from typing import List

# tạo file label.txt để input cho labelme2voc.py
datafolder = '/home/agent/Documents/graph/GNN_introduction/dataset'
labels = ['__ignore__', '_background_']

def get_shape_info(jsonp:str) -> List[int, int, int, int]:
    with open(jsonp, 'r') as f:
        data = json.load(f)
    bboxes = data['shapes']
    for box in bboxes:
        lbl = box['label'].strip()
        if lbl not in labels:
            labels.append(lbl)


def create_labeltxt(datafolder:str):
    jsons = glob(os.path.join(datafolder, '*.json'))    
    for jsonpath in jsons:
        get_shape_info(jsonpath)

    with open('labels.txt', 'w') as f:
        f.write('\n'.join(labels))


if __name__ == '__main__':
    datafolder = '/home/agent/Documents/graph/GNN_introduction/dataset'

    