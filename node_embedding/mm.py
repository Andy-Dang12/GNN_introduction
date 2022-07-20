from typing import Tuple
import torch, dgl
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
import re, cv2
import numpy as np
import pandas as pd
import os.path as osp
from colorama import Fore
from glob import glob, iglob
from random import shuffle
import warnings
warnings.filterwarnings('ignore')


def get_angle_vertical(bboxes):
    if len(bboxes) <= 0:
        return 0.0
    min_x = np.min(bboxes[..., 0::2], axis=1).reshape(-1, 1)
    max_x = np.max(bboxes[..., 0::2], axis=1).reshape(-1, 1)
    min_y = np.min(bboxes[..., 1::2], axis=1).reshape(-1, 1)
    max_y = np.max(bboxes[..., 1::2], axis=1).reshape(-1, 1)
    mean_x = min_x / 2 + max_x / 2
    mean_y = min_y / 2 + max_y / 2
    _tan = np.abs(mean_y.T - mean_y) / np.abs(mean_x.T - mean_x + 1e-9)
    angle = np.rad2deg(np.arctan(_tan))
    return angle


def get_iou_full( bboxes):
    alone_area = poly_area_numpy(bboxes).reshape(-1, 1)
    merged_area = get_area_merged_box(bboxes)
    iou = (alone_area.T + alone_area) / merged_area
    return iou


def poly_area_numpy(boxes, dim=1):
    x = boxes[..., 0::2]
    y = boxes[..., 1::2]
    result = 0.5 * np.abs(
        np.sum(x * np.roll(y, 1, axis=dim), axis=dim) - np.sum(y * np.roll(x, 1, axis=dim), axis=dim))
    return result


def get_area_merged_box(bboxes):
    y = np.pad(bboxes, ((0, 0), (0, bboxes.shape[1])), 'constant')
    x = np.pad(bboxes, ((0, 0), (bboxes.shape[1], 0)), 'constant')
    concat = (x + y.reshape(y.shape[0], 1, y.shape[1]))
    min_x = np.min(concat[..., 0::2], axis=2)
    min_y = np.min(concat[..., 1::2], axis=2)
    max_x = np.max(concat[..., 0::2], axis=2)
    max_y = np.max(concat[..., 1::2], axis=2)
    return (max_x - min_x) * (max_y - min_y)


def sort_document(bboxes, angle, thresh_iou, sorted_horizone=True):
    bboxes = np.array(bboxes)
    x = get_angle_vertical(bboxes)
    x = x < angle
    y = get_iou_full(bboxes)
    y = y > thresh_iou
    adj = x * y
    np.fill_diagonal(adj, 0)
    doc = dict()
    for idx in range(len(bboxes)):
        check = False
        for i in range(len(doc.keys())):
            box_prev = doc[i][-1]
            if adj[box_prev][idx]:
                check = True
                doc[i].append(idx)
                break
        if not check:
            doc[len(doc.keys())] = []
            doc[len(doc.keys()) - 1].append(idx)
    if sorted_horizone:
        # sorted(word_boxes, key=lambda x: min(x[::2])
        doc = dict(sorted(doc.items(), key=lambda x: np.min(bboxes[x[1][0]][1::2])))
    return doc


def draw_img_with_bound(src, bbox):
    return cv2.rectangle(src, [int(bbox[0]), int(bbox[1])], [int(bbox[2]), int(bbox[3])], color=[0, 255, 0],
                         thickness=1)
    
def get_iou(bb1, bb2):
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    min_area = bb1_area if bb1_area < bb2_area else bb2_area
    iou = intersection_area / min_area  # float(bb1_area + bb2_area - intersection_area)
    #     assert iou >= 0.0
    #     assert iou <= 1.0
    return iou


def get_box(bb):
    bb = np.array(bb)
    x_min = np.min(bb[0::2])
    x_max = np.max(bb[0::2])
    y_min = np.min(bb[1::2])
    y_max = np.max(bb[1::2])
    return [x_min, y_min, x_max, y_max]


def text_to_indicate(text: str, vocab: dict):
    return [vocab[char] for char in text]


#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch

try:
    import tesserocr
except ImportError:
    tesserocr = None

import unidecode
from engine import MMOCR

engine_ocr = MMOCR(det=None,
                 det_config='/workspace/text/MMOCR/mmocr-main/configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py',
                 det_ckpt='/workspace/text/MMOCR/mmocr-main/work_dirs/dbnet_r50dcnv2_fpnc_1200e_icdar2015/epoch_800.pth',
                 recog="SATRN",
                 recog_config='/workspace/text/MMOCR/mmocr-main/configs/textrecog/satrn/satrn_academic.py',
                 recog_ckpt='/workspace/text/MMOCR/mmocr-main/work_dirs/satrn_academic/epoch_180.pth',
                 kie='',
                 kie_config='',
                 kie_ckpt='',
                 config_dir=os.path.join(str(Path.cwd()), 'configs/'),
                 device="cuda",)


with open("/workspace/anhnt/vn_dict.txt", mode = "r") as f:
    chars = f.readlines()
    chars = "".join(chars).replace("\n", "")
    
print(chars)
charss = []
for char in chars:
    if unidecode.unidecode(char) == char:
        charss.append(char)
chars = "".join(charss)
len(chars)
idx2char = {key: val for val, key in zip(chars ,range(len(chars)))}
char2idx = {key: val for key, val in zip(chars ,range(len(chars)))}
import glob
import json
import re
import numpy as np
import cv2
import unidecode
from utils import *
src = "/workspace/anhnt/Graph/Graph/dkkd"
des = "/workspace/anhnt/Graph/Graph"
jsons = glob(src + "/*.json")

x = dict()
idx = 0
m = 0
word_boxes_sorteds = []
import time
def get_data(jsons):
    for js in jsons[:-5]:
        with open(js,encoding="utf-8") as f:
            data_global = json.load(f)
        try:
            with open(js.replace("/dkkd", ""), encoding="utf-8") as f:
                data_local = json.load(f)
        except:
            continue
        bboxes = data_local['shapes']
        comp_bboxes = data_global['shapes']
        src_box = None

        for comp_box in comp_bboxes:
            label = comp_box['label']
            comp_box = re.findall("\d+\.\d+", str(comp_box['points']))
            comp_box = [float(i) for i in comp_box]
            if label == "ten_cong_ty":
                src_box = comp_box
        if not isinstance(src_box, list): 
            continue
        src = []
        labels = []

        for box in bboxes:
            label = box['label']
            box = re.findall("\d+\.\d+", str(box['points']))
            if "key" in label: label = 0
            else: label = 1
            box = [float(i) for i in box]
            box_four = get_box(box)

            iou = get_iou(box_four, src_box)
            if iou > 0.5:
                src.append(box)
                labels.append(label)

        img = cv2.imread(f"{js[:-4]}jpg")
        src = sorted(zip(src, labels), key = lambda x: x[0][0])
        labels = [x[1] for x in src]
        src = [x[0] for x in src]
        box_img = []
        for box in src:
            box = get_box(box)
            box = [int(i) for i in box]
            box_img.append(img[box[1]:box[3], box[0]:box[2]])

        texts, out_enc = engine_ocr.readtext(box_img, batch_mode=True, recog_batch_size=64)
        texts = [unidecode.unidecode(text['text']) for text in texts]

        doc = sort_document(src, 10, 0.4)
        word_boxes_sorted = [] 
        for i in doc.keys():
            for j in doc[i]:
                min_x = min(src[j][0::2])
                max_x = max(src[j][0::2])
                word_boxes_sorted.append([[min_x, max_x], text_to_indicate(texts[j], char2idx), labels[j] ])
        word_boxes_sorteds.append(word_boxes_sorted)
    return word_boxes_sorteds
train_data = get_data(jsons[:-5])
val_data = get_data(jsons[-5:])


