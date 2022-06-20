from glob import glob
from typing import List, Tuple
import os, json, re, cv2, numpy as np
import os.path as osp
import xml.etree.ElementTree as ET
# from xml.dom.minidom import parseString
from xml.dom.expatbuilder import parseString


# tạo file label.txt để input cho labelme2voc.py
datafolder = '/home/agent/Documents/graph/GNN_introduction/dataset'
labels = ['__ignore__', '_background_']


def get_shape_info(jsonp:str) -> List[Tuple[int, int, int, int, str]]:
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


def save_file_xml(img:np.ndarray, abspath:str, folder_save:str, boxes:list) -> None:
    r"""    img     opencv
        abspath     đường dẫn đầy đủ của ảnh
    folder_save     là folder để save
          boxes     là list các box theo thứ tự xmin, ymin, xmax, ymax """ 
    
    basename = osp.basename(abspath)
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = basename
    
    #NOTE <source>
    # source = ET.SubElement(annotation, "source")
    # ET.SubElement(source, "database").text= "The VOC2007 Database"
    # ET.SubElement(source, "annotation").text= "PASCAL VOC2007"
    # ET.SubElement(source, "image").text= "flickr"
    
    #NOTE <size>
    # size = ET.SubElement(annotation, "size")
    # hei, wid, ch = img.shape
    # ET.SubElement(size, "width").text = str(wid)
    # ET.SubElement(size, "height").text = str(hei)
    # ET.SubElement(size, "depth").text = str(ch)
    
    #NOTE <segmented>
    # ET.SubElement(annotation, "segmented").text = "0"

    for box in boxes:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = str(box[4]).strip()   #NOTE classname
        
        #NOTE <actions>
        # actions = ET.SubElement(obj, "actions")
        # ET.SubElement(actions, "jumping").text = '0'
        # ET.SubElement(actions, "other").text = '0'
        # ET.SubElement(actions, "phoning").text = '1'
        # ET.SubElement(actions, "playinginstrument").text = '0'
        # ET.SubElement(actions, "reading").text = '0'
        # ET.SubElement(actions, "ridingbike").text = '0'
        # ET.SubElement(actions, "ridinghorse").text = '0'
        # ET.SubElement(actions, "running").text = '0'
        # ET.SubElement(actions, "takingphoto").text = '0'
        # ET.SubElement(actions, "usingcomputer").text = '0'
        # ET.SubElement(actions, "walking").text = '1'

        # ET.SubElement(obj, "truncated").text = "0"
        # ET.SubElement(obj, "difficult").text = "0"
        # ET.SubElement(obj, "pose").text = "Unspecified"
        
        #NOTE <point>
        # point = ET.SubElement(obj, "point")
        # ET.SubElement(point, "x").text = '260'  # example
        # ET.SubElement(point, "y").text = '135'  # example

        #NOTE <bndbox>
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(box[0])
        ET.SubElement(bndbox, "ymin").text = str(box[1])
        ET.SubElement(bndbox, "xmax").text = str(box[2])
        ET.SubElement(bndbox, "ymax").text = str(box[3])
    
    abspath_xml = osp.join(folder_save, basename.rsplit('.', maxsplit=1)[0] + '.xml')
    # tree = ET.ElementTree(annotation)
    # ET.indent(tree, space="\t", level=0)
    # tree.write(name_xml)
    dom = parseString(ET.tostring(annotation))
    xmlstr = dom.toprettyxml(indent='\t')
    
    # remove <?xml version="1.0" ?>
    lines = xmlstr.splitlines()[1:]
    xmlstr = '\n'.join(lines)
    with open(abspath_xml, 'w') as f:
        f.write(xmlstr)


def convert_Labelme2VOC(datafolder:str):
    jsons = glob(osp.join(datafolder, '*.json'))
    for js in jsons:
        boxes = get_shape_info(js)
        imgp = re.sub('.json$', '.jpg', js)
        save_file_xml(None, imgp, datafolder, boxes)
        print(js)
    
    #NOTE create labels.txt contain label name of dataset
    with open(osp.join(datafolder, 'labels.txt'), 'w') as f:
        f.write('\n'.join(labels))


if __name__ == '__main__':
    datafolder = '/home/agent/Documents/graph/GNN/dataset'
    convert_Labelme2VOC(datafolder)
    