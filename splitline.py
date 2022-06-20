from typing import List, Tuple
from convert_labelme2VOC import get_shape_info

boxinfo = Tuple[int, int, int, int, str]
#             xmin, ymin, xmax, ymax, label

def splitlines(boxes:List[boxinfo]) -> List[List[boxinfo]]:
    def _get_line_center_x2(box:boxinfo) -> int:
        return box[1] + box[3]
    
    #NOTE sort by ymin/ymax/ycenter from low to high
    boxes = sorted(boxes, key=lambda x:(x[1] + x[3]))
    
    lines = []
    line  = [boxes[0]]
    linecenter_x2 = _get_line_center_x2(boxes[0])
    for box in boxes[1:]:
        if 2*box[1] < linecenter_x2:
            line.append(box)
        
        else:
            lines.append(line.copy())
            line = [box]
        linecenter_x2 = _get_line_center_x2(box)
        
    lines.append(line)
    
    def _order_box_xmin(boxes:List[boxinfo]) -> List[boxinfo]:
        return sorted(boxes, key= lambda x: x[0])
    
    return [_order_box_xmin(l) for l in lines]
    
