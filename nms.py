"""
@File    :   nms.py    
@Contact :   pengtt0119@gmail.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/11/27 5:34 下午   ttpeng      1.0         None
"""
import numpy as np


def NMS(dets, Nt):
    """
    :param dets: bounding boxes, format: [[xmin,ymin,xmax,ymax,scores]....]
    :param thresh: threshold of nms
    """
    # 1. compute area of each bounding box
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = dets[:, 4]

    # 2. sort the scores of bbox
    picked_boxes = []
    order = np.argsort(scores)

    # 3. append current max bbox into return list, and delete the bbox which has an iou value higher than threshold
    # with the current max bbox
    while order.size > 0:
        index = order[-1]
        picked_boxes.append(index)
        # compute intersections
        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])
        w = np.maximum(0.0, x22 - x11 + 1)
        h = np.maximum(0.0, y22 - y11 + 1)
        intersection = w * h
        # compute ious
        ious = intersection / (areas[index] + areas[order[:-1]] - intersection)
        # select
        box_left = np.where(ious < Nt)
        order = order[box_left]

    return dets[picked_boxes]


if __name__ == '__main__':
    boxes = np.array([[100, 100, 210, 210, 0.72],
                      [250, 250, 420, 420, 0.8],
                      [220, 220, 320, 330, 0.92],
                      [100, 100, 210, 210, 0.72],
                      [230, 240, 325, 330, 0.81],
                      [220, 230, 315, 340, 0.9]])
    res_nms = NMS(boxes, 0.7)
    print(res_nms)
