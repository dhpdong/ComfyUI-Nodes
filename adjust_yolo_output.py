# -*- coding: utf-8 -*-
import cv2
import numpy as np


def get_iou(box1, box2):
    union = max(min(box1[2], box2[2]) - max(box1[0], box1[0]) + 1, 0) * max(min(box1[3], box2[3]) - max(box1[1], box2[1]) + 1, 0)
    intersection = (box1[3] - box1[1] + 1) * (box1[2] - box1[0] + 1) + (box1[3] - box1[1] + 1) * (box1[2] - box1[0] + 1) - union

    return union/intersection

def adjust(input_list):
    results = []
    flag = []
    if len(input_list) == 1:
        return input_list
    for i in range(len(input_list)):
        if i in flag:
            continue
        flag.append(i)
        res = input_list[i][:4] + [int(input_list[i][5])]
        for k in range(i, len(input_list)):
            if k in flag:
                continue
            b1 = input_list[i][:4]
            b2 = input_list[k][:4]
            box_iou = get_iou(b1, b2)
            if box_iou >= 0.8:
                cls = int(input_list[k][5])
                res.append(cls)
                flag.append(k)
        results.append(res)

    return results


def find_box(boxes, m):
    box_ind = []
    box_area = None
    for box in boxes:
        area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        if box_area is None:
            box_ind = box
            box_area = area
        elif m == 'max':
            if area > box_area:
                box_ind = box
                box_area = area
        else:
            if area < box_area:
                box_ind = box
                box_area = area
    return box_ind


def resize_img_pad(img, size):
    shape = img.shape[:2]  # current shape [height, width]
    new_shape = size
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        if img.ndim == 2:
            img = img[..., None]

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    h, w, c = img.shape
    if c == 3:
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114/255, 114/255, 114/255))
    else:  # multispectral
        pad_img = np.full((h + top + bottom, w + left + right, c), fill_value=114/255, dtype=img.dtype)
        pad_img[top: top + h, left: left + w] = img
        img = pad_img

    return img


def map_transformed_to_original(t_box, ori_shape, tar_shape):
    """
    将变换后的坐标映射回原图坐标
    Args:
        x_trans, y_trans: 变换后图像中的坐标
        orig_shape: 原图尺寸 (H_orig, W_orig)
        target_shape: 目标尺寸 (H_target, W_target)
    Returns:
        (x_orig, y_orig): 原图坐标
    """
    ho, wo = ori_shape
    ht, wt = tar_shape

    # 计算缩放比例和填充量（与resize_img_pad一致）
    r = min(ht / ho, wt / wo)
    new_unpad_w = int(round(wo * r))
    new_unpad_h = int(round(ho * r))
    dw = wt - new_unpad_w
    dh = ht - new_unpad_h

    # 去除填充偏移
    t_box[0] = t_box[0] - dw / 2
    t_box[1] = t_box[1] - dh / 2
    t_box[2] = t_box[2] - dw / 2
    t_box[3] = t_box[3] - dh / 2

    # 逆向缩放
    xmin = t_box[0] / r
    ymin = t_box[1] / r
    xmax = t_box[2] / r
    ymax = t_box[3] / r

    return [xmin, ymin, xmax, ymax]
