# -*- coding: utf-8 -*-
import os
import argparse

import torch
import numpy as np
from ultralytics import YOLO

from folder_paths import models_dir
from .adjust_yolo_output import get_iou, adjust, find_box, resize_img_pad, map_transformed_to_original

ultra_models_dir = os.path.join(models_dir, "ultralytics")


class YoloPredict:
    @classmethod
    def INPUT_TYPES(cls):
        files = []
        for root, dirs, filenames in os.walk(ultra_models_dir):
            for filename in filenames:
                if filename.endswith(".pt") or filename.endswith(".onnx"):
                    relative_path = os.path.relpath(os.path.join(root, filename), ultra_models_dir)
                    files.append(relative_path)
        return {
            "required": {
                "model_path": (sorted(files), {"model_upload": True}),
                "image": ("IMAGE",),
            },
            "optional": {
                "conf": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.01}),
                "iou": ("FLOAT", {"default": 0.45, "min": 0, "max": 1, "step": 0.01}),
                "imgsz": ("INT", {"default": 640, "min": 64, "max": 1280, "step": 32}),
                "device": (["cuda:0", "cpu"],),
                "half": ("BOOLEAN", {"default": False}),
                "augment": ("BOOLEAN", {"default": False}),
                "classes": ("STRING", {"default": "None"}),

            },
        }

    RETURN_TYPES = ("IMAGE", "LIST",)
    RETURN_NAMES = ("image", "output",)
    FUNCTION = "inference"
    CATEGORY = "YOLO"
    DESCRIPTION = """YOLO推理节点"""

    def inference(self, model_path, image, conf, iou, imgsz, device, half, augment, classes):
        model_full_path = os.path.join(ultra_models_dir, model_path)
        model = YOLO(model_full_path)

        if classes == "None":
            class_list = None
        else:
            class_list = [int(cls.strip()) for cls in classes.split(',')]

        if len(image.shape) != 4:
            re_image = image.cpu().numpy()
            hi, wi = re_image.shape[:2]
            re_image = resize_img_pad(re_image, imgsz)
            re_image = torch.from_numpy(re_image).unsqueeze(0).float()
            input_image = re_image.permute(0, 3, 1, 2)
        else:
            re_image = image.squeeze(0).cpu().numpy()
            hi, wi = re_image.shape[:2]
            re_image = resize_img_pad(re_image, imgsz)
            re_image = torch.from_numpy(re_image).unsqueeze(0).float()
            input_image = re_image.permute(0, 3, 1, 2)

        results = model.predict(input_image, imgsz=imgsz, conf=conf, iou=iou, batch=1, device=device, classes=class_list,
                                half=half, augment=augment, save=False, save_txt=False)
        data = results[0].boxes.data.cpu().tolist()
        if len(data) != 0:
            new_data = []
            for b in data:
                bb = map_transformed_to_original(b[:4], (hi, wi), (imgsz, imgsz))
                new_data.append(bb+b[4:])
            data = new_data

        return (re_image, data, )


class FindBox:
    """
    从检测模型结果输出最大或最小的检测目标，输入数据格式:
    [
        [xmin,ymin,xmax,ymax,conf,cls],
        ...
    ]

    输出数据格式：
    [xmin,ymin,xmax,ymax,cls1,cls2,cls3,...]
    """
    @classmethod
    def INPUT_TYPES(cls):
         return {
            "required": {
                "data": ("LIST", ),
            },
            "optional": {
                "mode": (["max", "min"], ),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("output",)
    FUNCTION = "inference"
    CATEGORY = "YOLO"
    DESCRIPTION = """从检测模型结果输出最大或最小检测目标"""

    def inference(self, data, mode):
        if len(data) == 0:
            return (data,)
        adj_data = adjust(data)
        output = find_box(adj_data, mode)

        return (output,)


class CategoryJudgment:
    """
    判断结果是否含有或不含有指定类别, 符合要求输出True，否则False
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("LIST", ),
            },
            "optional": {
                "contain": ("STRING", {"default": '0'}),
                "exclusive": ("STRING", {"default": '3, 5'}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "INT",)
    RETURN_NAMES = ("result", "result_int",)
    FUNCTION = "inference"
    CATEGORY = "YOLO"
    DESCRIPTION = """判断结果是否含有指定类别, 符合要求输出True，否则False"""

    def inference(self, data, contain, exclusive):
        if len(data) == 0:
            flag = False
            flag_int = 2
            return (flag, flag_int)

        # 整理字符串为列表
        contain = eval(contain)
        exclusive = eval(exclusive)

        # 读取数据类别
        clses = data[4:]
        flag = False
        for c in contain:
            if c not in clses:  # 需要包含的类别不存在
                continue
            flag = True  # 需要包含的类别均存在，flag暂时设置为True

        # 检查不包含的类别
        for e in exclusive:
            if e in clses:  # 不包含类别存在，flag设置为False且跳出循环
                flag = False
                break

        flag_int = 1 if flag else 2
        return (flag, flag_int)


class CropImage:
    """
    裁剪图片
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "data": ("LIST",),
            },
            "optional": {
                "ratio": ("FLOAT", {"default": 0.1, "min": 0, "max": 0.5, "step": 0.05 }),
                "center": ("BOOLEAN", {"default": True})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inference"
    CATEGORY = "YOLO"
    DESCRIPTION = """根据给定坐标裁剪图片"""

    def inference(self, image, data, ratio, center):
        if len(data) == 0:
            return (image,)
        # 调整图片格式为numpy数组
        image0 = image
        if len(image0.shape) == 4:
            image0 = image0.squeeze(0)
        if isinstance(image0, torch.Tensor):
            image0 = image0.cpu().numpy()
        (hi, wi) = image0.shape[:2]

        # 获取裁剪坐标
        w = data[2] - data[0] + 1
        h = data[3] - data[1] + 1
        c_w = w * ratio
        c_h = h * ratio
        if center:
            if round(data[0] - c_w) < 0:
                c_w = data[0]
            if round(data[2] + c_w) > wi:
                c_w = wi - data[2]
            if round(data[1] - c_h) < 0:
                c_h = data[1]
            if round(data[3] + c_h) > hi:
                c_h = hi - data[3]
            xmin = round(data[0] - c_w)
            ymin = round(data[1] - c_h)
            xmax = round(data[2] + c_w)
            ymax = round(data[3] + c_h)
        else:
            xmin = max(round(data[0] - c_w), 0)
            ymin = max(round(data[1] - c_h), 0)
            xmax = min(round(data[2] + c_w), wi)
            ymax = min(round(data[3] + c_h), hi)


        # 裁剪图片
        crop_image = image0[ymin: ymax+1, xmin: xmax+1, :]
        crop_image_tensor = torch.from_numpy(crop_image).unsqueeze(0).float()

        return (crop_image_tensor,)


YOLO_NODE_CLASS_MAPPINGS = {
    "YoloPredict": YoloPredict,
    "筛选最大或最小目标": FindBox,
    "判断目标是否符合指定类别": CategoryJudgment,
    "按坐标裁剪图像": CropImage,
}

YOLO_NODE_DISPLAY_NAME_MAPPINGS = {
    "YoloPredict": "Yolo推理",
    "筛选最大或最小目标": "筛选目标",
    "判断目标是否符合指定类别": "判断结果",
}
