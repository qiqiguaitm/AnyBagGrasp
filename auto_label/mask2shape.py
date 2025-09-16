# from .opt_utils import OptUtils
import copy
import json
import logging
import math
import os
import pdb
import random
import shutil
import sys
import tempfile
import textwrap
from collections import OrderedDict
from io import StringIO

import cv2
import numpy as np
from cv2 import RETR_CCOMP
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# log = logging.getLogger()
log = logging.getLogger(__name__)
log.setLevel(level=logging.DEBUG)
fileLogFormatter = logging.Formatter('%(asctime)s %(levelname)4.4s%(funcName)8.8s  %(filename)s:%(lineno)3.3s]-> %(message)s')
fileHandler = logging.FileHandler('log/log.log')
fileHandler.setFormatter(fileLogFormatter)
fileHandler.setLevel(logging.INFO)
log.addHandler(fileHandler)

consoleLogFormat = logging.Formatter('[%(levelname)4.4s%(funcName)8.8s  %(filename)s:%(lineno)3.3s]-> %(message)s', datefmt='%m-%d %H:%M')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(consoleLogFormat)
consoleHandler.setLevel(logging.DEBUG)
log.addHandler(consoleHandler)


def round_floats(obj, decimals=1):
    keys = ['iou']
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        # for k in keys if k in obj:
        part1 = {k: round_floats(v, decimals + 2) for k, v in obj.items() if k in keys}
        part2 = {k: round_floats(v, decimals) for k, v in obj.items() if k not in keys}
        return part1 | part2
    elif isinstance(obj, (list, tuple)):
        return [round_floats(x, decimals) for x in obj]
    return obj


def to_list(obj, decimals=3):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_list(x) for x in obj]
    return obj


def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list | tuple):
            result.extend(flatten(item))  # 递归展开
        else:
            result.append(item)
    return result


class BagOpt:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def fit_shape(self, coco, mask_key, mask_root=None, policy='auto'):
        cats = coco.loadCats(coco.getCatIds())
        id2name = {cat['id']: cat['name'] for cat in cats}

        for ann in coco.dataset['annotations']:
            if mask_key not in ann:
                raise NotImplementedError(f'{mask_key} not found')
            mask_path = ann[mask_key]
            if mask_root is not None:
                mask_path = os.path.join(mask_root, mask_path)
                if mask_path.endswith('.png'):
                    mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
                elif mask_path.endswith('.npy'):
                    mask = np.load(mask_path)
                else:
                    raise NotImplementedError(f'{mask_path} not found')
            else:
                mask = coco_mask.decode(mask_path)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)

            shapes = None
            if isinstance(policy, dict):
                cat_id = ann['category_id']
                cat = id2name[cat_id]
                shapes = [policy[cat]]
            elif policy == 'auto':
                shapes = ['rot_bbox', 'circle']
            elif policy == 'all':
                shapes = ['rot_bbox', 'circle', 'eclipse', 'convex', 'poly']
            elif isinstance(policy, str):
                shapes = [policy]
            else:
                raise NotImplementedError(f'{policy=}')

            ret = []
            for shape in shapes:
                guess_mask = np.zeros_like(mask)
                guess_mask = np.ascontiguousarray(guess_mask)  # this is must
                try:
                    if shape == 'rot_bbox':
                        guess = cv2.minAreaRect(largest_contour)
                        (cx, cy), (w, h), angle = guess
                        if w < h:
                            w, h = h, w  # 交换宽高
                            angle += 90  # 角度调整
                        angle = angle % 180  # in [0, 180] and w >=h which is the same as affordance

                        guess = [(cx, cy), (w, h), angle]

                        p4 = cv2.boxPoints(guess).astype(int)  # 4 points in (x,y)
                        # breakpoint()
                        guess = flatten(guess)  # xc, yc, w, h, angle(degree)
                        guess[4] = guess[4] / 180 * math.pi
                        # breakpoint()
                        cv2.drawContours(guess_mask, [p4], -1, 255, -1)

                    elif shape == 'circle':
                        guess = cv2.minEnclosingCircle(largest_contour)  # (x, y), radius
                        (x, y), radius = guess
                        guess = flatten(guess)
                        # breakpoint()
                        cv2.circle(guess_mask, (int(x), int(y)), int(radius), 255, -1)
                    elif shape == 'eclipse':
                        guess = cv2.fitEllipse(largest_contour)
                        cv2.ellipse(guess_mask, guess, 255, -1)  # 青色椭圆
                    elif shape == 'convex':
                        guess = cv2.convexHull(largest_contour)
                        cv2.drawContours(guess_mask, [guess], -1, 255, -1)  # 紫色凸包
                    elif shape == 'poly':
                        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                        guess = cv2.approxPolyDP(largest_contour, epsilon, True)
                        cv2.drawContours(guess_mask, [guess], -1, 255, -1)  # 黄色多边形
                    else:
                        raise NotImplementedError(f'{shape=}')
                    iou = self.calculate_iou(mask, guess_mask)
                    guess = to_list(guess)
                    rle = coco_mask.encode(np.asfortranarray(guess_mask))
                    rle['counts'] = rle['counts'].decode('utf-8')
                except Exception as e:
                    guess = 0
                    iou = 0
                    rle = str(e)
                    print(f'{e}: {shape=}')

                tmp = dict(name=shape, parameter=guess, iou=iou, guess_mask=rle)
                ret.append(tmp)
            ann['shapes'] = ret
        return coco

    def calculate_iou(self, mask1, mask2):
        """计算两个二值掩码之间的IoU"""
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union)
        return float(iou)


def add_shapes_to_annotations(coco_data, policy='auto'):
    """
    为COCO格式的标注数据添加shapes字段
    
    Args:
        coco_data: COCO格式的标注数据字典
        policy: 形状拟合策略，默认'auto'（使用rot_bbox和circle）
    
    Returns:
        更新后的COCO数据
    """
    from pycocotools import mask as coco_mask
    
    bag_opt = BagOpt()
    
    # 创建临时COCO对象
    class TempCOCO:
        def __init__(self, dataset):
            self.dataset = dataset
            
        def loadCats(self, cat_ids):
            return [cat for cat in self.dataset['categories'] if cat['id'] in cat_ids]
            
        def getCatIds(self):
            return [cat['id'] for cat in self.dataset['categories']]
    
    temp_coco = TempCOCO(coco_data)
    
    # 处理每个标注
    for ann in coco_data['annotations']:
        if 'segmentation' in ann and ann['segmentation']:
            try:
                # 解码RLE掩码
                mask_rle = ann['segmentation']
                mask = coco_mask.decode(mask_rle)
                
                # 找到轮廓
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                    
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 确定要拟合的形状
                shapes = []
                if policy == 'auto':
                    shapes = ['rot_bbox', 'circle']
                elif policy == 'all':
                    shapes = ['rot_bbox', 'circle', 'ellipse', 'convex', 'poly']
                elif isinstance(policy, list):
                    shapes = policy
                else:
                    shapes = [policy]
                
                ret = []
                for shape in shapes:
                    guess_mask = np.zeros_like(mask)
                    guess_mask = np.ascontiguousarray(guess_mask)
                    
                    try:
                        if shape == 'rot_bbox':
                            guess = cv2.minAreaRect(largest_contour)
                            (cx, cy), (w, h), angle = guess
                            if w < h:
                                w, h = h, w
                                angle += 90
                            angle = angle % 180
                            
                            guess_params = [float(cx), float(cy), float(w), float(h), float(angle)]
                            p4 = cv2.boxPoints(guess).astype(int)
                            cv2.drawContours(guess_mask, [p4], -1, 255, -1)
                            
                        elif shape == 'circle':
                            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                            guess_params = [float(x), float(y), float(radius)]
                            cv2.circle(guess_mask, (int(x), int(y)), int(radius), 255, -1)
                            
                        elif shape == 'ellipse':
                            if len(largest_contour) >= 5:
                                guess = cv2.fitEllipse(largest_contour)
                                (cx, cy), (w, h), angle = guess
                                guess_params = [float(cx), float(cy), float(w), float(h), float(angle)]
                                cv2.ellipse(guess_mask, guess, 255, -1)
                            else:
                                continue
                                
                        elif shape == 'convex':
                            guess = cv2.convexHull(largest_contour)
                            guess_params = guess.flatten().tolist()
                            cv2.drawContours(guess_mask, [guess], -1, 255, -1)
                            
                        elif shape == 'poly':
                            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                            guess = cv2.approxPolyDP(largest_contour, epsilon, True)
                            guess_params = guess.flatten().tolist()
                            cv2.drawContours(guess_mask, [guess], -1, 255, -1)
                            
                        else:
                            continue
                        
                        # 计算IoU
                        iou = bag_opt.calculate_iou(mask, guess_mask)
                        
                        # 编码猜测的掩码
                        rle = coco_mask.encode(np.asfortranarray(guess_mask))
                        rle['counts'] = rle['counts'].decode('utf-8')
                        
                        tmp = {
                            'name': shape,
                            'parameter': guess_params,
                            'iou': float(iou),
                            'guess_mask': rle
                        }
                        ret.append(tmp)
                        
                    except Exception as e:
                        print(f"处理形状 {shape} 时发生错误: {e}")
                        continue
                
                # 按IoU降序排序
                ret.sort(key=lambda x: x['iou'], reverse=True)
                ann['shapes'] = ret
                
            except Exception as e:
                print(f"处理标注 {ann['id']} 时发生错误: {e}")
                continue
    
    return coco_data


def calculate_affordance_from_shapes(ann):
    """
    根据shapes字段计算affordance值
    策略：选择IoU最大的shape，根据形状类型返回对应的affordance
    
    Args:
        ann: 标注字典，包含shapes字段
        
    Returns:
        affordance值 [xc, yc, w, h, angle]（rot_bbox）或 [x, y, radius]（circle）或 None
    """
    if 'shapes' not in ann or not ann['shapes']:
        return None
    
    # 找到IoU最大的shape
    best_shape = max(ann['shapes'], key=lambda x: x['iou'])
    
    if best_shape['name'] == 'rot_bbox' and len(best_shape['parameter']) >= 5:
        # 使用rot_bbox的参数 [xc, yc, w, h, angle]
        # 注意：参数中的角度是度，需要转换为弧度
        params = best_shape['parameter']
        return [
            float(params[0]),  # xc
            float(params[1]),  # yc  
            float(params[2]),  # w
            float(params[3]),  # h
            float(params[4]) * math.pi / 180.0   # angle (度转弧度)
        ]
    elif best_shape['name'] == 'circle' and len(best_shape['parameter']) >= 3:
        # 使用circle的参数 [x, y, radius]
        params = best_shape['parameter']
        return [
            float(params[0]),  # x
            float(params[1]),  # y
            float(params[2])   # radius
        ]
    
    # 对于其他形状，返回None（保持原来的affordance计算）
    return None
