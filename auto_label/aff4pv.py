#!/usr/bin/env python3
"""
Affordance for PV (Perspective View) Grasp Bag
生成侧视抓取Bag的Affordance
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import math
from typing import List, Dict, Tuple, Optional, Union
from pycocotools import mask as coco_mask

# 添加必要的路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入公共工具类
from common import VisualizationUtils, GeometryUtils, MaskUtils, HoleProcessingUtils

# 导入检测API
try:
    from dino_any_percept_api import DetectionAPI
    DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Detection API not available: {e}")
    DETECTION_AVAILABLE = False
    DetectionAPI = None


class Affordance4PVGraspBag:
    """
    生成侧视(back-front)抓取Bag的Affordance类
    """
    
    def __init__(self, 
                 detection_text: str = "paper bag.plastic bag.bag",
                 output_dir: str = "vis_aff_pv",
                 detection_api: Optional[DetectionAPI] = None,
                 verbose: bool = True,
                 enable_cache: bool = True,
                 cache_dir: str = ".cache"):
        """
        初始化Affordance生成器
        
        Args:
            detection_text: 检测目标的文本描述
            output_dir: 输出可视化结果的目录
            detection_api: 检测API实例（可选，如果提供则使用，否则创建新的）
            verbose: 是否输出详细信息
            enable_cache: 是否启用缓存
            cache_dir: 缓存目录
        """
        self.detection_text = detection_text
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化检测API
        if detection_api is not None:
            self.detection_api = detection_api
        elif DETECTION_AVAILABLE:
            self.detection_api = self._init_detection_api()
        else:
            self.detection_api = None
            if self.verbose:
                print("Warning: Detection API not available")
    
    def _init_detection_api(self):
        """初始化DetectionAPI"""
        from mmengine.config import Config
        
        cfg = Config()
        cfg.uri = r'/v2/task/dinox/detection'
        cfg.status_uri = r'/v2/task_status'  
        cfg.token = 'c4cdacb48bc4d1a1a335c88598a18e8c'
        cfg.model_name = 'DINO-X-1.0'
        
        if self.verbose:
            print("初始化DetectionAPI...")
        return DetectionAPI(cfg)
    
    def process(self, rgb_input: Union[str, Path]) -> Dict:
        """
        处理输入，可以是文件或目录
        
        Args:
            rgb_input: RGB图像文件或目录路径
            
        Returns:
            处理结果字典
        """
        rgb_path = Path(rgb_input)
        
        # 判断输入类型
        if rgb_path.is_file():
            # 处理单个文件
            if self.verbose:
                print(f"\nProcessing single file:")
                print(f"  RGB: {rgb_path}")
            return self._process_single(rgb_path)
        
        elif rgb_path.is_dir():
            # 处理目录
            if self.verbose:
                print(f"\nProcessing directory:")
                print(f"  RGB dir: {rgb_path}")
            return self._process_directory(rgb_path)
        
        else:
            raise ValueError("RGB input must be a file or directory")
    
    def _process_directory(self, rgb_dir: Path) -> Dict:
        """
        处理目录中的所有图像文件
        """
        # 获取RGB文件列表
        rgb_files = sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
        
        if not rgb_files:
            print(f"Warning: No RGB files found in {rgb_dir}")
            return {}
        
        results = {}
        
        for rgb_file in rgb_files:
            # 处理文件
            frame_result = self._process_single(rgb_file)
            frame_id = rgb_file.stem.split('_')[-1] if '_' in rgb_file.stem else rgb_file.stem
            results[frame_id] = frame_result
            
            if self.verbose:
                print(f"Processed frame {frame_id}: {len(frame_result.get('detections', []))} bags detected")
        
        return results
    
    def _process_single(self, rgb_file: Path) -> Dict:
        """
        处理单个RGB图像文件
        
        Args:
            rgb_file: RGB图像文件路径
            
        Returns:
            处理结果字典 {'rgb_file': str, 'detections': [...], 'affordances': [...]}
        """
        result = {
            'rgb_file': str(rgb_file),
            'detections': [],
            'affordances': []
        }
        
        # 1. 加载RGB图像
        rgb_image = cv2.imread(str(rgb_file))
        if rgb_image is None:
            print(f"Error: Cannot load RGB image {rgb_file}")
            return result
        
        h, w = rgb_image.shape[:2]
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {rgb_file.name}")
            print(f"Image size: {w}x{h}")
        
        # 2. 物体检测
        target_dets = self._detect_objects(rgb_image)
        result['detections'] = target_dets
        
        if not target_dets:
            if self.verbose:
                print("No bags detected in image")
            return result
        
        if self.verbose:
            print(f"Detected {len(target_dets)} object(s)")
        
        # 3. 对每个检测目标进行处理
        for idx, det in enumerate(target_dets):
            if self.verbose:
                print(f"\nProcessing detection {idx+1}/{len(target_dets)}:")
                print(f"  Bbox: {det['bbox']}")
                print(f"  Confidence: {det.get('confidence', 0):.3f}")
                print(f"  Label: {det.get('label', 'unknown')}")
            
            # 计算BF affordance
            affordance, handle_hole, rot_bbox_bag = self._calculate_bf_affordance_from_detection(
                det, rgb_image
            )
            
            # 构建affordance结果
            affordance_result = {
                'detection_idx': idx,
                'affordance': affordance,
                'handle_hole': handle_hole,
                'rot_bbox_bag': rot_bbox_bag,
                'grasp_policy': 'bf'
            }
            
            result['affordances'].append(affordance_result)
            
            if self.verbose:
                print(f"  Affordance: {affordance}")
                print(f"  Handle hole: {handle_hole}")
                print(f"  Rot bbox bag: {rot_bbox_bag}")
        
        # 4. 保存可视化结果
        self._save_visualization(rgb_image, rgb_file, result)
        
        return result
    
    def _detect_objects(self, rgb_image: np.ndarray) -> List[Dict]:
        """
        使用检测API检测物体
        """
        if not DETECTION_AVAILABLE or self.detection_api is None:
            # 检测不可用时返回空结果
            return []
        
        # 实际检测
        try:
            result = self.detection_api.detect_objects(
                rgb=rgb_image,
                prompt_text=self.detection_text,
                bbox_threshold=0.25,
                iou_threshold=0.8,
                mask_format='coco_rle'  # 明确请求mask
            )
            
            # 处理检测结果
            if result and 'objects' in result:
                target_dets = []
                for obj in result['objects']:
                    bbox = obj.get('bbox', [])
                    if len(bbox) >= 4:
                        # 转换bbox格式 [x1, y1, x2, y2] -> [x, y, width, height]
                        x1, y1, x2, y2 = bbox
                        det_dict = {
                            'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            'confidence': float(obj.get('score', 0.0)),
                            'label': obj.get('category', 'bag').lower()
                        }
                        # 添加mask（如果存在）
                        if 'mask' in obj and obj['mask'] is not None:
                            det_dict['mask'] = obj['mask']
                        target_dets.append(det_dict)
                return target_dets
            else:
                return []
        except Exception as e:
            print(f"Detection API error: {e}")
            return []
    
    def _calculate_bf_affordance_from_detection(self, detection: Dict, frame: np.ndarray, 
                                              fixed_width: float = 38.0) -> Tuple[List, Optional[List], Optional[List]]:
        """
        从检测结果计算BF affordance
        
        Args:
            detection: 检测结果字典 {'bbox': [...], 'mask': ..., 'label': ..., 'confidence': ...}
            frame: 图像帧
            fixed_width: 固定宽度（mm）
            
        Returns:
            Tuple of (affordance, handle_hole, rot_bbox_bag)
        """
        mask_rle = detection.get('mask')
        bbox = detection.get('bbox', [])
        category = detection.get('label', 'bag')
        
        # 检测API返回的bbox已经是[x1, y1, x2, y2]格式，直接使用
        if len(bbox) >= 4:
            bbox_xyxy = bbox[:4]  # 直接使用[x1, y1, x2, y2]格式
        else:
            return [], None, None
        
        return self._calculate_bf_affordance(mask_rle, bbox_xyxy, frame, 
                                           image_id=None, category=category, 
                                           fixed_width=fixed_width)
    
    def _process_single_legacy(self, mask_rle: Dict, bbox: List[float], frame: np.ndarray, 
                      image_id: Optional[int] = None, category: str = 'bag', 
                      fixed_width: float = 38.0) -> Tuple[List, Optional[List], Optional[List]]:
        """
        处理单个检测结果，计算PV视角的affordance
        
        Args:
            mask_rle: 掩码RLE编码
            bbox: 边界框 [x1, y1, x2, y2]
            frame: 图像帧
            image_id: 图像ID（可选）
            category: 检测类别
            fixed_width: 固定宽度（mm）
            
        Returns:
            Tuple of (affordance, handle_hole, rot_bbox_bag)
            - affordance: [cx, cy, w, h, angle] 或空列表
            - handle_hole: [cx, cy, radius] 或None
            - rot_bbox_bag: [cx, cy, w, h, angle] 或None
        """
        return self._calculate_bf_affordance(mask_rle, bbox, frame, image_id, category, fixed_width)
    
    def _calculate_bf_affordance(self, mask_rle, bag_bbox, frame, image_id=None, category='bag', fixed_width=38.0):
        """
        Backfront策略：
        - 对于bag：在mask中寻找最大的洞，用circle拟合作为affordance
        - 对于handle：直接用最小外接圆覆盖整个mask作为affordance
        """
        if not mask_rle or not isinstance(mask_rle, dict) or 'counts' not in mask_rle:
            return [], None, None
        
        try:
            # 解码掩码
            mask = coco_mask.decode(mask_rle)
            
            # 如果是handle类别，直接使用最小外接圆
            if 'handle' in category.lower():
                return self._process_handle_category(mask, bag_bbox, fixed_width)
            
            # 对于bag类别，寻找洞
            return self._process_bag_category(mask, bag_bbox, fixed_width)
            
        except Exception as e:
            print(f"计算backfront affordance时发生错误: {e}")
            return [], None, None
    
    def _process_handle_category(self, mask, bag_bbox, fixed_width):
        """处理handle类别的affordance"""
        # 找到mask的轮廓
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [], None, None
        
        # 获取最大轮廓（通常handle只有一个轮廓）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 检查handle mask面积是否在合理范围内 (5000 < area < 100000)
        handle_area = cv2.contourArea(largest_contour)
        if handle_area < 2000 or handle_area > 10000:
            if self.verbose:
                print(f"BF策略（handle）：mask面积={handle_area:.1f}不在范围内(2000, 10000)，返回空affordance")
            return [], None, None
        
        # 计算最小外接圆
        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
        
        # 检查circle半径是否小于fixed_width，如果是则返回空
        if radius < fixed_width/2.0:
            if self.verbose:
                print(f"BF策略（handle）：circle半径({radius:.1f}) < fixed_width/2.0({fixed_width/2.0:.1f})，返回空affordance")
            return [], None, None
        
        # 验证circle与bbox的交集比例
        intersection_ratio = self._calculate_circle_bbox_intersection(
            (cx, cy), radius, bag_bbox, mask.shape
        )
        
        if self.verbose:
            print(f"BF策略（handle）：最小外接圆 - center=({cx:.1f},{cy:.1f}), radius={radius:.1f}, area={handle_area:.1f}, 交集比例={intersection_ratio:.3f}")
        
        # 只有当交集比例大于60%时才返回affordance
        if intersection_ratio > 0.6:
            affordance = [float(cx), float(cy), float(radius)]
            handle_hole = [float(cx), float(cy), float(radius)]  # handle的圆也作为handle_hole
            if self.verbose:
                print(f"BF策略（handle）：交集比例满足条件(>{0.6:.1%})，返回affordance")
            return affordance, handle_hole, None
        else:
            if self.verbose:
                print(f"BF策略（handle）：交集比例不足(≤{0.6:.1%})，返回空affordance")
            return [], None, None
    
    def _process_bag_category(self, mask, bag_bbox, fixed_width):
        """处理bag类别的affordance - 寻找洞"""
        h, w = mask.shape
        
        # 使用HoleProcessingUtils进行渐进式孔洞检测
        mask_processed, optimal_kernel_size, has_large_hole = HoleProcessingUtils.detect_holes_progressive(
            mask, area_threshold=3000, max_kernel_size=49)
        
        # 用小kernel开操作去除噪点
        open_kernel_size = max(3, optimal_kernel_size // 5)
        open_kernel_size = open_kernel_size if open_kernel_size % 2 == 1 else open_kernel_size + 1
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
        mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_OPEN, open_kernel)
        
        if self.verbose:
            print(f"BF策略：渐进式预处理 - 最优膨胀-腐蚀kernel={optimal_kernel_size}, 开操作kernel={open_kernel_size}")
        
        # 使用HoleProcessingUtils提取孔洞轮廓
        hole_contours, holes_mask = HoleProcessingUtils.extract_hole_contours(mask_processed)
        
        if hole_contours and self.verbose:
            print(f"BF策略：检测到{len(hole_contours)}个孔洞轮廓")
        elif self.verbose:
            print(f"BF策略：预处理后仍未检测到孔洞轮廓")
        
        # 自适应形态学操作
        holes_mask = self._apply_adaptive_morphology(mask, holes_mask)
        
        # 如果找到洞，使用处理后的洞mask重新寻找轮廓
        if np.any(holes_mask):
            contours, _ = cv2.findContours(holes_mask.astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours = []
        
        if not contours and not hole_contours:
            if self.verbose:
                print(f"BF策略：未找到洞，返回空affordance")
            return [], None, None
        
        # 使用原始的hole_contours如果contours为空
        if not contours and hole_contours:
            contours = hole_contours
        
        # 找到最大的洞
        largest_hole = max(contours, key=cv2.contourArea)
        
        # 计算洞的面积
        hole_area = cv2.contourArea(largest_hole)
        mask_area = np.sum(mask)
        
        # 如果洞太小（小于mask的5%），返回空affordance
        if hole_area < mask_area * 0.05:
            if self.verbose:
                print(f"BF策略：洞太小 (面积比: {hole_area/mask_area:.3f})，返回空affordance")
            return [], None, None
        
        # 计算最大内接圆
        circle_cx, circle_cy, circle_radius = self._calculate_max_inscribed_circle(largest_hole)
        
        # 检查circle半径是否小于fixed_width，如果是则返回空
        if circle_radius < fixed_width/2.0:
            if self.verbose:
                print(f"BF策略：circle半径({circle_radius:.1f}) < fixed_width/2.0({fixed_width/2.0:.1f})，返回空affordance")
            return [], None, None
        
        # 创建handle_hole信息（圆形）
        handle_hole = [float(circle_cx), float(circle_cy), float(circle_radius)]
        if self.verbose:
            print(f"BF策略：检测到handle_hole - center=({circle_cx:.1f},{circle_cy:.1f}), radius={circle_radius:.1f}")
        
        # 使用独立函数进行基于circle和mask最高点的优化
        if self.verbose:
            print(f"BF策略：开始优化affordance - circle center=({circle_cx:.1f},{circle_cy:.1f}), radius={circle_radius:.1f}")
        optimization_result = self._optimize_affordance_with_circle_mask_intersection(
            circle_center=[circle_cx, circle_cy],
            circle_radius=circle_radius,
            mask=mask,
            bag_bbox=bag_bbox,
            fixed_width=38.0
        )
        
        if optimization_result[0] is not None:
            optimized_affordance, rot_bbox_bag = optimization_result
            if self.verbose:
                print(f"BF策略：使用优化后的affordance")
            return optimized_affordance, handle_hole, rot_bbox_bag
        else:
            # 如果优化失败，返回空
            affordance = []
            if self.verbose:
                print(f"BF策略：优化失败，返回空affordance")
            return affordance, handle_hole, None
    
    def _calculate_circle_bbox_intersection(self, center, radius, bbox, shape):
        """计算圆形与bbox的交集比例"""
        cx, cy = center
        x1, y1, x2, y2 = bbox[:4]
        h, w = shape
        
        # 计算圆形的面积
        circle_area = np.pi * radius * radius
        
        # 创建圆形mask来计算交集
        circle_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circle_mask, (int(cx), int(cy)), int(radius), 255, -1)
        
        # 创建bbox mask
        bbox_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(bbox_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
        
        # 计算交集
        intersection = np.logical_and(circle_mask > 0, bbox_mask > 0)
        intersection_area = np.sum(intersection)
        
        # 计算交集占圆形的比例
        intersection_ratio = intersection_area / np.sum(circle_mask > 0) if np.sum(circle_mask > 0) > 0 else 0
        
        return intersection_ratio
    
    def _apply_adaptive_morphology(self, mask, holes_mask):
        """应用自适应形态学操作"""
        h, w = mask.shape
        hole_ratio = 0  # 初始化hole_ratio
        
        if np.any(holes_mask):
            # 获取mask的尺寸
            mask_size = min(h, w)
            
            # 计算初始洞的面积（用于评估洞的大小）
            initial_hole_area = np.sum(holes_mask > 0)
            
            # 计算局部区域的mask面积
            if initial_hole_area > 0:
                # 获取洞的轮廓
                hole_contours_temp, _ = cv2.findContours(holes_mask.astype(np.uint8), 
                                                        cv2.RETR_EXTERNAL, 
                                                        cv2.CHAIN_APPROX_SIMPLE)
                if hole_contours_temp:
                    # 获取所有洞的边界框
                    x_min, y_min = h, w
                    x_max, y_max = 0, 0
                    for contour in hole_contours_temp:
                        x, y, w_box, h_box = cv2.boundingRect(contour)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x + w_box)
                        y_max = max(y_max, y + h_box)
                    
                    # 扩展边界框以包含周边区域（扩展20%）
                    expand_ratio = 0.2
                    w_expand = int((x_max - x_min) * expand_ratio)
                    h_expand = int((y_max - y_min) * expand_ratio)
                    
                    x_min = max(0, x_min - w_expand)
                    y_min = max(0, y_min - h_expand)
                    x_max = min(w, x_max + w_expand)
                    y_max = min(h, y_max + h_expand)
                    
                    # 计算局部区域的mask面积
                    local_mask = mask[y_min:y_max, x_min:x_max]
                    local_mask_area = np.sum(local_mask > 0)
                    
                    # 使用局部mask面积计算hole_ratio
                    hole_ratio = initial_hole_area / local_mask_area if local_mask_area > 0 else 0
                    
                    if self.verbose:
                        print(f"BF策略：洞区域 [{x_min},{y_min},{x_max},{y_max}], 局部mask面积={local_mask_area}, 洞面积={initial_hole_area}")
                else:
                    # 如果找不到轮廓，使用整个mask
                    mask_area = np.sum(mask > 0)
                    hole_ratio = initial_hole_area / mask_area if mask_area > 0 else 0
            else:
                # 没有洞，ratio为0
                hole_ratio = 0
            
            # 根据洞占局部mask的比例自适应调整kernel大小
            if hole_ratio > 0.40:
                # 特大洞：允许非常大的断裂连接
                kernel_size = max(17, int(mask_size * 0.07))
            elif hole_ratio > 0.25:
                # 大洞：允许较大的断裂连接
                kernel_size = max(13, int(mask_size * 0.05))
            elif hole_ratio > 0.15:
                # 中等洞：中等程度的断裂连接
                kernel_size = max(9, int(mask_size * 0.035))
            else:
                # 小洞：较小程度的断裂连接，但仍有一定容错
                kernel_size = max(7, int(mask_size * 0.025))
            
            # 确保kernel大小为奇数
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            # 提高最大kernel大小限制，增加容错性
            kernel_size = min(kernel_size, 25)
            
            if self.verbose:
                print(f"BF策略：自适应形态学 - 洞面积比={hole_ratio:.3f}, kernel大小={kernel_size}")
            
            # 多级形态学操作以增加断裂容错性
            
            # 第一级：使用较小的kernel进行初步连接
            if kernel_size > 7:
                pre_kernel_size = max(5, kernel_size // 2)
                pre_kernel_size = pre_kernel_size if pre_kernel_size % 2 == 1 else pre_kernel_size + 1
                pre_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pre_kernel_size, pre_kernel_size))
                holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_CLOSE, pre_kernel)
                if self.verbose:
                    print(f"BF策略：第一级形态学处理 - kernel={pre_kernel_size}")
            
            # 第二级：使用完整kernel进行主要连接
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_CLOSE, kernel)
            
            # 第三级：对于大断裂，使用膨胀-腐蚀策略
            if hole_ratio > 0.15 and kernel_size >= 9:
                # 使用更激进的膨胀来连接大断裂
                dilate_size = max(3, kernel_size // 3)
                dilate_size = dilate_size if dilate_size % 2 == 1 else dilate_size + 1
                dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
                
                # 膨胀操作连接断裂
                holes_mask = cv2.dilate(holes_mask, dilate_kernel, iterations=2)
                # 腐蚀恢复大小
                holes_mask = cv2.erode(holes_mask, dilate_kernel, iterations=2)
                if self.verbose:
                    print(f"BF策略：应用膨胀-腐蚀增强连接 - dilate_kernel={dilate_size}")
                
                # 最终平滑处理
                smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_CLOSE, smooth_kernel)
                
            # 对于特大洞，进行形状优化
            if hole_ratio > 0.30:
                # 使用开操作去除细小分支，保持主要结构
                open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_OPEN, open_kernel)
                if self.verbose:
                    print(f"BF策略：优化特大洞形状")
        else:
            # 如果没有检测到初始洞，使用默认参数
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_CLOSE, kernel)
        
        return holes_mask
    
    def _calculate_max_inscribed_circle(self, contour):
        """
        计算轮廓的最大内接圆
        
        Args:
            contour: 轮廓点
            
        Returns:
            tuple: (center_x, center_y, radius)
        """
        try:
            # 创建轮廓的mask
            bbox = cv2.boundingRect(contour)
            x, y, w, h = bbox
            
            # 创建一个稍大的区域来确保计算准确性
            margin = 10
            mask_w = w + 2 * margin
            mask_h = h + 2 * margin
            
            # 创建mask
            mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
            
            # 将轮廓坐标调整到新的坐标系
            adjusted_contour = contour.copy()
            adjusted_contour[:, :, 0] -= (x - margin)
            adjusted_contour[:, :, 1] -= (y - margin)
            
            # 填充轮廓
            cv2.fillPoly(mask, [adjusted_contour], 255)
            
            # 使用距离变换计算内接圆
            # 距离变换给出每个点到最近边界的距离
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            
            # 找到距离变换的最大值点，这就是最大内接圆的中心
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
            
            # 最大内接圆的半径就是最大距离值
            radius = max_val
            
            # 将坐标转换回原始坐标系
            center_x = max_loc[0] + (x - margin)
            center_y = max_loc[1] + (y - margin)
            
            if self.verbose:
                print(f"最大内接圆: center=({center_x:.1f},{center_y:.1f}), radius={radius:.1f}")
            
            return float(center_x), float(center_y), float(radius)
            
        except Exception as e:
            print(f"计算最大内接圆时发生错误: {e}")
            # 如果计算失败，退回到最小外接圆
            (cx, cy), r = cv2.minEnclosingCircle(contour)
            if self.verbose:
                print(f"退回到最小外接圆: center=({cx:.1f},{cy:.1f}), radius={r:.1f}")
            return float(cx), float(cy), float(r)
    
    def _optimize_affordance_with_circle_mask_intersection(self, circle_center, circle_radius, mask, bag_bbox=None, fixed_width=38.0):
        """
        基于mask的rot_bbox和circle_center计算优化的affordance
        
        Args:
            circle_center: 圆心坐标 [cx, cy]
            circle_radius: 圆半径
            mask: 二值掩码
            bag_bbox: 边界框 [x1, y1, x2, y2]，用于过滤affordance
            fixed_width: 固定短边宽度，默认38.0
            
        Returns:
            affordance: [xc, yc, w, h, angle] 或者 None（如果优化失败或被过滤）
            rot_bbox_bag: [xc, yc, w, h, angle] 或者 None
        """
        try:
            # 1. 计算mask的rot_bbox_bag
            mask_contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not mask_contours:
                if self.verbose:
                    print(f"优化失败：没有mask轮廓")
                return None, None
            
            # 获取最大的mask轮廓
            largest_mask_contour = max(mask_contours, key=cv2.contourArea)
            
            # 计算mask的最小外接旋转矩形
            rot_rect = cv2.minAreaRect(largest_mask_contour)
            (bag_cx, bag_cy), (bag_w, bag_h), bag_angle = rot_rect
            
            # 添加rot_rect选择逻辑：如果rot_rect和bag_bbox的夹角>阈值，rot_rect改用bag_bbox
            angle_threshold = 15.0  # 角度阈值（度）
            if bag_bbox is not None:
                # 将bag_bbox [x1, y1, x2, y2] 转换为rot_rect格式
                x1, y1, x2, y2 = bag_bbox[:4]
                bbox_cx = (x1 + x2) / 2
                bbox_cy = (y1 + y2) / 2
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                bbox_angle = 0.0  # 普通bbox的角度为0
                bag_bbox_rot_rect = ((bbox_cx, bbox_cy), (bbox_w, bbox_h), bbox_angle)
                
                # 计算两个矩形顶边的夹角
                angle_diff = GeometryUtils.calculate_rectangles_top_edge_angle(rot_rect, bag_bbox_rot_rect)
                if self.verbose:
                    print(f"rot_rect与bag_bbox顶边夹角: {angle_diff:.1f}度")
                
                # 如果夹角大于阈值，使用bag_bbox
                if angle_diff > angle_threshold:
                    if self.verbose:
                        print(f"夹角({angle_diff:.1f}度) > {angle_threshold}度阈值，使用bag_bbox代替rot_rect")
                    rot_rect = bag_bbox_rot_rect
                    (bag_cx, bag_cy), (bag_w, bag_h), bag_angle = rot_rect
                elif self.verbose:
                    print(f"夹角({angle_diff:.1f}度) <= {angle_threshold}度阈值，保持使用rot_rect")
            
            if self.verbose:
                print(f"最终选择的rot_bbox: center=({bag_cx:.1f},{bag_cy:.1f}), size=({bag_w:.1f}x{bag_h:.1f}), angle={bag_angle:.1f}deg")
            
            # 获取rot_bbox的四个顶点
            box_points = cv2.boxPoints(rot_rect)
            
            # 计算旋转矩形的两个主方向
            angle_rad = math.radians(bag_angle)
            
            # 方向1: 对应矩形的宽度方向
            dir1 = np.array([math.cos(angle_rad), math.sin(angle_rad)])
            # 方向2: 对应矩形的高度方向，垂直于方向1
            dir2 = np.array([-math.sin(angle_rad), math.cos(angle_rad)])
            
            # 找到所有边的两个中点（每对平行边的中点）
            # 计算每条边的中点
            edge_midpoints = []
            edge_directions = []
            for i in range(4):
                p1 = box_points[i]
                p2 = box_points[(i + 1) % 4]
                midpoint = (p1 + p2) / 2
                edge_vector = p2 - p1
                edge_direction = edge_vector / np.linalg.norm(edge_vector)
                edge_midpoints.append(midpoint)
                edge_directions.append(edge_direction)
            
            # 按边的中点y坐标进行分组，找到y值较小的边定义为edge1
            edges_with_y = []
            for i in range(4):
                edges_with_y.append((i, edge_midpoints[i][1]))  # (边索引, y坐标)
            
            # 按y坐标排序
            edges_with_y.sort(key=lambda x: x[1])
            
            # 选择y值最小的边作为edge1的参考
            edge1_idx = edges_with_y[0][0]  # y值最小的边
            edge1_direction = edge_directions[edge1_idx]
            
            # 确定edge1方向和垂直方向ev2
            ev1 = edge1_direction
            ev2 = np.array([-ev1[1], ev1[0]])  # 顺时针旋转90度得到垂直方向
            
            ev1_angle = math.degrees(math.atan2(ev1[1], ev1[0]))
            ev2_angle = math.degrees(math.atan2(ev2[1], ev2[0]))
            
            if self.verbose:
                print(f"edge1方向(y值较小的边): 角度={ev1_angle:.1f}deg")
                print(f"ev2方向(垂直方向): 角度={ev2_angle:.1f}deg")
            
            # 找到平行于edge1方向的所有边
            ev1_edges = []
            edge_y_values = []
            
            for i in range(4):
                p1 = box_points[i]
                p2 = box_points[(i + 1) % 4]
                edge_vector = p2 - p1
                edge_vector_norm = edge_vector / np.linalg.norm(edge_vector)
                
                # 检查边的方向是否与ev1平行（考虑正负方向）
                dot_product = abs(np.dot(edge_vector_norm, ev1))
                if dot_product > 0.9:  # 接近平行（考虑数值误差）
                    edge_midpoint_y = (p1[1] + p2[1]) / 2  # 计算边的中点y坐标
                    ev1_edges.append((p1, p2))
                    edge_y_values.append(edge_midpoint_y)
            
            if len(ev1_edges) == 0:
                if self.verbose:
                    print(f"优化失败：未找到平行于edge1方向的边")
                return None, None
            
            # 选择y值较小的边作为真正的edge1边
            if len(ev1_edges) >= 2:
                min_y_index = edge_y_values.index(min(edge_y_values))
                selected_edge = ev1_edges[min_y_index]
                ev1_edges = [selected_edge]  # 只使用y值最小的边
            
            # 2. 计算circle_center沿ev2方向（正向或反向）与edge1的交点o
            circle_center_array = np.array(circle_center)
            
            # 尝试ev2的正向和反向，找到最近的有效交点
            max_distance = max(bag_w, bag_h) * 3  # 足够长的射线长度
            
            intersection_point = None
            min_distance = float('inf')
            chosen_direction = None
            
            # 尝试ev2的正向和反向
            for direction_name, direction in [("正向", ev2), ("反向", -ev2)]:
                ray_end = circle_center_array + direction * max_distance
                
                # 检查射线与所有edge1方向边的交点
                for p1, p2 in ev1_edges:
                    intersect = GeometryUtils.line_intersect(circle_center_array, ray_end, p1, p2)
                    if intersect is not None:
                        intersect = np.array(intersect)
                        # 检查交点是否在射线的正方向上
                        to_intersect = intersect - circle_center_array
                        if np.dot(to_intersect, direction) > 0:  # 在指定方向上
                            distance = np.linalg.norm(to_intersect)
                            if distance < min_distance:
                                min_distance = distance
                                intersection_point = intersect
                                chosen_direction = direction
            
            if intersection_point is None:
                if self.verbose:
                    print(f"优化失败：未找到circle_center沿ev2方向与edge1的交点")
                return None, None
            
            ox, oy = intersection_point
            if self.verbose:
                print(f"最终选择交点o=({ox:.1f},{oy:.1f}), 距离circle_center={min_distance:.1f}")
            
            # 3. 定义aff_box是一个rot_bbox，使用几何点构建方法
            aff_center_x, aff_center_y = ox, oy
            
            # circle_center为中心点，沿ev1方向扩展得到width/2=38/2=19得到aff_box的p3和p4点
            half_width = fixed_width / 2.0  # 38/2 = 19
            
            # 计算p3和p4点：沿ev1方向从circle_center扩展±19
            p3 = circle_center_array + ev1 * half_width   # circle_center + ev1方向 * 19
            p4 = circle_center_array - ev1 * half_width   # circle_center - ev1方向 * 19
            
            # circle_center与o的连线方向cov，长度*2为l_co
            co_vector = intersection_point - circle_center_array
            l_co = min_distance * 2.0  # 长度*2
            cov_normalized = co_vector / np.linalg.norm(co_vector)  # 归一化方向向量
            
            # p4沿着cov方向和长度l_co得到p1
            p1 = p4 + cov_normalized * l_co
            
            # p3沿着cov方向和长度l_co得到p2  
            p2 = p3 + cov_normalized * l_co
            
            if self.verbose:
                print(f"构建aff_box的四个顶点:")
                print(f"  - p1: ({p1[0]:.1f},{p1[1]:.1f}) = p4 + cov*{l_co:.1f}")
                print(f"  - p2: ({p2[0]:.1f},{p2[1]:.1f}) = p3 + cov*{l_co:.1f}")
                print(f"  - p3: ({p3[0]:.1f},{p3[1]:.1f})")
                print(f"  - p4: ({p4[0]:.1f},{p4[1]:.1f})")
            
            # (p1,p2,p3,p4)构成aff_box的rot_bbox
            # 计算aff_box的中心、尺寸和角度
            box_points = np.array([p1, p2, p3, p4])
            
            # 计算宽度和高度
            aff_w = np.linalg.norm(p1 - p4)
            aff_h = np.linalg.norm(p3 - p4)
            
            # 角度：第一条边（宽度边）的方向，即cov方向与x轴的夹角
            aff_angle_rad = math.atan2(cov_normalized[1], cov_normalized[0])
            aff_angle_deg = math.degrees(aff_angle_rad)
            
            # 将角度转换到[-90, 0]范围
            if aff_angle_deg > 0:
                if aff_angle_deg <= 90:
                    aff_angle_deg = aff_angle_deg - 90
                else:
                    aff_angle_deg = aff_angle_deg - 180
            else:
                if aff_angle_deg >= -90:
                    pass  # 已经在范围内
                else:
                    aff_angle_deg = aff_angle_deg + 180
            
            if aff_angle_deg < 0: 
                aff_angle_deg += 180  # 转换到[0, 180] 
            
            aff_angle_rad = math.radians(aff_angle_deg)
            
            if self.verbose:
                print(f"aff_box构建完成:")
                print(f"  - aff_center(o): ({aff_center_x:.1f},{aff_center_y:.1f})")
                print(f"  - circle_center: ({circle_center[0]:.1f},{circle_center[1]:.1f})")
                print(f"  - 宽度w(p1-p4距离): {aff_w:.1f}")
                print(f"  - 高度h(p3-p4距离): {aff_h:.1f}")
                print(f"  - angle(cov方向与x轴夹角): {aff_angle_deg:.1f}deg ({aff_angle_rad:.3f}rad)")
            
            # 构建rot_bbox_bag信息 [xc, yc, w, h, angle_in_radians]
            rot_bbox_bag = [float(bag_cx), float(bag_cy), float(bag_w), float(bag_h), float(angle_rad)]
            
            # 返回旋转矩形格式的affordance [xc, yc, w, h, angle_in_radians]
            affordance = [float(aff_center_x), float(aff_center_y), float(aff_w), float(aff_h), float(aff_angle_rad)]
            
            # 过滤：如果affordance和bbox的交集/affordance > 80%，返回空
            if bag_bbox is not None:
                intersection_ratio = GeometryUtils.calculate_rotated_bbox_intersection_ratio(affordance, bag_bbox)
                if self.verbose:
                    print(f"affordance与bbox交集比例: {intersection_ratio:.1%}")
                if intersection_ratio > 0.8:
                    if self.verbose:
                        print(f"affordance与bbox交集比例过高(>{0.8:.1%})，返回空")
                    return None, None
            
            # 过滤：基于affordance和mask的重叠度
            aff_mask_overlap_threshold = 0.60  # 60% 阈值
            if bag_bbox is not None:
                # 计算affordance、bbox交集 overlap1
                overlap1_mask = GeometryUtils.calculate_rotated_rect_bbox_intersection_mask(
                    affordance, bag_bbox, mask.shape)
                
                # 计算overlap1、mask的交集 overlap2
                overlap2_mask = GeometryUtils.calculate_mask_intersection(overlap1_mask, mask * 255)
                
                # 计算overlap2/overlap1比例
                overlap_ratio = GeometryUtils.calculate_mask_overlap_ratio(overlap1_mask, overlap2_mask)
                if self.verbose:
                    print(f"affordance-bbox-mask重叠比例: {overlap_ratio:.1%}")
                
                if overlap_ratio > aff_mask_overlap_threshold:
                    if self.verbose:
                        print(f"affordance-bbox-mask重叠比例过高(>{aff_mask_overlap_threshold:.1%})，返回空")
                    return None, None
            
            return affordance, rot_bbox_bag
            
        except Exception as e:
            print(f"优化affordance时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def process_frame(self, frame: np.ndarray, detection_result: Dict, image_id: Optional[int] = None) -> List[Dict]:
        """
        处理单帧图像，计算BF视角的affordance
        
        Args:
            frame: RGB图像帧
            detection_result: 检测结果字典
            image_id: 图像ID（可选）
            
        Returns:
            包含affordance的标注列表
        """
        annotations = []
        
        if not detection_result or 'objects' not in detection_result:
            return annotations
        
        for obj in detection_result.get('objects', []):
            bbox = obj.get('bbox', [])
            mask_rle = obj.get('mask', None)
            category = obj.get('category', 'unknown').lower()
            score = obj.get('score', 0.0)
            
            # 确定类别ID
            if 'bag' in category:
                category_id = 0
            elif 'handle' in category:
                category_id = 1
                continue   # 对handle类别不进行标注
            else:
                continue
            
            if len(bbox) >= 4:
                # 计算BF affordance
                affordance, handle_hole, rot_bbox_bag = self._process_single(
                    mask_rle, bbox, frame, image_id, category
                )
                
                # 验证affordance格式
                if not affordance or (isinstance(affordance, list) and len(affordance) != 5):
                    continue
                
                # 转换bbox格式 [x1, y1, x2, y2] -> [x, y, width, height]
                x1, y1, x2, y2 = bbox
                bbox_coco = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                area = float((x2 - x1) * (y2 - y1))
                
                annotation = {
                    'bbox': bbox_coco,
                    'category_id': int(category_id),
                    'area': area,
                    'score': float(score),
                    'affordance': [affordance],  # 包装成列表
                    'grasp_policy': 'bf',
                    'handle_hole': handle_hole if handle_hole else None,
                    'rot_bbox_bag': rot_bbox_bag if rot_bbox_bag else None,
                    'segmentation': mask_rle if mask_rle else None
                }
                
                annotations.append(annotation)
        
        return annotations
    
    def test_single_sample(self, rgb_path: str, mask_path: str = None):
        """
        测试单个样本的affordance计算
        
        Args:
            rgb_path: RGB图像路径
            mask_path: mask文件路径（可选）
        """
        # 读取图像
        frame = cv2.imread(rgb_path)
        if frame is None:
            print(f"无法读取图像: {rgb_path}")
            return
        
        print(f"测试图像: {rgb_path}")
        print(f"图像尺寸: {frame.shape}")
        
        # 如果没有mask，使用检测API
        if mask_path is None and self.detection_api is not None:
            print("使用检测API检测bag.handle...")
            result = self.detection_api.detect_objects(
                rgb=frame,
                prompt_text="bag.handle",
                bbox_threshold=0.25,
                iou_threshold=0.8
            )
            
            if result and 'objects' in result:
                objects = result.get('objects', [])
                print(f"检测到 {len(objects)} 个对象")
                
                for idx, obj in enumerate(objects):
                    bbox = obj.get('bbox', [])
                    mask_rle = obj.get('mask', None)
                    category = obj.get('category', 'unknown').lower()
                    score = obj.get('score', 0.0)
                    
                    print(f"\n对象 {idx+1}:")
                    print(f"  类别: {category}")
                    print(f"  置信度: {score:.3f}")
                    print(f"  边界框: {bbox}")
                    
                    if 'bag' in category and mask_rle:
                        # 计算affordance
                        affordance, handle_hole, rot_bbox_bag = self._process_single(
                            mask_rle, bbox, frame, image_id=1, category=category
                        )
                        
                        print(f"  Affordance: {affordance}")
                        print(f"  Handle hole: {handle_hole}")
                        print(f"  Rot bbox bag: {rot_bbox_bag}")
            else:
                print("未检测到对象")
        else:
            print("需要提供mask文件或启用检测API")
    
    def _save_visualization(self, rgb_image: np.ndarray, rgb_file: Path, result: Dict):
        """
        保存可视化结果到pv_vis目录
        
        Args:
            rgb_image: 原始RGB图像
            rgb_file: 图像文件路径
            result: 处理结果字典
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Rectangle, Circle, Polygon as MplPolygon
        import numpy as np
        from pycocotools import mask as coco_mask
        import cv2
        
        try:
            # 创建可视化图像
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            
            # 显示原始图像 (BGR to RGB)
            rgb_display = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb_display)
            
            # 绘制检测结果和masks
            for i, detection in enumerate(result['detections']):
                bbox = detection.get('bbox', [])
                mask_rle = detection.get('mask', None)
                label = detection.get('label', 'bag')
                confidence = detection.get('confidence', 0.0)
                
                # 绘制mask（如果存在）
                if mask_rle and isinstance(mask_rle, dict) and 'counts' in mask_rle:
                    try:
                        # 解码mask
                        mask_array = coco_mask.decode(mask_rle)
                        # 创建透明的彩色mask
                        mask_colored = np.zeros((mask_array.shape[0], mask_array.shape[1], 4))
                        mask_colored[:, :, 0] = 1.0  # 红色通道
                        mask_colored[:, :, 3] = mask_array * 0.3  # 透明度
                        ax.imshow(mask_colored, alpha=0.5)
                        
                        # 绘制mask边界
                        import cv2
                        contours, _ = cv2.findContours(mask_array.astype(np.uint8), 
                                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            if len(contour) > 2:
                                contour_points = contour.reshape(-1, 2)
                                # 闭合轮廓
                                contour_points = np.vstack([contour_points, contour_points[0]])
                                ax.plot(contour_points[:, 0], contour_points[:, 1], 
                                       color='red', linewidth=2, alpha=0.8)
                    except Exception as e:
                        print(f"Warning: Failed to draw mask for detection {i}: {e}")
                
                # 绘制检测框
                if len(bbox) >= 4:
                    # bbox格式是[x, y, width, height]
                    x, y, width, height = bbox[:4]
                    rect = Rectangle((x, y), width, height, 
                                   linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
                    ax.add_patch(rect)
                    
                    # 标注检测信息
                    ax.text(x, y-5, f"{label} {confidence:.2f}", 
                           color='red', fontsize=10, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # 绘制affordances和相关信息
            for i, aff_result in enumerate(result['affordances']):
                affordance = aff_result.get('affordance', [])
                handle_hole = aff_result.get('handle_hole', None)
                rot_bbox_bag = aff_result.get('rot_bbox_bag', None)
                
                # 绘制affordance矩形
                if affordance and len(affordance) >= 5:
                    cx, cy, w, h, angle = affordance[:5]
                    
                    # 计算旋转矩形的四个顶点
                    cos_a = np.cos(angle)
                    sin_a = np.sin(angle)
                    
                    # 矩形的四个角点（相对于中心）
                    dx, dy = w/2, h/2
                    corners = np.array([
                        [-dx, -dy],
                        [dx, -dy], 
                        [dx, dy],
                        [-dx, dy]
                    ])
                    
                    # 旋转并平移到实际位置
                    rotated_corners = []
                    for corner in corners:
                        x_rot = corner[0] * cos_a - corner[1] * sin_a + cx
                        y_rot = corner[0] * sin_a + corner[1] * cos_a + cy
                        rotated_corners.append([x_rot, y_rot])
                    
                    # 绘制affordance矩形
                    rotated_corners = np.array(rotated_corners)
                    polygon = MplPolygon(rotated_corners, fill=False, 
                                        edgecolor='blue', linewidth=3, alpha=0.9)
                    ax.add_patch(polygon)
                    
                    # 标注affordance中心
                    ax.plot(cx, cy, 'bo', markersize=8)
                    ax.text(cx+10, cy-10, f"Aff {i+1}", color='blue', 
                           fontsize=10, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
                
                # 绘制handle hole
                if handle_hole and len(handle_hole) >= 3:
                    hx, hy, radius = handle_hole[:3]
                    circle = Circle((hx, hy), radius, fill=False, 
                                  edgecolor='green', linewidth=3, alpha=0.9)
                    ax.add_patch(circle)
                    
                    # 标注handle hole中心
                    ax.plot(hx, hy, 'go', markersize=8)
                    ax.text(hx+10, hy+10, f"Handle {i+1}", color='green', 
                           fontsize=10, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
                
                # 绘制rot_bbox_bag（包的旋转边界框）
                if rot_bbox_bag and len(rot_bbox_bag) >= 5:
                    bag_cx, bag_cy, bag_w, bag_h, bag_angle = rot_bbox_bag[:5]
                    
                    # 计算旋转矩形的四个顶点
                    cos_a = np.cos(bag_angle)
                    sin_a = np.sin(bag_angle)
                    
                    # 矩形的四个角点（相对于中心）
                    dx, dy = bag_w/2, bag_h/2
                    corners = np.array([
                        [-dx, -dy],
                        [dx, -dy], 
                        [dx, dy],
                        [-dx, dy]
                    ])
                    
                    # 旋转并平移到实际位置
                    rotated_corners = []
                    for corner in corners:
                        x_rot = corner[0] * cos_a - corner[1] * sin_a + bag_cx
                        y_rot = corner[0] * sin_a + corner[1] * cos_a + bag_cy
                        rotated_corners.append([x_rot, y_rot])
                    
                    # 绘制rot_bbox_bag矩形
                    rotated_corners = np.array(rotated_corners)
                    polygon = MplPolygon(rotated_corners, fill=False, 
                                        edgecolor='orange', linewidth=2, alpha=0.7, linestyle='--')
                    ax.add_patch(polygon)
                    
                    # 标注rot_bbox_bag中心
                    ax.plot(bag_cx, bag_cy, 'o', color='orange', markersize=6)
                    ax.text(bag_cx-20, bag_cy+20, f"Bag {i+1}", color='orange', 
                           fontsize=9, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
            
            # 设置图像属性
            ax.set_xlim(0, rgb_image.shape[1])
            ax.set_ylim(rgb_image.shape[0], 0)  # 翻转y轴
            ax.set_title(f"PV Affordance Visualization - {rgb_file.name}", fontsize=16, weight='bold')
            ax.axis('off')
            
            # 添加图例
            legend_elements = [
                mpatches.Patch(color='red', label='Detection + Mask'),
                mpatches.Patch(color='blue', label='Affordance'),
                mpatches.Patch(color='green', label='Handle Hole'),
                mpatches.Patch(color='orange', label='Rot Bbox Bag')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
            
            # 显示统计信息
            detections = result.get('detections', [])
            affordances = result.get('affordances', [])
            
            # 统计类别信息
            category_counts = {}
            for detection in detections:
                cat = detection.get('category', 'bag')
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            category_text = ", ".join([f"{cat}: {count}" for cat, count in category_counts.items()])
            graspable = len([aff for aff in affordances if aff.get('graspable', True)])
            
            info_text1 = f"Detected: {len(detections)} ({category_text})"
            info_text2 = f"Graspable: {graspable}"
            
            # 在图像上添加统计信息文本
            ax.text(10, 30, info_text1, fontsize=12, color='white', weight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.7))
            ax.text(10, 60, info_text2, fontsize=12, color='white', weight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.7))
            
            # 保存可视化结果
            vis_filename = f"vis_{rgb_file.stem}.jpg"
            vis_path = self.output_dir / vis_filename
            
            plt.tight_layout()
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            if self.verbose:
                print(f"  Visualization saved: {vis_path}")
                
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Failed to save visualization: {e}")


def main():
    """测试主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate PV Grasp Affordance for Bags')
    parser.add_argument('--rgb', type=str, 
                       default='data/bag/0912_backfront/images/rgb/rgb_frame_000000.jpg',
                       help='RGB image file or directory')
    parser.add_argument('--text', type=str, 
                       default="paper bag. plastic bag. bag",
                       help='Detection text prompt')
    parser.add_argument('--output', type=str, default='vis_aff_pv',
                       help='Output directory for visualizations')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching')
    parser.add_argument('--cache-dir', type=str, default='.cache',
                       help='Cache directory')
    
    args = parser.parse_args()
    
    # 创建PV Affordance生成器
    pv_affordance = Affordance4PVGraspBag(
        detection_text=args.text,
        output_dir=args.output,
        verbose=args.verbose,
        enable_cache=not args.no_cache,
        cache_dir=args.cache_dir
    )
    
    # 处理输入
    results = pv_affordance.process(args.rgb)
    
    # 输出总结
    if isinstance(results, dict) and results:
        if 'detections' in results:
            # 单文件结果
            print(f"\nSummary:")
            print(f"  Detections: {len(results['detections'])}")
            print(f"  Affordances: {len(results['affordances'])}")
        else:
            # 多文件结果
            total_dets = sum(len(r.get('detections', [])) for r in results.values())
            total_affs = sum(len(r.get('affordances', [])) for r in results.values())
            print(f"\nSummary:")
            print(f"  Frames processed: {len(results)}")
            print(f"  Total detections: {total_dets}")
            print(f"  Total affordances: {total_affs}")
            print(f"  Output saved to: {args.output}/")


if __name__ == "__main__":
    main()