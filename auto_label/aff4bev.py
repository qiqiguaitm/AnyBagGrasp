#!/usr/bin/env python3
"""
Affordance for BEV (Bird's Eye View) Grasp Bag
生成俯视抓取Bag的Affordance
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import time
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 添加必要的路径
sys.path.insert(0, str(Path(__file__).parent.parent / "cdm"))

# 导入检测API
try:
    from dino_any_percept_api import DetectionAPI
    from cache_utils import CachedDetectionAPI
    DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Detection API not available: {e}")
    DETECTION_AVAILABLE = False
    DetectionAPI = None
    CachedDetectionAPI = None

# 导入CDM处理
try:
    from simple_cdm_fixed import process_depth_with_cdm
    CDM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CDM processor not available: {e}")
    CDM_AVAILABLE = False
    process_depth_with_cdm = lambda rgb, depth: depth  # Fallback to raw depth

# 导入CDM缓存
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "cdm"))
    from cdm_cache import CDMCache
    CDM_CACHE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CDM cache not available: {e}")
    CDM_CACHE_AVAILABLE = False
    CDMCache = None


class Affordance4BEVGraspBag:
    """
    生成俯视抓取Bag的Affordance类
    """
    
    def __init__(self, 
                 detection_text: str = "paper bag.plastic bag.bag",
                 output_dir: str = "vis_aff",
                 verbose: bool = True,
                 enable_cache: bool = True,
                 cache_dir: str = ".cache",
                 save_individual: bool = False):
        """
        初始化Affordance生成器
        
        Args:
            detection_text: 检测目标的文本描述
            output_dir: 输出可视化结果的目录
            verbose: 是否输出详细信息
            enable_cache: 是否启用缓存
            cache_dir: 缓存目录
            save_individual: 是否保存单个检测的可视化图像
        """
        self.detection_text = detection_text
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        self.save_individual = save_individual
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化检测API（带缓存）
        if DETECTION_AVAILABLE:
            base_detector = self._init_detection_api()
            if self.enable_cache and CachedDetectionAPI:
                self.detector = CachedDetectionAPI(
                    base_detector,
                    cache_dir=f"{self.cache_dir}/detection",
                    enable_cache=True
                )
                if self.verbose:
                    print(f"Detection API initialized with caching enabled")
            else:
                self.detector = base_detector
                if self.verbose:
                    print(f"Detection API initialized without caching")
            if self.verbose:
                print(f"Detection text: {detection_text}")
        else:
            self.detector = None
            print("Warning: Detection API not available, using dummy detections")
        
        # CDM状态和缓存
        self.cdm_available = CDM_AVAILABLE
        if self.enable_cache and CDM_CACHE_AVAILABLE:
            self.cdm_cache = CDMCache(cache_dir=f"{self.cache_dir}/cdm")
            if self.verbose:
                print(f"CDM processor: Available with caching")
        else:
            self.cdm_cache = None
            if self.verbose:
                print(f"CDM processor: {'Available' if self.cdm_available else 'Not available'} without caching")
    
    def _init_detection_api(self):
        """初始化DetectionAPI"""
        from mmengine.config import Config
        
        cfg = Config()
        cfg.uri = r'/v2/task/dinox/detection'
        cfg.status_uri = r'/v2/task_status'  
        cfg.token = 'c4cdacb48bc4d1a1a335c88598a18e8c'
        cfg.model_name = 'DINO-X-1.0'
        
        print("初始化DetectionAPI...")
        return DetectionAPI(cfg)
    
    def process(self, 
                rgb_input: Union[str, Path],
                depth_input: Union[str, Path]) -> Dict:
        """
        处理输入，可以是文件或目录
        
        Args:
            rgb_input: RGB图像文件或目录路径
            depth_input: 深度数据文件或目录路径
            
        Returns:
            处理结果字典
        """
        rgb_path = Path(rgb_input)
        depth_path = Path(depth_input)
        
        # 判断输入类型
        if rgb_path.is_file() and depth_path.is_file():
            # 处理单个文件
            if self.verbose:
                print(f"\nProcessing single file pair:")
                print(f"  RGB: {rgb_path}")
                print(f"  Depth: {depth_path}")
            return self._process_single(rgb_path, depth_path)
        
        elif rgb_path.is_dir() and depth_path.is_dir():
            # 处理目录
            if self.verbose:
                print(f"\nProcessing directories:")
                print(f"  RGB dir: {rgb_path}")
                print(f"  Depth dir: {depth_path}")
            return self._process_directory(rgb_path, depth_path)
        
        else:
            raise ValueError("RGB and depth inputs must both be files or both be directories")
    
    def _process_directory(self, rgb_dir: Path, depth_dir: Path) -> Dict:
        """
        处理目录中的所有文件对
        """
        # 获取RGB文件列表
        rgb_files = sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
        
        if not rgb_files:
            print(f"Warning: No RGB files found in {rgb_dir}")
            return {}
        
        results = {}
        
        for rgb_file in rgb_files:
            # 构建对应的深度文件名
            # 假设命名规则: rgb_frame_000000.jpg -> depth_frame_000000.npy
            frame_id = rgb_file.stem.split('_')[-1]  # 获取帧编号
            depth_file = depth_dir / f"depth_frame_{frame_id}.npy"
            
            if not depth_file.exists():
                print(f"Warning: Depth file not found for {rgb_file.name}, skipping")
                continue
            
            # 处理文件对
            frame_result = self._process_single(rgb_file, depth_file)
            results[frame_id] = frame_result
            
            if self.verbose:
                print(f"Processed frame {frame_id}: {len(frame_result.get('detections', []))} bags detected")
        
        return results
    

    def _process_single(self, rgb_file: Path, depth_file: Path) -> Dict:
        """
        处理单个RGB-深度文件对
        """
        result = {
            'rgb_file': str(rgb_file),
            'depth_file': str(depth_file),
            'detections': [],
            'affordances': []
        }
        
        # 1. 加载RGB和深度数据
        rgb_image = cv2.imread(str(rgb_file))
        if rgb_image is None:
            print(f"Error: Cannot load RGB image {rgb_file}")
            return result
        
        depth_raw = np.load(str(depth_file))
        
        h, w = rgb_image.shape[:2]
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {rgb_file.name}")
            print(f"Image size: {w}x{h}")
            valid_depths = depth_raw[depth_raw>0]
            if len(valid_depths) > 0:
                print(f"Depth range: {valid_depths.min():.3f} - {valid_depths.max():.3f} m")
        
        # 2. 物体检测
        target_dets = self._detect_objects(rgb_image)
        result['detections'] = target_dets
        
        if not target_dets:
            if self.verbose:
                print("No bags detected in image")
            return result
        
        if self.verbose:
            print(f"Detected {len(target_dets)} bag(s)")
        
        # 3. CDM深度增强
        t_start = time.time()
        cdm_depth = self._enhance_depth_with_cdm(rgb_image, depth_raw)
        t_cdm = (time.time() - t_start) * 1000
        
        if self.verbose:
            print(f"CDM depth enhancement: {t_cdm:.2f} ms")
        
        # 4. 对每个检测目标进行处理
        for idx, det in enumerate(target_dets):
            if self.verbose:
                print(f"\nProcessing detection {idx+1}/{len(target_dets)}:")
                print(f"  Bbox: {det['bbox']}")
                print(f"  Confidence: {det.get('confidence', 0):.3f}")
            
            # 提取目标区域的深度（使用mask如果可用）
            target_depth = self._extract_target_depth_with_mask(cdm_depth, det, idx)
            
            # 5. 计算affordance
            affordance = self._compute_affordance(det, target_depth)
            
            # 检查affordance中心点是否在边界5%区域内
            if affordance.get('grasp_center'):
                grasp_center = affordance['grasp_center']
                # 边界5%阈值
                boundary_ratio = 0.05
                min_x = w * boundary_ratio
                max_x = w * (1 - boundary_ratio)
                min_y = h * boundary_ratio  
                max_y = h * (1 - boundary_ratio)
                
                # 检查是否在边界区域
                if (grasp_center[0] < min_x or grasp_center[0] > max_x or
                    grasp_center[1] < min_y or grasp_center[1] > max_y):
                    # 标记为边界区域内的affordance
                    affordance['near_boundary'] = True
                    if self.verbose:
                        print(f"  Affordance discarded: center {grasp_center} is within 5% boundary")
                    # 清除所有抓取相关信息
                    affordance['grasp_center'] = None
                    affordance['grasp_angle'] = None
                    affordance['quality_score'] = 0.0
                    affordance['width_feasible'] = False
                    affordance['rot_bbox'] = None
                    affordance['aff_rot_bbox'] = []  # 空列表
                    affordance['left_finger'] = None
                    affordance['right_finger'] = None
                    affordance['gripper_body_bbox'] = None
                    affordance['gripper_open_bbox'] = None
                else:
                    affordance['near_boundary'] = False
            
            result['affordances'].append(affordance)
            
            # 可视化结果
            vis_path = self._visualize_result(
                rgb_image, cdm_depth, det, 
                rgb_file.stem, idx, affordance
            )
            
            if self.verbose:
                if affordance.get('grasp_center') and not affordance.get('near_boundary', False):
                    print(f"  Grasp center: {affordance['grasp_center']}")
                    print(f"  Object width: {affordance.get('object_width', 0):.1f}mm")
                    print(f"  Grasp width: {affordance.get('grasp_width_actual', 0):.1f}mm")
                    print(f"  Grasp quality: {affordance.get('quality_score', 0):.3f}")
                    print(f"  Width feasible: {affordance.get('width_feasible', False)}")
                if vis_path and self.save_individual:
                    print(f"  Visualization saved: {vis_path}")
        
        # 生成全局可视化
        all_vis_path = str(self.output_dir / f"{rgb_file.stem}_all_results.png")
        self._visualize_all_results(
            rgb_image, result['detections'], result['affordances'],
            cdm_depth, all_vis_path
        )
        
        return result
    
    def _detect_objects(self, rgb_image: np.ndarray) -> List[Dict]:
        """
        使用检测API检测物体
        """
        if not DETECTION_AVAILABLE or self.detector is None:
            # 检测不可用时返回空结果
            return []
        
        # 实际检测 - 使用detect_objects方法
        try:
            result = self.detector.detect_objects(
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
            # 异常情况返回空结果
            return []
    
    def _enhance_depth_with_cdm(self, rgb_image: np.ndarray, depth_raw: np.ndarray) -> np.ndarray:
        """
        使用CDM增强深度（带缓存）
        """
        if not self.cdm_available:
            return depth_raw
        
        # 转换BGR到RGB for CDM
        rgb_for_cdm = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # 尝试从缓存获取
        if self.cdm_cache:
            cached_result = self.cdm_cache.get(rgb_for_cdm, depth_raw)
            if cached_result is not None:
                return cached_result
        
        try:
            # 执行CDM处理
            cdm_depth = process_depth_with_cdm(rgb_for_cdm, depth_raw)
            
            if cdm_depth is not None:
                # 保存到缓存
                if self.cdm_cache:
                    self.cdm_cache.set(rgb_for_cdm, depth_raw, cdm_depth)
                return cdm_depth
            else:
                print("Warning: CDM processing failed, using raw depth")
                return depth_raw
        except Exception as e:
            print(f"Error in CDM processing: {e}")
            return depth_raw
    
    def _extract_target_depth(self, depth_image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        提取目标区域的深度信息
        
        Args:
            depth_image: 完整深度图
            bbox: [x, y, width, height]
            
        Returns:
            目标区域的深度数据
        """
        x, y, w, h = bbox
        
        # 转换为整数并确保边界框在图像范围内
        h_img, w_img = depth_image.shape
        x = int(max(0, min(x, w_img)))
        y = int(max(0, min(y, h_img)))
        w = int(min(w, w_img - x))
        h = int(min(h, h_img - y))
        
        # 提取区域
        target_depth = depth_image[y:y+h, x:x+w].copy()
        
        return target_depth
    
    def _extract_target_depth_with_mask(self, depth_image: np.ndarray, det: Dict, det_idx: int = -1) -> np.ndarray:
        """
        使用mask提取目标区域的深度信息
        
        Args:
            depth_image: 完整深度图
            det: 检测结果，包含bbox和可选的mask
            
        Returns:
            目标区域的深度数据（仅mask内的区域有效，其他为0）
        """
        bbox = det['bbox']
        x, y, w, h = bbox
        
        # 转换为整数并确保边界框在图像范围内
        h_img, w_img = depth_image.shape
        x = int(max(0, min(x, w_img)))
        y = int(max(0, min(y, h_img)))
        w = int(min(w, w_img - x))
        h = int(min(h, h_img - y))
        
        # 提取bbox区域
        target_depth = depth_image[y:y+h, x:x+w].copy()
        
        # 如果有mask，应用mask
        if 'mask' in det and det['mask'] is not None:
            try:
                from pycocotools import mask as coco_mask
                # 解码mask
                mask_rle = det['mask']
                full_mask = coco_mask.decode(mask_rle)
                
                # 提取对应bbox区域的mask
                mask_region = full_mask[y:y+h, x:x+w]
                
                # 应用mask - 只保留mask内的深度值
                target_depth[mask_region == 0] = 0
            except Exception as e:
                print(f"Warning: Failed to apply mask: {e}")
                # 如果mask处理失败，返回整个bbox区域
        
        return target_depth
    
    
    def _colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        将深度图像转换为彩色可视化
        
        Args:
            depth: 深度图像
            
        Returns:
            彩色深度图像
        """
        # 归一化深度到0-255
        valid_mask = depth > 0
        if np.any(valid_mask):
            depth_norm = depth.copy()
            depth_norm[valid_mask] = (depth_norm[valid_mask] - depth_norm[valid_mask].min()) / \
                                    (depth_norm[valid_mask].max() - depth_norm[valid_mask].min())
            depth_norm[~valid_mask] = 0
            
            # 转换为彩色图
            depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            depth_color[~valid_mask] = [0, 0, 0]
            return depth_color
        else:
            return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    
    def _visualize_all_results(self, image: np.ndarray, detections: List[Dict], 
                              affordances: List[Dict], depth_image: np.ndarray,
                              output_path: str):
        """
        可视化所有检测和抓取结果在一张图上
        
        Args:
            image: RGB图像
            detections: 所有检测结果
            affordances: 所有affordance结果
            depth_image: 深度图像
            output_path: 输出路径
        """
        # 创建可视化图像
        vis_rgb = image.copy()
        vis_depth = self._colorize_depth(depth_image)
        
        # 创建mask叠加层（RGB和深度分别创建）
        mask_overlay_rgb = np.zeros_like(vis_rgb)
        mask_overlay_depth = np.zeros_like(vis_depth)
        
        # 绘制所有检测框和抓取点
        for i, (det, aff) in enumerate(zip(detections, affordances)):
            bbox = det['bbox']
            x, y, w, h = [int(v) for v in bbox]
            
            # 选择颜色（使用不同颜色区分不同检测）
            colors = [
                (0, 255, 0),    # 绿色
                (255, 0, 0),    # 蓝色
                (0, 0, 255),    # 红色
                (255, 255, 0),  # 青色
                (255, 0, 255),  # 品红
                (0, 255, 255),  # 黄色
                (128, 255, 128),# 浅绿
                (128, 128, 255),# 浅红
            ]
            color = colors[i % len(colors)]
            
            # 绘制mask（如果有）
            if 'mask' in det and det['mask']:
                try:
                    from pycocotools import mask as mask_util
                    # 解码COCO RLE格式的mask
                    if isinstance(det['mask'], dict) and 'counts' in det['mask']:
                        binary_mask = mask_util.decode(det['mask'])
                        # 在叠加层上绘制mask
                        mask_color = np.array(color, dtype=np.uint8)
                        mask_overlay_rgb[binary_mask > 0] = mask_color
                        mask_overlay_depth[binary_mask > 0] = mask_color
                        
                        # 绘制mask的轮廓以突出边界（使用更亮的颜色和更粗的线）
                        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), 
                                                       cv2.RETR_EXTERNAL, 
                                                       cv2.CHAIN_APPROX_SIMPLE)
                        # 使用更亮的颜色画轮廓
                        bright_color = tuple(min(255, int(c * 1.5)) for c in color)
                        cv2.drawContours(vis_rgb, contours, -1, bright_color, 4)  # 更粗的轮廓线
                        cv2.drawContours(vis_depth, contours, -1, bright_color, 4)
                        # 再画一条细的白色边界增加对比度
                        cv2.drawContours(vis_rgb, contours, -1, (255, 255, 255), 1)
                        cv2.drawContours(vis_depth, contours, -1, (255, 255, 255), 1)
                except ImportError:
                    pass  # 如果没有pycocotools，跳过mask绘制
            
            # 绘制边界框（加粗）
            cv2.rectangle(vis_rgb, (x, y), (x+w, y+h), color, 3)
            cv2.rectangle(vis_depth, (x, y), (x+w, y+h), color, 3)
            
            # 绘制检测标签（包含类别信息）
            category = det.get('label', 'unknown')
            confidence = det.get('confidence', det.get('score', 0))
            label = f"{category} ({confidence:.2f})"
            
            # 计算标签尺寸
            font_scale = 0.6
            font_thickness = 2
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            # 绘制标签背景
            label_y_offset = 25
            cv2.rectangle(vis_rgb, (x, y-label_y_offset), (x+label_size[0]+8, y-label_y_offset+label_size[1]+8), color, -1)
            cv2.rectangle(vis_depth, (x, y-label_y_offset), (x+label_size[0]+8, y-label_y_offset+label_size[1]+8), color, -1)
            
            # 绘制标签文字
            cv2.putText(vis_rgb, label, (x+4, y-label_y_offset+label_size[1]+4), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            cv2.putText(vis_depth, label, (x+4, y-label_y_offset+label_size[1]+4), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            
            # 只有在抓取点有效时才绘制旋转边界框
            if aff.get('grasp_center') is not None and not aff.get('near_boundary', False):
                # 绘制旋转边界框（如果有）
                if aff.get('rot_bbox') is not None:
                    # rot_bbox应该包含旋转矩形的4个顶点
                    rot_box = np.array(aff['rot_bbox'], dtype=np.int32)
                    # 顶部区域的rot_bbox使用细线，半透明效果
                    overlay_rgb = vis_rgb.copy()
                    overlay_depth = vis_depth.copy()
                    cv2.drawContours(overlay_rgb, [rot_box], 0, color, 2)  # 细线
                    cv2.drawContours(overlay_depth, [rot_box], 0, color, 2)
                    # 半透明叠加
                    vis_rgb = cv2.addWeighted(vis_rgb, 0.7, overlay_rgb, 0.3, 0)
                    vis_depth = cv2.addWeighted(vis_depth, 0.7, overlay_depth, 0.3, 0)
                
                # 绘制aff_rot_bbox（红色高亮粗体）- 长80mm，宽19mm的抓取框
                if aff.get('aff_rot_bbox') is not None:
                    # aff_rot_bbox是list of list，每个元素格式: [cx, cy, w, h, angle]
                    for arb in aff['aff_rot_bbox']:
                        rect = ((arb[0], arb[1]), (arb[2], arb[3]), np.degrees(arb[4]))
                        aff_box = cv2.boxPoints(rect)
                        aff_box = np.int32(aff_box)
                        cv2.drawContours(vis_rgb, [aff_box], 0, (0, 0, 255), 8)  # 红色粗体
                        cv2.drawContours(vis_depth, [aff_box], 0, (0, 0, 255), 8)
            
            # 绘制抓取点和方向（跳过边界区域的affordance）
            if aff.get('grasp_center') is not None and not aff.get('near_boundary', False):
                cx, cy = aff['grasp_center']
                cx, cy = int(cx), int(cy)
                
                # 抓取中心点
                cv2.circle(vis_rgb, (cx, cy), 5, color, -1)
                cv2.circle(vis_rgb, (cx, cy), 7, (255, 255, 255), 2)
                cv2.circle(vis_depth, (cx, cy), 5, color, -1)
                
                # 抓取方向（红色，加长）
                if aff.get('grasp_angle') is not None:
                    angle = aff['grasp_angle']
                    arrow_len = 80  # 加长一倍
                    end_x = cx + arrow_len * np.cos(angle)
                    end_y = cy + arrow_len * np.sin(angle)
                    cv2.arrowedLine(vis_rgb, (cx, cy), (int(end_x), int(end_y)), 
                                  (0, 0, 255), 5, tipLength=0.3)  # 改为红色，加粗
                    cv2.arrowedLine(vis_depth, (cx, cy), (int(end_x), int(end_y)), 
                                  (0, 0, 255), 5, tipLength=0.3)  # 改为红色，加粗
                
                # 夹爪位置
                if aff.get('grasp_width_pixels') is not None:
                    width_pixels = aff['grasp_width_pixels'] / 2
                    perpendicular_angle = angle + np.pi/2
                    
                    left_x = cx - width_pixels * np.cos(perpendicular_angle)
                    left_y = cy - width_pixels * np.sin(perpendicular_angle)
                    right_x = cx + width_pixels * np.cos(perpendicular_angle)
                    right_y = cy + width_pixels * np.sin(perpendicular_angle)
                    
                    # 绘制夹爪
                    cv2.circle(vis_rgb, (int(left_x), int(left_y)), 3, (0, 0, 255), -1)
                    cv2.circle(vis_rgb, (int(right_x), int(right_y)), 3, (0, 0, 255), -1)
                    cv2.line(vis_rgb, (int(left_x), int(left_y)), 
                            (int(right_x), int(right_y)), (0, 0, 255), 1)
                
                # 显示宽度信息
                if aff.get('object_width_mm') is not None:
                    width_text = f"{aff['object_width_mm']:.1f}mm"
                    cv2.putText(vis_rgb, width_text, (cx+10, cy-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 将mask叠加到主图像上（增加mask的可见度）
        vis_rgb = cv2.addWeighted(vis_rgb, 0.7, mask_overlay_rgb, 0.3, 0)
        vis_depth = cv2.addWeighted(vis_depth, 0.7, mask_overlay_depth, 0.3, 0)
        
        # 添加统计信息（包含类别分布）
        graspable = sum(1 for a in affordances if a.get('grasp_center') is not None and not a.get('near_boundary', False))
        near_boundary = sum(1 for a in affordances if a.get('near_boundary', False))
        
        # 统计各类别数量
        category_counts = {}
        for det in detections:
            cat = det.get('label', 'unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # 生成类别统计文本
        category_text = ", ".join([f"{cat}: {count}" for cat, count in category_counts.items()])
        
        # 显示统计信息
        info_text1 = f"Detected: {len(detections)} ({category_text})"
        info_text2 = f"Graspable: {graspable}, Boundary: {near_boundary}"
        
        # 绘制带阴影的文本
        cv2.putText(vis_rgb, info_text1, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(vis_rgb, info_text1, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(vis_rgb, info_text2, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(vis_rgb, info_text2, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 组合图像
        combined = np.hstack([vis_rgb, vis_depth])
        
        # 保存
        cv2.imwrite(output_path, combined)
        if self.verbose:
            print(f"  All results visualization saved: {output_path}")
    
    def _visualize_result(self, rgb_image: np.ndarray, depth_image: np.ndarray,
                       detection: Dict,
                       frame_name: str, det_idx: int, affordance: Dict = None) -> Path:
        """
        可视化抓取结果
        """
        fig = plt.figure(figsize=(10, 5))
        
        # 1. RGB with detection box and grasp visualization
        ax1 = fig.add_subplot(121)
        rgb_vis = rgb_image.copy()
        x, y, w, h = detection['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)  # Convert to integers for drawing
        cv2.rectangle(rgb_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 如果有affordance且不在边界区域，绘制抓取信息
        if affordance and affordance.get('grasp_center') is not None and not affordance.get('near_boundary', False):
            # 绘制抓取中心（绿色圆点）
            cx, cy = affordance['grasp_center']
            cv2.circle(rgb_vis, (int(cx), int(cy)), 5, (0, 255, 0), -1)
            
            # 绘制左右夹爪位置（红色小圆）
            if affordance.get('left_finger'):
                lx, ly = affordance['left_finger']
                cv2.circle(rgb_vis, (int(lx), int(ly)), 3, (0, 0, 255), -1)
            if affordance.get('right_finger'):
                rx, ry = affordance['right_finger']
                cv2.circle(rgb_vis, (int(rx), int(ry)), 3, (0, 0, 255), -1)
            
            # 只绘制夹爪张开状态的投影（蓝色实线框）
            if affordance.get('gripper_open_bbox'):
                go = affordance['gripper_open_bbox']
                rect_center = (go[0], go[1])
                rect_size = (go[2], go[3])
                rect_angle = np.degrees(go[4])
                
                rect = ((rect_center[0], rect_center[1]), 
                       (rect_size[0], rect_size[1]), 
                       rect_angle)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(rgb_vis, [box], 0, (0, 100, 255), 6)  # 橙色实线框表示张开的夹爪，加粗
            
            # 绘制aff_rot_bbox（红色高亮粗体）
            if affordance.get('aff_rot_bbox'):
                # aff_rot_bbox是list of list，每个元素格式: [cx, cy, w, h, angle]
                for arb in affordance['aff_rot_bbox']:
                    rect = ((arb[0], arb[1]), (arb[2], arb[3]), np.degrees(arb[4]))
                    aff_box = cv2.boxPoints(rect)
                    aff_box = np.int32(aff_box)
                    cv2.drawContours(rgb_vis, [aff_box], 0, (0, 0, 255), 8)  # 红色粗体
            
            # 绘制抓取方向箭头（更明显）
            if affordance.get('grasp_angle') is not None:
                angle = affordance['grasp_angle']
                # 绘制主方向箭头
                arrow_len = 100  # 加长一倍
                end_x = cx + arrow_len * np.cos(angle)
                end_y = cy + arrow_len * np.sin(angle)
                cv2.arrowedLine(rgb_vis, (int(cx), int(cy)), 
                              (int(end_x), int(end_y)), (0, 0, 255), 6, tipLength=0.3)  # 红色粗箭头
                
                # 绘制反方向的短线表示抓取轴
                back_len = 20
                back_x = cx - back_len * np.cos(angle)
                back_y = cy - back_len * np.sin(angle)
                cv2.line(rgb_vis, (int(cx), int(cy)), 
                        (int(back_x), int(back_y)), (0, 255, 255), 3)
        
            # 添加宽度信息文字
            if affordance.get('object_width'):
                text = f"W: {affordance['object_width']:.1f}mm"
                cv2.putText(rgb_vis, text, (int(cx-30), int(cy-20)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        ax1.imshow(cv2.cvtColor(rgb_vis, cv2.COLOR_BGR2RGB))
        title = f"Detection {det_idx+1}"
        if affordance and affordance.get('quality_score'):
            title += f" (Score: {affordance['quality_score']:.2f})"
        ax1.set_title(title)
        ax1.axis('off')
        
        # 2. Depth visualization with grasp
        ax2 = fig.add_subplot(122)
        depth_vis = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
        ax2.imshow(depth_vis, cmap='turbo')
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
        ax2.add_patch(rect)
        
        # 在深度图上也显示抓取点
        if affordance and affordance.get('grasp_center') is not None:
            cx, cy = affordance['grasp_center']
            ax2.plot(cx, cy, 'go', markersize=8)
            
            # 显示左右夹爪
            if affordance.get('left_finger'):
                lx, ly = affordance['left_finger']
                ax2.plot(lx, ly, 'ro', markersize=5)
            if affordance.get('right_finger'):
                rx, ry = affordance['right_finger']
                ax2.plot(rx, ry, 'ro', markersize=5)
                
            # 显示抓取线
            if affordance.get('left_finger') and affordance.get('right_finger'):
                ax2.plot([lx, rx], [ly, ry], 'r-', linewidth=2)
        
        ax2.set_title("Depth with Grasp")
        ax2.axis('off')
        
        plt.tight_layout()
        
        # 保存图像（如果启用）
        vis_path = None
        if self.save_individual:
            vis_filename = f"{frame_name}_det{det_idx:02d}_grasp.png"
            vis_path = self.output_dir / vis_filename
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return vis_path
    
    def _compute_affordance(self, detection: Dict, target_depth: np.ndarray) -> Dict:
        """
        计算二指夹爪抓取affordance（自适应宽度）
        
        Args:
            detection: 检测结果
            target_depth: 目标深度（米）
            
        Returns:
            Affordance字典
        """
        # 夹爪参数（毫米）
        GRIPPER_MAX_OPEN_MM = 70.0  # 最大张开宽度
        GRIPPER_MIN_OPEN_MM = 10.0  # 最小有效抓取宽度
        GRIPPER_BODY_WIDTH_MM = 30.0  # 夹爪本体宽度（用于可视化）
        GRASP_DEPTH_MM = 30.0  # 抓取深度
        
        affordance = {
            'grasp_center': None,
            'grasp_angle': None,
            'grasp_width_actual': None,  # 实际抓取宽度
            'grasp_depth': GRASP_DEPTH_MM,
            'left_finger': None,
            'right_finger': None,
            'gripper_body_bbox': None,  # 夹爪本体投影
            'gripper_open_bbox': None,  # 夹爪张开投影
            'quality_score': 0.0,
            'object_width': None,
            'width_feasible': False,
            'depth_at_grasp': None,
            'method': 'adaptive_width_grasp',
            'aff_rot_bbox': []  # 初始化为空列表
        }
        
        # 检查输入有效性
        if target_depth is None or target_depth.size == 0:
            return affordance
            
        valid_mask = target_depth > 0
        if not np.any(valid_mask):
            return affordance
            
        try:
            # 1. 逐步搜索可抓取的顶部区域（从35mm到10mm）
            valid_depths = target_depth[valid_mask]
            min_depth = np.min(valid_depths)
            
            # 从35mm开始，逐步减小到10mm，寻找可抓取的区域
            depth_thresholds = [0.035, 0.030, 0.025, 0.020, 0.015, 0.010]  # 35mm到10mm
            best_grasp = None
            
            for threshold in depth_thresholds:
                depth_threshold = min_depth + threshold
                top_mask = (target_depth > 0) & (target_depth <= depth_threshold)
                
                if not np.any(top_mask):
                    continue  # 跳过空的顶部区域
                
                # 尝试在这个深度范围内寻找抓取
                grasp_result = self._find_best_grasp_line_adaptive(
                    target_depth, top_mask, detection['bbox'],
                    GRIPPER_MIN_OPEN_MM, GRIPPER_MAX_OPEN_MM
                )
                
                if grasp_result is not None:
                    # 找到可抓取的区域
                    if self.verbose:
                        print(f"  Found graspable region at depth +{threshold*1000:.0f}mm")
                    best_grasp = grasp_result
                    break  # 使用找到的第一个（最深的）可抓取区域
            
            if best_grasp is None:
                # 如果所有深度都无法抓取，尝试使用全部有效区域
                if self.verbose:
                    print(f"  No graspable region found in 10-35mm range, trying full depth")
                best_grasp = self._find_best_grasp_line_adaptive(
                    target_depth, valid_mask, detection['bbox'],
                    GRIPPER_MIN_OPEN_MM, GRIPPER_MAX_OPEN_MM
                )
            
            if best_grasp is None:
                # 不返回任何旋转边界框和抓取信息
                return affordance
            
            # 3. 提取抓取参数
            grasp_center = best_grasp['center']
            grasp_angle = best_grasp['angle']
            grasp_score = best_grasp['score']
            depth_at_grasp = best_grasp['depth']
            object_width_mm = best_grasp['object_width']
            
            # 4. 确定实际抓取宽度（留5mm余量）
            actual_grasp_width_mm = min(object_width_mm + 5, GRIPPER_MAX_OPEN_MM)
            affordance['grasp_width_actual'] = float(actual_grasp_width_mm)
            affordance['object_width'] = float(object_width_mm)
            
            # 检查宽度是否可行
            if object_width_mm < GRIPPER_MIN_OPEN_MM or object_width_mm > GRIPPER_MAX_OPEN_MM:
                affordance['width_feasible'] = False
                # 如果宽度不可行，清除所有抓取相关信息
                return affordance
            
            affordance['width_feasible'] = True
            
            # 5. 计算左右夹爪位置（像素坐标）
            pixels_per_mm = self._estimate_pixels_per_mm(depth_at_grasp)
            actual_width_pixels = actual_grasp_width_mm * pixels_per_mm
            
            # 夹爪张开方向与抓取方向相同（都垂直于顶线）
            # grasp_angle已经垂直于顶线（长边）
            # 夹爪沿此方向张开，会在顶线的两侧
            open_direction = np.array([np.cos(grasp_angle), 
                                      np.sin(grasp_angle)])
            
            # 左右夹爪位置（在顶线两侧，连线垂直于顶线）
            offset = actual_width_pixels / 2
            left_finger = grasp_center - offset * open_direction
            right_finger = grasp_center + offset * open_direction
            
            # 6. 生成夹爪本体bbox和张开bbox
            grasp_depth_pixels = GRASP_DEPTH_MM * pixels_per_mm
            body_width_pixels = GRIPPER_BODY_WIDTH_MM * pixels_per_mm
            
            # 夹爪本体投影
            gripper_body_bbox = [
                float(grasp_center[0]),
                float(grasp_center[1]),
                float(body_width_pixels),
                float(grasp_depth_pixels),
                float(grasp_angle)
            ]
            
            # 夹爪张开投影
            gripper_open_bbox = [
                float(grasp_center[0]),
                float(grasp_center[1]),
                float(actual_width_pixels),
                float(grasp_depth_pixels),
                float(grasp_angle)
            ]
            
            # 7. 填充返回结果
            affordance['grasp_center'] = grasp_center.tolist()
            affordance['grasp_angle'] = float(grasp_angle)
            affordance['left_finger'] = left_finger.tolist()
            affordance['right_finger'] = right_finger.tolist()
            affordance['gripper_body_bbox'] = gripper_body_bbox
            affordance['gripper_open_bbox'] = gripper_open_bbox
            affordance['quality_score'] = float(grasp_score)
            affordance['depth_at_grasp'] = float(depth_at_grasp)
            
            # 添加旋转边界框信息（如果有）
            if best_grasp and 'rot_bbox' in best_grasp:
                affordance['rot_bbox'] = best_grasp['rot_bbox']
            
            # 添加aff_rot_bbox - 以center为中心，angle为长度方向，长80mm，宽19mm
            aff_length_mm = 80.0
            aff_width_mm = 19.0
            aff_length_pixels = aff_length_mm * pixels_per_mm
            aff_width_pixels = aff_width_mm * pixels_per_mm
            
            # 创建候选旋转矩形列表
            aff_rot_bbox_list = []
            
            # 检查是否为plastic bag类别（包含长宽比条件）
            label_is_plastic = detection.get('label', '').lower() == 'plastic bag'
            
            # 计算bbox的长宽比
            bbox = detection.get('bbox', [0, 0, 1, 1])  # [x, y, width, height]
            bbox_width = bbox[2]
            bbox_height = bbox[3]
            # 长宽比（取较大值/较小值）
            aspect_ratio = max(bbox_width, bbox_height) / min(bbox_width, bbox_height) if min(bbox_width, bbox_height) > 0 else 1.0
            
            # 满足label且长宽比在0.5-2.0之间才认为是plastic bag
            is_plastic_bag = label_is_plastic and (0.5 < aspect_ratio < 2.0)
            
            if is_plastic_bag:
                # 对于plastic bag，生成4个候选，每45度一个
                base_angle = grasp_angle
                for i in range(4):
                    candidate_angle = base_angle + i * np.pi / 4  # 每45度一个候选
                    
                    # 将角度规范化到 [0, π] 范围
                    angle_normalized = candidate_angle % (2 * np.pi)
                    if angle_normalized > np.pi:
                        angle_normalized = angle_normalized - np.pi
                    
                    # 使用 cx, cy, w, h, angle 格式表示
                    candidate_bbox = [
                        float(grasp_center[0]),  # cx
                        float(grasp_center[1]),  # cy
                        float(aff_length_pixels),  # w (长度)
                        float(aff_width_pixels),   # h (宽度)
                        float(angle_normalized)  # angle (0-π)
                    ]
                    aff_rot_bbox_list.append(candidate_bbox)
            else:
                # 对于其他类别，只生成一个候选
                aff_angle = grasp_angle
                
                # 将角度规范化到 [0, π] 范围
                aff_angle_normalized = aff_angle % (2 * np.pi)
                if aff_angle_normalized > np.pi:
                    aff_angle_normalized = aff_angle_normalized - np.pi
                
                # 使用 cx, cy, w, h, angle 格式表示
                single_bbox = [
                    float(grasp_center[0]),  # cx
                    float(grasp_center[1]),  # cy
                    float(aff_length_pixels),  # w (长度)
                    float(aff_width_pixels),   # h (宽度)
                    float(aff_angle_normalized)  # angle (0-π)
                ]
                aff_rot_bbox_list.append(single_bbox)
            
            affordance['aff_rot_bbox'] = aff_rot_bbox_list
            
        except Exception as e:
            print(f"Error computing affordance: {e}")
            
        return affordance
    
    def _find_best_grasp_line_adaptive(self, depth_image: np.ndarray, top_mask: np.ndarray,
                                      bbox: List[float], min_width_mm: float, 
                                      max_width_mm: float) -> Optional[Dict]:
        """
        寻找最佳抓取线（自适应宽度版本）
        使用旋转边界框来准确测量顶部区域宽度
        
        Args:
            depth_image: 深度图
            top_mask: 顶部区域mask
            bbox: 目标边界框
            min_width_mm: 最小可抓宽度(mm)
            max_width_mm: 最大可抓宽度(mm)
            
        Returns:
            最佳抓取参数字典
        """
        # 清理顶部区域：去除离散点，只保留最大连通区域
        top_mask_uint8 = top_mask.astype(np.uint8) * 255
        
        # 形态学开运算去除小噪声
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(top_mask_uint8, cv2.MORPH_OPEN, kernel)
        
        # 找到连通组件
        num_labels, labels = cv2.connectedComponents(cleaned_mask)
        
        if num_labels <= 1:  # 没有找到有效区域
            return None
        
        # 找到最大的连通组件（除了背景）
        max_label = 0
        max_size = 0
        for label_id in range(1, num_labels):
            size = np.sum(labels == label_id)
            if size > max_size:
                max_size = size
                max_label = label_id
        
        # 只使用最大连通组件
        cleaned_mask = (labels == max_label)
        
        # 获取清理后的顶部区域的点
        y_coords, x_coords = np.where(cleaned_mask)
        if len(x_coords) < 10:
            return None
        
        # 使用旋转边界框来拟合顶部区域
        points = np.column_stack([x_coords, y_coords])
        
        # 获取最小面积矩形
        rect = cv2.minAreaRect(points)
        (rect_cx, rect_cy), (rect_w, rect_h), rect_angle = rect
        
        # 获取旋转矩形的顶点
        box = cv2.boxPoints(rect)
        
        # 创建旋转矩形的mask
        rect_mask = np.zeros_like(cleaned_mask, dtype=np.uint8)
        cv2.fillPoly(rect_mask, [np.int32(box)], 1)
        
        # 只保留旋转矩形内部且属于顶部区域的点
        refined_mask = cleaned_mask & (rect_mask > 0)
        
        # 在旋转矩形内部重新计算中心点
        refined_y, refined_x = np.where(refined_mask)
        if len(refined_x) < 10:
            # 如果过滤后点太少，使用原始清理后的mask
            refined_mask = cleaned_mask
            center = np.array([rect_cx, rect_cy])
        else:
            # 使用旋转矩形内部点的质心作为抓取中心
            center = np.array([np.mean(refined_x), np.mean(refined_y)])
        
        # 确保宽度是较短边
        if rect_h > rect_w:
            rect_w, rect_h = rect_h, rect_w
            rect_angle = rect_angle + 90
        
        # 将角度标准化到 [0, 180)
        rect_angle = rect_angle % 180
        
        # 获取中心深度用于像素-毫米转换（使用精炼后的中心点）
        cx, cy = int(center[0]), int(center[1])
        if 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
            center_depth = depth_image[cy, cx]
        else:
            center_depth = np.mean(depth_image[top_mask])
        
        pixels_per_mm = self._estimate_pixels_per_mm(center_depth)
        
        # 在边界框内偏移到全局坐标
        x_offset, y_offset = int(bbox[0]), int(bbox[1])
        global_center = center + np.array([x_offset, y_offset])
        
        # 计算旋转矩形的实际宽度（较短边）
        rect_width_mm = rect_h / pixels_per_mm  # 使用较短边作为宽度
        
        # 检查宽度是否在可抓范围内
        if rect_width_mm < min_width_mm:
            return None
        
        if rect_width_mm > max_width_mm:
            # 如果整体太宽，尝试寻找局部可抓区域
            return self._find_local_graspable_region(
                depth_image, top_mask, bbox, center_depth, 
                min_width_mm, max_width_mm
            )
        
        # 抓取方向应该垂直于顶线（长边）
        # rect_angle是长边的方向，抓取方向垂直于此，所以加90度
        # 转换角度到弧度（OpenCV的角度是逆时针的）
        grasp_angle = np.radians(rect_angle + 90)
        
        # 评估这个抓取方向的质量（使用精炼后的mask）
        direction = np.array([np.cos(grasp_angle), np.sin(grasp_angle)])
        score = self._evaluate_grasp_angle_adaptive(
            depth_image, refined_mask, center, direction, 
            rect_width_mm, min_width_mm, max_width_mm
        )
        
        # 如果主方向不好，尝试垂直方向
        if score < 0.5:
            perpendicular_angle = grasp_angle + np.pi/2
            perpendicular_dir = np.array([np.cos(perpendicular_angle), np.sin(perpendicular_angle)])
            perpendicular_score = self._evaluate_grasp_angle_adaptive(
                depth_image, refined_mask, center, perpendicular_dir,
                rect_w / pixels_per_mm, min_width_mm, max_width_mm
            )
            
            if perpendicular_score > score:
                grasp_angle = perpendicular_angle
                rect_width_mm = rect_w / pixels_per_mm
                score = perpendicular_score
        
        # 将旋转边界框转换到全局坐标
        global_box = box + np.array([x_offset, y_offset])
        
        best_grasp = {
            'center': global_center,
            'angle': grasp_angle,
            'score': score,
            'depth': center_depth,
            'object_width': rect_width_mm,
            'rot_bbox': global_box.tolist()  # 添加旋转边界框信息
        }
        
        return best_grasp
    
    def _find_local_graspable_region(self, depth_image: np.ndarray, top_mask: np.ndarray,
                                    bbox: List[float], center_depth: float,
                                    min_width_mm: float, max_width_mm: float) -> Optional[Dict]:
        """
        当整体太宽时，寻找局部可抓区域或在中心进行部分抓取
        """
        pixels_per_mm = self._estimate_pixels_per_mm(center_depth)
        x_offset, y_offset = int(bbox[0]), int(bbox[1])
        
        # 先进行形态学清理
        top_mask_uint8 = top_mask.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(top_mask_uint8, cv2.MORPH_OPEN, kernel)
        
        # 使用连通组件分析找到独立的区域
        num_labels, labels = cv2.connectedComponents(cleaned_mask)
        
        best_grasp = None
        best_score = -np.inf
        
        # 分析每个连通组件
        for label_id in range(1, num_labels):
            component_mask = (labels == label_id)
            component_points = np.where(component_mask)
            
            if len(component_points[0]) < 20:  # 忽略太小的组件
                continue
            
            points = np.column_stack([component_points[1], component_points[0]])
            
            # 获取这个组件的旋转边界框
            rect = cv2.minAreaRect(points)
            (rect_cx, rect_cy), (rect_w, rect_h), rect_angle = rect
            
            # 确保宽度是较短边
            if rect_h > rect_w:
                rect_w, rect_h = rect_h, rect_w
                rect_angle = rect_angle + 90
            
            width_mm = rect_h / pixels_per_mm
            
            # 检查宽度是否合适
            if min_width_mm <= width_mm <= max_width_mm:
                center = np.array([rect_cx, rect_cy])
                global_center = center + np.array([x_offset, y_offset])
                grasp_angle = np.radians(rect_angle)
                
                # 评估质量
                direction = np.array([np.cos(grasp_angle), np.sin(grasp_angle)])
                score = self._evaluate_grasp_angle_adaptive(
                    depth_image, component_mask, center, direction,
                    width_mm, min_width_mm, max_width_mm
                )
                
                if score > best_score:
                    best_score = score
                    # 获取旋转边界框的顶点并转换到全局坐标
                    rot_box = cv2.boxPoints(rect)
                    global_rot_box = rot_box + np.array([x_offset, y_offset])
                    best_grasp = {
                        'center': global_center,
                        'angle': grasp_angle,
                        'score': score,
                        'depth': center_depth,
                        'object_width': width_mm,
                        'rot_bbox': global_rot_box.tolist()
                    }
        
        return best_grasp
    
    def _measure_object_width(self, mask: np.ndarray, center: np.ndarray,
                              direction: np.ndarray, max_search_dist: float) -> float:
        """
        测量特定方向上的物体宽度
        
        Args:
            mask: 物体mask
            center: 中心点
            direction: 方向单位向量
            max_search_dist: 最大搜索距离(像素)
            
        Returns:
            物体宽度(像素)
        """
        perpendicular = np.array([-direction[1], direction[0]])  # 垂直方向
        
        # 从中心向两边搜索边界
        left_dist = 0
        right_dist = 0
        
        # 向左搜索
        for dist in range(1, int(max_search_dist)):
            point = center - dist * perpendicular
            x, y = int(point[0]), int(point[1])
            
            if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
                break
            if not mask[y, x]:
                break
            left_dist = dist
        
        # 向右搜索
        for dist in range(1, int(max_search_dist)):
            point = center + dist * perpendicular
            x, y = int(point[0]), int(point[1])
            
            if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
                break
            if not mask[y, x]:
                break
            right_dist = dist
        
        return left_dist + right_dist
    
    def _evaluate_grasp_angle_adaptive(self, depth_image: np.ndarray, mask: np.ndarray,
                                      center: np.ndarray, direction: np.ndarray,
                                      object_width_mm: float, min_width_mm: float,
                                      max_width_mm: float) -> float:
        """
        评估特定角度的抓取质量（自适应版本）
        
        Args:
            depth_image: 深度图
            mask: 有效区域mask
            center: 抓取中心
            direction: 抓取方向单位向量
            object_width_mm: 物体宽度(mm)
            min_width_mm: 最小可抓宽度(mm)
            max_width_mm: 最大可抓宽度(mm)
            
        Returns:
            质量分数
        """
        # 1. 宽度适合度（50mm最理想）
        ideal_width = 50.0
        width_fitness = 1.0 - abs(object_width_mm - ideal_width) / 20.0
        width_fitness = max(0, min(1, width_fitness))
        
        # 2. 深度平坦度（沿抓取线采样）
        line_length = 30  # 采样长度（像素）
        num_samples = 20
        samples = []
        
        for t in np.linspace(-line_length/2, line_length/2, num_samples):
            point = center + t * direction
            x, y = int(point[0]), int(point[1])
            
            if (0 <= y < depth_image.shape[0] and 
                0 <= x < depth_image.shape[1] and mask[y, x]):
                samples.append(depth_image[y, x])
        
        if len(samples) < 5:
            return -np.inf
            
        samples = np.array(samples)
        depth_consistency = 1.0 / (1.0 + np.std(samples))
        
        # 3. 覆盖完整度
        coverage = len(samples) / num_samples
        
        # 4. 综合评分
        score = (0.3 * width_fitness + 
                0.3 * depth_consistency + 
                0.2 * coverage +
                0.2 * (1.0 if object_width_mm < max_width_mm * 0.9 else 0.5))
        
        return score
    
    def _find_best_grasp_line(self, depth_image: np.ndarray, top_mask: np.ndarray, 
                              bbox: List[float]) -> Optional[Dict]:
        """
        在顶部区域寻找最佳抓取线
        
        Args:
            depth_image: 深度图
            top_mask: 顶部区域mask
            bbox: 目标边界框
            
        Returns:
            最佳抓取参数字典
        """
        # 获取顶部区域的点
        y_coords, x_coords = np.where(top_mask)
        if len(x_coords) < 10:
            return None
            
        # 计算质心作为初始抓取中心
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        center = np.array([center_x, center_y])
        
        # 在边界框内偏移到全局坐标
        x_offset, y_offset = int(bbox[0]), int(bbox[1])
        global_center = center + np.array([x_offset, y_offset])
        
        # 尝试不同角度的抓取线
        angles = np.linspace(0, np.pi, 8, endpoint=False)
        best_score = -np.inf
        best_grasp = None
        
        for angle in angles:
            # 沿着当前角度采样点
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # 计算这个角度的抓取质量
            score = self._evaluate_grasp_angle(depth_image, top_mask, center, direction)
            
            if score > best_score:
                best_score = score
                # 获取抓取点的深度
                cx, cy = int(center[1]), int(center[0])
                if 0 <= cx < depth_image.shape[0] and 0 <= cy < depth_image.shape[1]:
                    depth_at_center = depth_image[cx, cy]
                else:
                    depth_at_center = np.mean(depth_image[top_mask])
                    
                best_grasp = {
                    'center': global_center,
                    'angle': angle,
                    'score': score,
                    'depth': depth_at_center
                }
        
        return best_grasp
    
    def _evaluate_grasp_angle(self, depth_image: np.ndarray, mask: np.ndarray,
                             center: np.ndarray, direction: np.ndarray) -> float:
        """
        评估特定角度的抓取质量
        
        Args:
            depth_image: 深度图
            mask: 有效区域mask
            center: 抓取中心
            direction: 抓取方向单位向量
            
        Returns:
            质量分数
        """
        # 沿着抓取线采样点
        line_length = 30  # 采样长度（像素）
        num_samples = 20
        
        samples = []
        for t in np.linspace(-line_length/2, line_length/2, num_samples):
            point = center + t * direction
            x, y = int(point[0]), int(point[1])
            
            # 检查点是否在有效范围内
            if (0 <= y < depth_image.shape[0] and 
                0 <= x < depth_image.shape[1] and
                mask[y, x]):
                samples.append(depth_image[y, x])
        
        if len(samples) < 5:
            return -np.inf
            
        samples = np.array(samples)
        
        # 计算质量指标
        # 1. 深度一致性（标准差越小越好）
        depth_consistency = 1.0 / (1.0 + np.std(samples))
        
        # 2. 覆盖率（采样点数量）
        coverage = len(samples) / num_samples
        
        # 3. 深度平滑性（相邻点深度差）
        if len(samples) > 1:
            depth_diffs = np.abs(np.diff(samples))
            smoothness = 1.0 / (1.0 + np.mean(depth_diffs))
        else:
            smoothness = 0.0
        
        # 综合分数
        score = 0.4 * depth_consistency + 0.3 * coverage + 0.3 * smoothness
        
        return score
    
    def _draw_dashed_line(self, img, pt1, pt2, color, thickness, dash_length):
        """
        绘制虚线
        
        Args:
            img: 图像
            pt1: 起点
            pt2: 终点
            color: 颜色
            thickness: 线宽
            dash_length: 虚线段长度
        """
        dist = np.linalg.norm(np.array(pt2) - np.array(pt1))
        num_dashes = int(dist / (dash_length * 2))
        
        for i in range(num_dashes):
            t1 = i * 2 * dash_length / dist
            t2 = min((i * 2 + 1) * dash_length / dist, 1.0)
            
            start = (int(pt1[0] + t1 * (pt2[0] - pt1[0])),
                    int(pt1[1] + t1 * (pt2[1] - pt1[1])))
            end = (int(pt1[0] + t2 * (pt2[0] - pt1[0])),
                  int(pt1[1] + t2 * (pt2[1] - pt1[1])))
            
            cv2.line(img, start, end, color, thickness)
    
    def _estimate_pixels_per_mm(self, depth_m: float) -> float:
        """
        估算给定深度下的像素/毫米比例
        
        Args:
            depth_m: 深度（米）
            
        Returns:
            像素/毫米比例
        """
        # 假设相机参数（D435）
        # FOV约为60度，分辨率1280x720
        # 在深度d处，视场宽度约为 2*d*tan(30°) = 1.15*d 米
        
        image_width = 1280  # 像素
        fov_deg = 60  # 度
        
        # 计算该深度下的视场宽度（米）
        fov_width_m = 2 * depth_m * np.tan(np.radians(fov_deg/2))
        
        # 转换为毫米
        fov_width_mm = fov_width_m * 1000
        
        # 像素/毫米
        pixels_per_mm = image_width / fov_width_mm
        
        return pixels_per_mm


def main():
    """
    测试主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate BEV Grasp Affordance for Bags')
    parser.add_argument('--rgb', type=str, 
                       default='data/bag/0912_topdownbags/images/rgb/rgb_frame_000000.jpg',
                       help='RGB image file or directory')
    parser.add_argument('--depth', type=str, 
                       default='data/bag/0912_topdownbags/images/depth/depth_frame_000000.npy',
                       help='Depth data file or directory')
    parser.add_argument('--text', type=str, 
                       default="paper bag. plastic bag. bag",
                       help='Detection text prompt')
    parser.add_argument('--output', type=str, default='vis_aff',
                       help='Output directory for visualizations')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching')
    parser.add_argument('--cache-dir', type=str, default='.cache',
                       help='Cache directory')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear cache before processing')
    parser.add_argument('--save-individual', action='store_true',
                       help='Save individual detection visualizations')
    
    args = parser.parse_args()
    
    # 清理缓存（如果需要）
    if args.clear_cache:
        print("Clearing cache...")
        if CDM_CACHE_AVAILABLE:
            cache = CDMCache(cache_dir=f"{args.cache_dir}/cdm")
            cache.clear()
        if CachedDetectionAPI:
            from cache_utils import DetectionCache
            det_cache = DetectionCache(f"{args.cache_dir}/detection")
            det_cache.clear()
        print("Cache cleared.")
    
    # 创建Affordance生成器
    affordance_gen = Affordance4BEVGraspBag(
        detection_text=args.text,
        output_dir=args.output,
        verbose=args.verbose,
        enable_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        save_individual=args.save_individual
    )
    
    # 处理输入
    results = affordance_gen.process(args.rgb, args.depth)
    
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
            print(f"\nSummary:")
            print(f"  Frames processed: {len(results)}")
            print(f"  Total detections: {total_dets}")
            print(f"  Output saved to: {args.output}/")


if __name__ == "__main__":
    main()
