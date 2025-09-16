import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from pycocotools import mask as coco_mask
from dino_any_percept_api import DetectionAPI
import math
from common import VisualizationUtils, GeometryUtils, MaskUtils, COCOUtils, HoleProcessingUtils, VideoUtils

class VideoProcessor:
    def __init__(self, video_path, output_dir="data", frame_rate=0.5, crop_top_half=True, detect_objects=True, max_frames=None, start_frame=0):
        """
        视频处理类 - 支持目标检测和COCO格式保存
        
        Args:
            video_path (str): 输入视频路径
            output_dir (str): 输出根目录
            frame_rate (float): 提取帧率（帧/秒），默认0.5（每2秒1帧）
            crop_top_half (bool): 是否只提取图像上半部分
            detect_objects (bool): 是否进行目标检测
            max_frames (int): 最大处理帧数，None表示不限制
            start_frame (int): 开始处理的帧数，默认从第0帧开始
        """
        self.video_path = video_path
        self.video_name = Path(video_path).stem
        self.output_dir = output_dir
        self.frame_rate = frame_rate
        self.crop_top_half = crop_top_half
        self.detect_objects = detect_objects
        self.max_frames = max_frames
        self.start_frame = start_frame
        
        # 创建目录结构
        self.img_dir = os.path.join(output_dir, 'images')
        self.ann_dir = os.path.join(output_dir, 'anno')
        self.vis_dir = os.path.join(output_dir, 'vis')
        
        Path(self.img_dir).mkdir(parents=True, exist_ok=True)
        Path(self.ann_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vis_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化DetectionAPI
        if self.detect_objects:
            self.detection_api = self._init_detection_api()
        
        # 初始化视频捕获
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
            
        # 获取视频属性
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 计算帧间隔（每2秒1帧）
        self.frame_interval = max(1, int(self.original_fps / frame_rate))
        self.target_fps = frame_rate
        
        # COCO数据格式
        self.coco_data = {
            'info': {
                'description': f'handle.bag detection from {self.video_name}',
                'url': '',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'AutoLabel',
                'date_created': datetime.now().isoformat()
            },
            'licenses': [{
                'url': '',
                'id': 1,
                'name': 'Unknown'
            }],
            'images': [],
            'annotations': [],
            'categories': [
                {'id': 1, 'name': 'bag', 'supercategory': 'object'},
                {'id': 2, 'name': 'handle', 'supercategory': 'object'}
            ]
        }
        
        self.annotation_id = 1
        
        print(f"视频信息:")
        print(f"  原始FPS: {self.original_fps:.2f}")
        print(f"  总帧数: {self.total_frames}")
        print(f"  分辨率: {self.video_width}x{self.video_height}")
        print(f"  目标FPS: {self.target_fps:.2f} (每{1/self.target_fps:.1f}秒1帧)")
        print(f"  帧间隔: {self.frame_interval}")
        if self.start_frame > 0:
            print(f"  开始帧: {self.start_frame}")
        if self.max_frames is not None:
            print(f"  最大处理帧数: {self.max_frames}")
        print(f"  输出目录结构:")
        print(f"    - 图像: {self.img_dir}")
        print(f"    - 标注: {self.ann_dir}")
        print(f"    - 可视化: {self.vis_dir}")

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

    def crop_image(self, frame):
        """
        裁剪图像，提取上半部分
        
        Args:
            frame (np.ndarray): 输入图像
            
        Returns:
            np.ndarray: 裁剪后的图像
        """
        if self.crop_top_half:
            height = frame.shape[0]
            # 提取上半部分
            cropped_frame = frame[:height//2, :]
            return cropped_frame
        return frame

    def process_video(self):
        """
        处理视频，提取帧并进行目标检测，保存COCO格式结果
        
        Returns:
            dict: 处理统计信息
        """
        frame_count = 0
        saved_count = 0
        processed_count = 0
        
        # 跳转到开始帧
        if self.start_frame > 0:
            print(f"\n跳转到第 {self.start_frame} 帧...")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            frame_count = self.start_frame
        
        print(f"\n开始处理视频并检测 bag...")
        start_time = time.time()
        
        while True:
            # 检查是否达到最大帧数限制
            if self.max_frames is not None and (frame_count - self.start_frame) >= self.max_frames:
                print(f"达到最大处理帧数限制: {self.max_frames}")
                break
                
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # 按帧率间隔处理（每2秒1帧）
            # 对于start_frame后的帧，需要调整计算逻辑
            if (frame_count - self.start_frame) % self.frame_interval == 0:
                print(f"\n开始处理第 {frame_count} 帧...")
                # 裁剪图像（如果需要）
                processed_frame = self.crop_image(frame)
                
                # 生成文件名
                filename = f"{self.video_name}_frame_{frame_count:06d}.jpg"
                img_path = os.path.join(self.img_dir, filename)
                
                # 保存原始图像
                success = cv2.imwrite(img_path, processed_frame)
                if success:
                    saved_count += 1
                    
                    # 进行目标检测
                    if self.detect_objects:
                        detection_result = self._detect_and_process_frame(processed_frame, img_path, frame_count)
                        processed_count += 1
                        
                        if processed_count % 5 == 0:
                            print(f"已处理 {processed_count} 帧，检测到 {len(self.coco_data['annotations'])} 个对象...")
                    else:
                        # 仅保存图像信息到COCO
                        self._add_image_to_coco(img_path, processed_frame.shape, frame_count)
                else:
                    print(f"保存失败: {img_path}")
            
            frame_count += 1
            
        self.cap.release()
        
        # 保存COCO标注文件
        if self.detect_objects:
            COCOUtils.save_coco_annotations(self.coco_data, self.ann_dir, self.video_name)
            # 生成可视化视频
            VideoUtils.create_visualization_video(self.vis_dir, self.video_name, self.target_fps)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n处理完成!")
        print(f"总共处理帧数: {frame_count}")
        print(f"成功保存图像: {saved_count}")
        print(f"检测处理帧数: {processed_count}")
        print(f"总标注数量: {len(self.coco_data['annotations'])}")
        print(f"处理时间: {processing_time:.2f} 秒")
        print(f"平均每帧: {processing_time/processed_count:.2f} 秒" if processed_count > 0 else "")
        
        return {
            'total_frames': frame_count,
            'saved_images': saved_count,
            'processed_frames': processed_count,
            'total_annotations': len(self.coco_data['annotations']),
            'processing_time': processing_time
        }

    def _detect_and_process_frame(self, frame, file_path, frame_count):
        """检测并处理单帧图像"""
        try:
            # 使用DetectionAPI检测bag
            result = self.detection_api.detect_objects(
                rgb=frame,
                prompt_text="bag.handle",
                bbox_threshold=0.25,
                iou_threshold=0.8
            )
            
            # 添加图像信息到COCO
            image_id = frame_count + 1
            self._add_image_to_coco(
                file_path,
                frame.shape,
                image_id
            )
            
            # 处理检测结果
            if result and 'objects' in result:
                self._process_detection_results(result, image_id, frame)
            
            # 生成可视化结果
            VisualizationUtils.create_visualization(frame, result, self.vis_dir, os.path.basename(file_path), self.coco_data, image_id)
            
            return result
            
        except Exception as e:
            print(f"检测帧 {file_path} 时发生错误: {e}")
            return None

    def _add_image_to_coco(self, file_path, shape, image_id):
        """添加图像信息到COCO数据"""
        height, width = shape[:2]
        
        # 先转换成绝对路径，再去掉前缀 */auto_label/data/
        abs_file_path = os.path.abspath(file_path)
        if '/auto_label/data/' in abs_file_path:
            file_path = abs_file_path.split('/auto_label/data/')[-1]
        
        self.coco_data['images'].append({
            'id': int(image_id),
            'width': int(width),
            'height': int(height),
            'file_name': file_path,
            'license': 1,
            'date_captured': datetime.now().isoformat()
        })

    def _process_detection_results(self, result, image_id, frame):
        """处理检测结果并添加到COCO标注"""
        objects = result.get('objects', [])
        
        # 根据视频文件名确定grasp_policy
        grasp_policy = self._get_grasp_policy_from_filename()
        
        for obj in objects:
            bbox = obj.get('bbox', [])
            mask_rle = obj.get('mask', None)
            category = obj.get('category', 'unknown').lower()
            score = obj.get('score', 0.0)  # 检测API返回的是score字段
            
            # 确定类别ID
            if 'bag' in category:
                category_id = 1
            elif 'handle' in category:
                category_id = 2
                continue   # 对handle类别不进行标注
            else:
                continue # 默认为bag
                
            
            if len(bbox) >= 4:
                # 转换bbox格式 [x1, y1, x2, y2] -> [x, y, width, height]
                x1, y1, x2, y2 = bbox
                bbox_coco = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                area = float((x2 - x1) * (y2 - y1))
                
                # 如果是handle类别，检查bbox面积，如果在5000-100000范围内则跳过
                if 'handle' in category:
                    if area < 5000.0  or  area > 100000.0:
                        print(f"跳过handle: bbox面积={area:.1f} 不在范围(5000, 100000)内，忽略此检测")
                        continue
                    else:
                        print(f"handle: bbox面积={area:.1f} 在范围(5000, 100000)内，继续分析")
                
                # 计算affordance（根据grasp_policy使用不同策略）
                handle_hole = None
                rot_bbox_bag = None
                if grasp_policy == 'bf':                    # BF策略：传递frame用于handle检测
                    bf_result = self._calculate_bf_affordance(mask_rle, bbox, frame, image_id, category)
                    if isinstance(bf_result, tuple) and len(bf_result) == 3:
                        affordance, handle_hole, rot_bbox_bag = bf_result
                    else:
                        affordance = bf_result if bf_result else []
                        handle_hole = None
                        rot_bbox_bag = None
                else:
                    # TD策略：使用原有方法
                    affordance = self._calculate_td_affordance(mask_rle, bbox, frame)
                
                # 验证affordance格式：如果affordance空或不是5个元素，那么过滤掉，不加入coco_data
                if not affordance or (isinstance(affordance, list) and len(affordance) != 5):
                    print(f"过滤无效affordance: {affordance} (长度={len(affordance) if isinstance(affordance, list) else 'N/A'})")
                    continue  # 跳过此检测，不添加到coco_data
                
                valid_affordance = True
                
                annotation = {
                    'id': int(self.annotation_id),
                    'image_id': int(image_id),
                    'category_id': int(category_id),
                    'cat': int(category_id),  # 添加cat字段，内容与category_id一致，用于兼容
                    'bbox': bbox_coco,
                    'area': area,
                    'iscrowd': 0,
                    'score': float(score),  # 检测置信度
                    'affordance': [affordance] if valid_affordance else [],  # 改成list of list，单个affordance包装成列表
                    'grasp_policy': grasp_policy if valid_affordance else 'no',  # 如果没有有效affordance，grasp_policy为'no'
                    'handle_hole': handle_hole if handle_hole else None,  # 添加handle_hole字段
                    'rot_bbox_bag': rot_bbox_bag if rot_bbox_bag else None  # 添加rot_bbox_bag字段
                }
                
                # 添加分割掩码（如果存在）
                if mask_rle:
                    try:
                        # 确保掩码格式正确
                        if isinstance(mask_rle, dict) and 'counts' in mask_rle:
                            annotation['segmentation'] = mask_rle
                        annotation['area'] = float(coco_mask.area(mask_rle))
                    except Exception as e:
                        print(f"处理掩码时发生错误: {e}")
                
                self.coco_data['annotations'].append(annotation)
                self.annotation_id += 1


    def _get_grasp_policy_from_filename(self):
        """根据视频文件名确定grasp_policy"""
        video_name_lower = self.video_name.lower()
        
        if 'topdown' in video_name_lower:
            return 'td'
        elif 'backfront' in video_name_lower:
            return 'bf'
        else:
            return 'no'

    def _calculate_td_affordance(self, mask_rle, bbox, frame, enable_region_check=False):
        """
        Topdown策略：使用原有的形状拟合方法
        比较rot_bbox和circle，选择IoU最大且超过阈值的形状
        
        Args:
            mask_rle: 掩码RLE编码
            bbox: 边界框
            frame: 图像帧
            enable_region_check: 是否启用区域检查（默认False）
                如果启用，object不80%在frame中心3/4区域内则不计算affordance
        """
        iou_threshold = 0.7  # topdown使用0.7阈值
        
        if mask_rle and isinstance(mask_rle, dict) and 'counts' in mask_rle:
            try:
                # 解码掩码
                mask = coco_mask.decode(mask_rle)
                
                # 可选的区域检查：检查object是否80%在frame中心3/4区域内
                if enable_region_check:
                    frame_h, frame_w = frame.shape[:2]
                    
                    # 计算中心3/4区域
                    center_w = int(frame_w * 3 / 4)
                    center_h = int(frame_h * 3 / 4)
                    center_x1 = (frame_w - center_w) // 2
                    center_y1 = (frame_h - center_h) // 2
                    center_x2 = center_x1 + center_w
                    center_y2 = center_y1 + center_h
                    
                    print(f"TD策略：启用区域检查 - frame尺寸={frame_w}x{frame_h}, 中心3/4区域=[{center_x1},{center_y1},{center_x2},{center_y2}]")
                    
                    # 创建中心区域mask
                    center_region_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
                    center_region_mask[center_y1:center_y2, center_x1:center_x2] = 1
                    
                    # 计算object在中心区域的重叠比例
                    object_in_center = np.logical_and(mask, center_region_mask)
                    object_area = np.sum(mask > 0)
                    overlap_area = np.sum(object_in_center)
                    
                    if object_area > 0:
                        overlap_ratio = overlap_area / object_area
                        print(f"TD策略：object面积={object_area}, 重叠面积={overlap_area}, 重叠比例={overlap_ratio:.3f}")
                        
                        if overlap_ratio < 0.8:
                            print(f"TD策略：object不80%在中心3/4区域内（比例={overlap_ratio:.3f}），返回空affordance")
                            return []
                        else:
                            print(f"TD策略：object 80%以上在中心3/4区域内（比例={overlap_ratio:.3f}），继续计算affordance")
                    else:
                        print(f"TD策略：object面积为0，返回空affordance")
                        return []
                else:
                    print(f"TD策略：区域检查未启用，直接进行形状拟合")
                
                # 找到轮廓
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    return []  # 返回空affordance
                
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 计算两种形状的affordance和IoU
                shapes_results = []
                
                # 计算rot_bbox
                rot_bbox_result = self._calculate_rot_bbox_affordance(largest_contour, mask)
                if rot_bbox_result:
                    shapes_results.append(rot_bbox_result)
                
                # 计算circle
                circle_result = self._calculate_circle_affordance(largest_contour, mask)
                if circle_result:
                    shapes_results.append(circle_result)
                
                # 选择IoU最大且超过阈值的形状
                if shapes_results:
                    # 按IoU降序排序
                    shapes_results.sort(key=lambda x: x['iou'], reverse=True)
                    
                    best_shape = shapes_results[0]
                    if best_shape['iou'] >= iou_threshold:
                        return best_shape['affordance']
                    else:
                        print(f"所有形状IoU均低于阈值 {iou_threshold}，最佳IoU: {best_shape['iou']:.3f}")
                
                # 如果所有形状IoU都低于阈值或计算失败，返回空affordance
                return []
                
            except Exception as e:
                print(f"计算topdown affordance时发生错误: {e}")
                return []
        
        # 如果没有掩码，返回空affordance
        return []
    
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
                    print(f"BF策略（handle）：mask面积={handle_area:.1f}不在范围内(5000, 100000)，返回空affordance")
                    return [], None, None
                
                # 计算最小外接圆
                (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                
                # 检查circle半径是否小于fixed_width，如果是则返回空
                if radius < fixed_width:
                    print(f"BF策略（handle）：circle半径({radius:.1f}) < fixed_width({fixed_width:.1f})，返回空affordance")
                    return [], None, None
                
                # 验证circle与bbox的交集比例
                # bbox格式: [x1, y1, x2, y2]
                x1, y1, x2, y2 = bag_bbox[:4]
                
                # 计算圆形的面积
                circle_area = np.pi * radius * radius
                
                # 创建圆形mask来计算交集
                h, w = mask.shape
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
                
                print(f"BF策略（handle）：最小外接圆 - center=({cx:.1f},{cy:.1f}), radius={radius:.1f}, area={handle_area:.1f}, 交集比例={intersection_ratio:.3f}")
                
                # 只有当交集比例大于60%时才返回affordance
                if intersection_ratio > 0.6:
                    affordance = [float(cx), float(cy), float(radius)]
                    handle_hole = [float(cx), float(cy), float(radius)]  # handle的圆也作为handle_hole
                    print(f"BF策略（handle）：交集比例满足条件(>{0.6:.1%})，返回affordance")
                    return affordance, handle_hole, None
                else:
                    print(f"BF策略（handle）：交集比例不足(≤{0.6:.1%})，返回空affordance")
                    return [], None, None
            
            # 寻找洞（holes）- 使用轮廓层次结构检测内部轮廓
            h, w = mask.shape
            
            # 使用HoleProcessingUtils进行渐进式孔洞检测
            mask_processed, optimal_kernel_size, has_large_hole = HoleProcessingUtils.detect_holes_progressive(
                mask, area_threshold=3000, max_kernel_size=49)
            
            # 用小kernel开操作去除噪点
            open_kernel_size = max(3, optimal_kernel_size // 5)
            open_kernel_size = open_kernel_size if open_kernel_size % 2 == 1 else open_kernel_size + 1
            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
            mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_OPEN, open_kernel)
            
            print(f"BF策略：渐进式预处理 - 最优膨胀-腐蚀kernel={optimal_kernel_size}, 开操作kernel={open_kernel_size}")
            
            # 使用HoleProcessingUtils提取孔洞轮廓
            hole_contours, holes_mask = HoleProcessingUtils.extract_hole_contours(mask_processed)
            
            # 可视化hole_contours - 保存原始检测到的洞（默认关闭）
            if hole_contours:
                # frame_id = image_id - 1 if image_id else 0  # image_id是1-based，转换为0-based的frame_id
                # # 使用原始mask进行可视化，但hole_contours是从处理后的mask中提取的
                # HoleProcessingUtils.visualize_hole_contours(frame, mask, hole_contours, self.vis_dir, self.video_name, frame_id, bag_bbox)
                print(f"BF策略：检测到{len(hole_contours)}个孔洞轮廓")
            else:
                print(f"BF策略：预处理后仍未检测到孔洞轮廓")
            
            # 自适应形态学操作 - 根据mask大小和洞的初始大小调整kernel
            # 计算自适应kernel大小
            hole_ratio = 0  # 初始化hole_ratio，防止变量未定义错误
            if np.any(holes_mask):
                # 获取mask的尺寸
                h, w = mask.shape
                mask_size = min(h, w)
                
                # 计算初始洞的面积（用于评估洞的大小）
                initial_hole_area = np.sum(holes_mask > 0)
                
                # 找到洞的边界框，计算局部区域的mask面积
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
                        
                        print(f"BF策略：洞区域 [{x_min},{y_min},{x_max},{y_max}], 局部mask面积={local_mask_area}, 洞面积={initial_hole_area}")
                    else:
                        # 如果找不到轮廓，使用整个mask
                        mask_area = np.sum(mask > 0)
                        hole_ratio = initial_hole_area / mask_area if mask_area > 0 else 0
                else:
                    # 没有洞，ratio为0
                    hole_ratio = 0
                
                # 根据洞占局部mask的比例自适应调整kernel大小
                # 基于局部区域的比例会更高，所以调整阈值
                # 大洞（>40%）使用更大的kernel以连接更大的断裂
                # 小洞（<15%）使用较小的kernel以保持精确性
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
                
                print(f"BF策略：自适应形态学 - 洞面积比={hole_ratio:.3f}, kernel大小={kernel_size}")
                
                # 多级形态学操作以增加断裂容错性
                
                # 第一级：使用较小的kernel进行初步连接
                if kernel_size > 7:
                    pre_kernel_size = max(5, kernel_size // 2)
                    pre_kernel_size = pre_kernel_size if pre_kernel_size % 2 == 1 else pre_kernel_size + 1
                    pre_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pre_kernel_size, pre_kernel_size))
                    holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_CLOSE, pre_kernel)
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
                    print(f"BF策略：应用膨胀-腐蚀增强连接 - dilate_kernel={dilate_size}")
                    
                    # 最终平滑处理
                    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_CLOSE, smooth_kernel)
                    
                # 对于特大洞，进行形状优化
                if hole_ratio > 0.30:
                    # 使用开操作去除细小分支，保持主要结构
                    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_OPEN, open_kernel)
                    print(f"BF策略：优化特大洞形状")
            else:
                # 如果没有检测到初始洞，使用默认参数
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_CLOSE, kernel)
            
            # 如果找到洞，使用处理后的洞mask重新寻找轮廓
            if np.any(holes_mask):
                contours, _ = cv2.findContours(holes_mask.astype(np.uint8), 
                                              cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)
            else:
                contours = []
            
            if not contours and not hole_contours:
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
                print(f"BF策略：洞太小 (面积比: {hole_area/mask_area:.3f})，返回空affordance")
                return [], None, None
            
            # 对最大的洞进行圆形拟合
            (circle_cx, circle_cy), circle_radius = cv2.minEnclosingCircle(largest_hole)
            
            # 检查circle半径是否小于fixed_width，如果是则返回空
            if circle_radius < fixed_width:
                print(f"BF策略：circle半径({circle_radius:.1f}) < fixed_width({fixed_width:.1f})，返回空affordance")
                return [], None, None
            
            # 创建handle_hole信息（圆形）
            handle_hole = [float(circle_cx), float(circle_cy), float(circle_radius)]
            print(f"BF策略：检测到handle_hole - center=({circle_cx:.1f},{circle_cy:.1f}), radius={circle_radius:.1f}")
            
            # 使用独立函数进行基于circle和mask最高点的优化
            print(f"BF策略：开始优化affordance - circle center=({circle_cx:.1f},{circle_cy:.1f}), radius={circle_radius:.1f}")
            optimization_result = self._optimize_affordance_with_circle_mask_intersection(
                circle_center=[circle_cx, circle_cy],
                circle_radius=circle_radius,
                mask=mask,
                bag_bbox = bag_bbox,
                fixed_width=38.0
            )
            
            if optimization_result[0] is not None:
                optimized_affordance, rot_bbox_bag = optimization_result
                print(f"BF策略：使用优化后的affordance")
                return optimized_affordance, handle_hole, rot_bbox_bag
            else:
                # 如果优化失败，使用原始圆形拟合作为fallback
                #affordance = [float(circle_cx), float(circle_cy), float(circle_radius)]
                affordance = []
                print(f"BF策略：优化失败，使用原始圆形拟合 - center=({circle_cx:.1f},{circle_cy:.1f}), radius={circle_radius:.1f}")
                return affordance, handle_hole, None
            
        except Exception as e:
            print(f"计算backfront affordance时发生错误: {e}")
            return [], None, None

    def _calculate_rot_bbox_affordance(self, contour, mask):
        """计算旋转矩形affordance和IoU"""
        try:
            # 使用minAreaRect计算旋转矩形
            rect = cv2.minAreaRect(contour)
            (cx, cy), (w, h), angle = rect
            
            # 确保w >= h
            if w < h:
                w, h = h, w
                angle += 90
            
            # 角度归一化到[0, 180]
            angle = angle % 180
            angle_rad = angle / 180 * math.pi
            
            affordance = [float(cx), float(cy), float(w), float(h), float(angle_rad)]
            
            # 计算IoU
            guess_mask = np.zeros_like(mask)
            guess_mask = np.ascontiguousarray(guess_mask)
            
            rect_points = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(guess_mask, [rect_points], -1, 255, -1)
            
            iou = GeometryUtils.calculate_iou(mask, guess_mask)
            
            return {
                'name': 'rot_bbox',
                'affordance': affordance,
                'iou': iou
            }
            
        except Exception as e:
            print(f"计算rot_bbox affordance时发生错误: {e}")
            return None

    def _calculate_circle_affordance(self, contour, mask):
        """计算圆形affordance和IoU"""
        try:
            # 使用minEnclosingCircle计算圆形
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            affordance = [float(x), float(y), float(radius)]
            
            # 计算IoU
            guess_mask = np.zeros_like(mask)
            guess_mask = np.ascontiguousarray(guess_mask)
            
            cv2.circle(guess_mask, (int(x), int(y)), int(radius), 255, -1)
            
            iou = GeometryUtils.calculate_iou(mask, guess_mask)
            
            return {
                'name': 'circle',
                'affordance': affordance,
                'iou': iou
            }
            
        except Exception as e:
            print(f"计算circle affordance时发生错误: {e}")
            return None


    
    def _optimize_affordance_with_circle_mask_intersection(self, circle_center, circle_radius, mask, bag_bbox=None, fixed_width=38.0):
        """
        基于mask的rot_bbox和circle_center计算优化的affordance
        
        Args:
            circle_center: 圆心坐标 [cx, cy]
            circle_radius: 圆半径
            mask: 二值掩码
            bbox: 边界框 [x1, y1, x2, y2]，用于过滤affordance
            fixed_width: 固定短边宽度，默认38.0
            
        Returns:
            affordance: [xc, yc, w, h, angle] 或者 None（如果优化失败或被过滤）
            rot_bbox_bag: [xc, yc, w, h, angle] 或者 None
            
        Note:
            如果affordance与bbox的交集面积/affordance面积 > 80%，返回None, None
        """
        try:
            # 1. 计算mask的rot_bbox_bag
            mask_contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not mask_contours:
                print(f"优化失败：没有mask轮廓")
                return None
            
            # 获取最大的mask轮廓
            largest_mask_contour = max(mask_contours, key=cv2.contourArea)
            
            # 计算mask的最小外接旋转矩形
            rot_rect = cv2.minAreaRect(largest_mask_contour)
            (bag_cx, bag_cy), (bag_w, bag_h), bag_angle = rot_rect
            
            print(f"mask的rot_bbox: center=({bag_cx:.1f},{bag_cy:.1f}), size=({bag_w:.1f}x{bag_h:.1f}), angle={bag_angle:.1f}deg")
            
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
            # 找到两对平行边
            edges_with_y = []
            for i in range(4):
                edges_with_y.append((i, edge_midpoints[i][1]))  # (边索引, y坐标)
            
            # 按y坐标排序
            edges_with_y.sort(key=lambda x: x[1])
            
            # 选择y值最小的边作为edge1的参考
            edge1_idx = edges_with_y[0][0]  # y值最小的边
            edge1_direction = edge_directions[edge1_idx]
            
            # 确定edge1方向和垂直方向ev2
            # edge1方向就是y值较小的边的方向
            ev1 = edge1_direction
            # ev2是垂直于edge1的方向
            ev2 = np.array([-ev1[1], ev1[0]])  # 顺时针旋转90度得到垂直方向
            
            # 确保ev2的方向是合理的（可以选择向上或向下）
            # 根据实际需要调整ev2方向
            ev1_angle = math.degrees(math.atan2(ev1[1], ev1[0]))
            ev2_angle = math.degrees(math.atan2(ev2[1], ev2[0]))
            
            print(f"edge1方向(y值较小的边): 角度={ev1_angle:.1f}deg")
            print(f"ev2方向(垂直方向): 角度={ev2_angle:.1f}deg")
            
            # 找到平行于edge1方向的所有边，并按y值分类
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
                    print(f"找到edge1方向边: ({p1[0]:.1f},{p1[1]:.1f}) -> ({p2[0]:.1f},{p2[1]:.1f}), 中点y={edge_midpoint_y:.1f}, dot={dot_product:.3f}")
            
            if len(ev1_edges) == 0:
                print(f"优化失败：未找到平行于edge1方向的边")
                return None
            
            # 选择y值较小的边作为真正的edge1边
            if len(ev1_edges) >= 2:
                min_y_index = edge_y_values.index(min(edge_y_values))
                selected_edge = ev1_edges[min_y_index]
                ev1_edges = [selected_edge]  # 只使用y值最小的边
                print(f"选择y值最小的边: ({selected_edge[0][0]:.1f},{selected_edge[0][1]:.1f}) -> ({selected_edge[1][0]:.1f},{selected_edge[1][1]:.1f}), y={min(edge_y_values):.1f}")
            else:
                print(f"只有一条edge1方向边，直接使用")
            
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
                
                print(f"尝试circle_center沿ev2{direction_name}射线: ({circle_center[0]:.1f},{circle_center[1]:.1f}) -> ({ray_end[0]:.1f},{ray_end[1]:.1f})")
                
                # 检查射线与所有edge1方向边的交点
                for p1, p2 in ev1_edges:
                    intersect = GeometryUtils.line_intersect(circle_center_array, ray_end, p1, p2)
                    if intersect is not None:
                        intersect = np.array(intersect)
                    if intersect is not None:
                        # 检查交点是否在射线的正方向上
                        to_intersect = intersect - circle_center_array
                        if np.dot(to_intersect, direction) > 0:  # 在指定方向上
                            distance = np.linalg.norm(to_intersect)
                            if distance < min_distance:
                                min_distance = distance
                                intersection_point = intersect
                                chosen_direction = direction
                                print(f"找到有效交点: ({intersect[0]:.1f},{intersect[1]:.1f}), 距离={distance:.1f}, 方向={direction_name}")
            
            if intersection_point is None:
                print(f"优化失败：未找到circle_center沿ev2方向与edge1的交点")
                return None
            
            ox, oy = intersection_point
            print(f"最终选择交点o=({ox:.1f},{oy:.1f}), 距离circle_center={min_distance:.1f}")
            
            # 3. 定义aff_box是一个rot_bbox，使用几何点构建方法
            # aff_center为o点
            aff_center_x, aff_center_y = ox, oy
            
            # circle_center为中心点，沿ev1方向扩展得到width/2=38/2=19得到aff_box的p3和p4点
            half_width = fixed_width / 2.0  # 38/2 = 19
            
            # 计算p3和p4点：沿ev1方向从circle_center扩展±19
            p3 = circle_center_array + ev1 * half_width   # circle_center + ev1方向 * 19
            p4 = circle_center_array - ev1 * half_width   # circle_center - ev1方向 * 19
            
            print(f"circle_center为中心沿ev1方向扩展±{half_width:.1f}:")
            print(f"  - p3: ({p3[0]:.1f},{p3[1]:.1f})")
            print(f"  - p4: ({p4[0]:.1f},{p4[1]:.1f})")
            
            # circle_center与o的连线方向cov，长度*2为l_co
            co_vector = intersection_point - circle_center_array
            l_co = min_distance * 2.0  # 长度*2
            cov_normalized = co_vector / np.linalg.norm(co_vector)  # 归一化方向向量
            
            print(f"circle_center与o的连线方向cov，长度*2={l_co:.1f}")
            
            # p4沿着cov方向和长度l_co得到p1
            p1 = p4 + cov_normalized * l_co
            
            # p3沿着cov方向和长度l_co得到p2  
            p2 = p3 + cov_normalized * l_co
            
            print(f"构建aff_box的四个顶点:")
            print(f"  - p1: ({p1[0]:.1f},{p1[1]:.1f}) = p4 + cov*{l_co:.1f}")
            print(f"  - p2: ({p2[0]:.1f},{p2[1]:.1f}) = p3 + cov*{l_co:.1f}")
            print(f"  - p3: ({p3[0]:.1f},{p3[1]:.1f})")
            print(f"  - p4: ({p4[0]:.1f},{p4[1]:.1f})")
            
            # (p1,p2,p3,p4)构成aff_box的rot_bbox
            # 计算aff_box的中心、尺寸和角度
            box_points = np.array([p1, p2, p3, p4])
            
            # 计算宽度和高度
            # 宽度：p1-p4或p2-p3的距离（沿cov方向）
            aff_w = np.linalg.norm(p1 - p4)
            # 高度：p3-p4或p1-p2的距离（沿ev1方向）  
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
            
            if aff_angle_deg < 0: aff_angle_deg += 180  # 转换到[0, 180] 
            
            
            aff_angle_rad = math.radians(aff_angle_deg)
            
            print(f"aff_box构建完成:")
            print(f"  - aff_center(o): ({aff_center_x:.1f},{aff_center_y:.1f})")
            print(f"  - circle_center: ({circle_center[0]:.1f},{circle_center[1]:.1f})")
            print(f"  - 宽度w(p1-p4距离): {aff_w:.1f}")
            print(f"  - 高度h(p3-p4距离): {aff_h:.1f}")
            print(f"  - angle(cov方向与x轴夹角): {aff_angle_deg:.1f}deg ({aff_angle_rad:.3f}rad)")
            print(f"  - edge1方向(y值较小的边): 角度={ev1_angle:.1f}deg")
            print(f"  - ev2方向(垂直边方向): 角度={ev2_angle:.1f}deg")
            print(f"  - 几何关系: 通过(p1,p2,p3,p4)四点构建rot_bbox")
            
            # 构建rot_bbox_bag信息 [xc, yc, w, h, angle_in_radians]
            rot_bbox_bag = [float(bag_cx), float(bag_cy), float(bag_w), float(bag_h), float(angle_rad)]
            
            # 返回旋转矩形格式的affordance [xc, yc, w, h, angle_in_radians]
            affordance = [float(aff_center_x), float(aff_center_y), float(aff_w), float(aff_h), float(aff_angle_rad)]
            
            # 过滤：如果affordance和bbox的交集/affordance > 80%，返回空
            if bag_bbox is not None:
                intersection_ratio = GeometryUtils.calculate_rotated_bbox_intersection_ratio(affordance, bag_bbox)
                print(f"affordance与bbox交集比例: {intersection_ratio:.1%}")
                if intersection_ratio > 0.8:
                    print(f"affordance与bbox交集比例过高(>{0.8:.1%})，返回空")
                    return None, None
            
            return affordance, rot_bbox_bag
            
        except Exception as e:
            print(f"优化affordance时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    


    def get_frame_info(self):
        """
        获取视频帧信息
        
        Returns:
            dict: 视频信息字典
        """
        return {
            'video_path': self.video_path,
            'original_fps': self.original_fps,
            'target_fps': self.target_fps,
            'total_frames': self.total_frames,
            'width': self.video_width,
            'height': self.video_height,
            'crop_top_half': self.crop_top_half,
            'frame_interval': self.frame_interval,
            'output_dir': self.output_dir,
            'max_frames': self.max_frames,
            'start_frame': self.start_frame
        }

    def __del__(self):
        """析构函数，确保释放视频资源"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()


def main():
    """主函数，提供命令行接口"""
    parser = argparse.ArgumentParser(description='视频帧提取和目标检测工具')
    parser.add_argument('--video', '-v', type=str, 
                       default='data/bag/0912/topdown01.mp4',
                       help='输入视频路径')
    parser.add_argument('--output', '-o', type=str, 
                       default='data/bag/0912',
                       help='输出根目录')
    parser.add_argument('--fps', '-f', type=float, default=0.1,
                       help='目标帧率（帧/秒），默认0.5（每2秒1帧）')
    parser.add_argument('--no-crop', action='store_true',
                       help='不裁剪图像，保留完整画面')
    parser.add_argument('--no-detect', action='store_true',
                       help='不进行目标检测，仅提取帧')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='最大处理帧数，None表示不限制')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='开始处理的帧数，默认从第0帧开始')
    
    args = parser.parse_args()
    
    # 检查视频文件是否存在
    if not os.path.exists(args.video):
        print(f"错误: 视频文件不存在 - {args.video}")
        return
    
    try:
        # 创建视频处理器
        processor = VideoProcessor(
            video_path=args.video,
            output_dir=args.output,
            frame_rate=args.fps,
            crop_top_half=not args.no_crop,
            detect_objects=not args.no_detect,
            max_frames=args.max_frames,
            start_frame=args.start_frame
        )
        
        # 显示视频信息
        info = processor.get_frame_info()
        print(f"\n处理设置:")
        print(f"  输入视频: {info['video_path']}")
        print(f"  输出目录: {info['output_dir']}")
        print(f"  是否裁剪上半部分: {info['crop_top_half']}")
        print(f"  是否进行目标检测: {not args.no_detect}")
        if info['start_frame'] > 0:
            print(f"  开始帧: {info['start_frame']}")
        if info['max_frames'] is not None:
            print(f"  最大处理帧数: {info['max_frames']}")
        
        # 处理视频
        result = processor.process_video()
        
        if result['saved_images'] > 0:
            print(f"\n✓ 处理完成!")
            print(f"  成功保存图像: {result['saved_images']}")
            if not args.no_detect:
                print(f"  检测处理帧数: {result['processed_frames']}")
                print(f"  总标注数量: {result['total_annotations']}")
                print(f"  处理时间: {result['processing_time']:.2f} 秒")
            print(f"  图像保存到: {processor.img_dir}")
            if not args.no_detect:
                print(f"  标注保存到: {processor.ann_dir}")
                print(f"  可视化保存到: {processor.vis_dir}")
        else:
            print(f"\n✗ 没有成功提取任何图像")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()