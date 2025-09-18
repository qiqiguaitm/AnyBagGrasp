import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
import h5py
from pycocotools import mask as coco_mask
from common import COCOUtils, VideoUtils
from aff4bev import Affordance4BEVGraspBag
from aff4pv import Affordance4PVGraspBag

class VideoProcessor:
    def __init__(self, video_path, output_dir="data", frame_rate=2.0, crop_top_half=True, max_frames=None, start_frame=0, depth_file=None):
        """
        视频和深度数据处理类 - 支持目标检测和COCO格式保存
        
        Args:
            video_path (str): 输入视频路径（MP4文件）
            output_dir (str): 输出根目录
            frame_rate (float): 提取帧率（帧/秒），默认2.0（每0.5秒1帧）
            crop_top_half (bool): 是否只提取图像上半部分
            max_frames (int): 最大处理帧数，None表示不限制
            start_frame (int): 开始处理的帧数，默认从第0帧开始
            depth_file (str): 深度数据文件路径（H5文件），用于TD策略，None表示使用模拟深度
        """
        self.video_path = video_path
        self.video_name = Path(video_path).stem
        self.output_dir = output_dir
        self.frame_rate = frame_rate
        self.crop_top_half = crop_top_half
        self.max_frames = max_frames
        self.start_frame = start_frame
        self.depth_file = depth_file
        
        # 创建目录结构
        self.img_dir = os.path.join(output_dir, 'images')
        self.ann_dir = os.path.join(output_dir, 'anno')
        
        Path(self.img_dir).mkdir(parents=True, exist_ok=True)
        Path(self.ann_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化Affordance处理器
        self.bev_affordance = Affordance4BEVGraspBag(
            detection_text="paper bag.plastic bag.bag",
            output_dir=os.path.join(output_dir, 'bev_vis'),
            verbose=True,
            enable_cache=True,
            save_individual=False
        )
        
        self.pv_affordance = Affordance4PVGraspBag(
            detection_text="paper bag.plastic bag.bag",
            output_dir=os.path.join(output_dir, 'pv_vis'),
            verbose=True,
            enable_cache=True
        )
        
        # 初始化视频捕获
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
            
        # 获取视频属性
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 初始化深度数据（如果提供）
        self.depth_data = None
        self.depth_dataset = None
        if self.depth_file is not None:
            self._init_depth_data()
        
        # 计算帧间隔（每2秒1帧）
        self.frame_interval = max(1, int(self.original_fps / frame_rate))
        self.target_fps = frame_rate
        
        # COCO数据格式
        self.coco_data = {
            'info': {
                'description': f'handle.bag detection from {self.video_name}',
                'url': '',
                'version': '1.2',
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
                {'id': 0, 'name': 'bag', 'supercategory': 'object'},
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
        
        # 验证深度文件（如果提供）
        if self.depth_file is not None:
            self._validate_depth_file()

    def _init_depth_data(self):
        """初始化深度数据，打开H5文件并准备数据集"""
        try:
            if not os.path.exists(self.depth_file):
                print(f"Warning: 深度文件不存在: {self.depth_file}")
                return
                
            # 打开H5文件（保持打开状态）
            self.depth_data = h5py.File(self.depth_file, 'r')
            
            # 找到主要数据集
            if 'depth' in self.depth_data:
                self.depth_dataset = self.depth_data['depth']
                dataset_name = 'depth'
            elif 'raw_data' in self.depth_data:
                self.depth_dataset = self.depth_data['raw_data']
                dataset_name = 'raw_data'
            else:
                # 尝试第一个数据集
                dataset_names = list(self.depth_data.keys())
                if dataset_names:
                    self.depth_dataset = self.depth_data[dataset_names[0]]
                    dataset_name = dataset_names[0]
                else:
                    print(f"Warning: H5文件中没有找到数据集")
                    self.depth_data.close()
                    self.depth_data = None
                    return
            
            print(f"  深度数据已初始化: 数据集 '{dataset_name}', 形状 {self.depth_dataset.shape}")
            
        except Exception as e:
            print(f"Warning: 初始化深度数据失败: {e}")
            if self.depth_data is not None:
                self.depth_data.close()
                self.depth_data = None

    def _validate_depth_file(self):
        """验证深度数据文件的格式和内容"""
        try:
            if not os.path.exists(self.depth_file):
                print(f"Warning: 深度文件不存在: {self.depth_file}")
                return
                
            with h5py.File(self.depth_file, 'r') as f:
                print(f"  深度文件信息:")
                print(f"    - 文件: {self.depth_file}")
                
                # 列出所有数据集
                dataset_names = list(f.keys())
                print(f"    - 数据集: {dataset_names}")
                
                # 检查主要数据集
                main_dataset = None
                if 'depth' in f:
                    main_dataset = f['depth']
                    dataset_name = 'depth'
                elif 'raw_data' in f:
                    main_dataset = f['raw_data']
                    dataset_name = 'raw_data'
                elif dataset_names:
                    main_dataset = f[dataset_names[0]]
                    dataset_name = dataset_names[0]
                
                if main_dataset is not None:
                    print(f"    - 主数据集: {dataset_name}")
                    print(f"    - 形状: {main_dataset.shape}")
                    print(f"    - 数据类型: {main_dataset.dtype}")
                    
                    # 检查数据范围
                    if main_dataset.shape[0] > 0:
                        sample_frame = main_dataset[0]
                        print(f"    - 深度范围: {np.min(sample_frame):.3f} - {np.max(sample_frame):.3f}")
                else:
                    print(f"    - Warning: 未找到可用的数据集")
                    
        except Exception as e:
            print(f"Warning: 验证深度文件失败: {e}")

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

    def process_data(self):
        """
        处理视频和深度数据，提取帧并进行目标检测，保存COCO格式结果
        
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
                    
                    # 进行目标检测和affordance计算（使用同步深度数据）
                    # 从已打开的深度数据集中同步读取深度帧
                    depth_frame = None
                    if self.depth_dataset is not None:
                        try:
                            if frame_count < self.depth_dataset.shape[0]:
                                depth_frame = self.depth_dataset[frame_count]
                            else:
                                if processed_count == 0:  # 只在第一次警告
                                    print(f"Warning: 深度数据不足，视频帧数 > 深度帧数 ({self.depth_dataset.shape[0]})")
                        except Exception as e:
                            if processed_count == 0:  # 只在第一次警告
                                print(f"Warning: 读取深度帧 {frame_count} 失败: {e}")
                    
                    self._process_frame_with_affordance_sync(processed_frame, img_path, frame_count, depth_frame)
                    processed_count += 1
                    
                    if processed_count % 5 == 0:
                        print(f"已处理 {processed_count} 帧，检测到 {len(self.coco_data['annotations'])} 个对象...")
                else:
                    print(f"保存失败: {img_path}")
            
            frame_count += 1
            
        self.cap.release()
        
        # 关闭深度数据文件
        if self.depth_data is not None:
            self.depth_data.close()
            print("深度数据文件已关闭")
        
        # 保存COCO标注文件
        COCOUtils.save_coco_annotations(self.coco_data, self.ann_dir, self.video_name)
        # 生成可视化视频 - 根据策略选择正确的可视化目录
        grasp_policy = self._get_grasp_policy_from_filename()
        if grasp_policy == 'bf':
            # BF策略使用pv_vis目录
            pv_vis_dir = os.path.join(self.output_dir, 'pv_vis')
            VideoUtils.create_visualization_video(pv_vis_dir, self.video_name, self.target_fps)
        elif grasp_policy == 'td':
            # TD策略使用bev_vis目录  
            bev_vis_dir = os.path.join(self.output_dir, 'bev_vis')
            VideoUtils.create_visualization_video(bev_vis_dir, self.video_name, self.target_fps)
        
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

    def _process_frame_with_affordance(self, frame, file_path, frame_count):
        """使用Affordance类处理单帧图像"""
        try:
            # 添加图像信息到COCO
            image_id = frame_count + 1
            self._add_image_to_coco(file_path, frame.shape, image_id)
            
            # 根据策略选择对应的Affordance类
            grasp_policy = self._get_grasp_policy_from_filename()
            
            if grasp_policy == 'td':
                # TD策略：使用Affordance4BEVGraspBag
                depth_data = self._load_depth_data(file_path, frame_count)
                self._process_with_bev_affordance(frame, file_path, image_id, depth_data=depth_data)
            elif grasp_policy == 'bf':
                # BF策略：使用Affordance4PVGraspBag  
                self._process_with_pv_affordance(frame, file_path, image_id)
            else:
                print(f"未知的grasp_policy: {grasp_policy}")
                
        except Exception as e:
            print(f"处理帧 {file_path} 时发生错误: {e}")
    
    def _process_frame_with_affordance_sync(self, frame, file_path, frame_count, depth_frame=None):
        """使用Affordance类处理单帧图像，支持同步深度数据"""
        try:
            # 添加图像信息到COCO
            image_id = frame_count + 1
            self._add_image_to_coco(file_path, frame.shape, image_id)
            
            # 根据策略选择对应的Affordance类
            grasp_policy = self._get_grasp_policy_from_filename()
            
            if grasp_policy == 'td':
                # TD策略：使用Affordance4BEVGraspBag，传入同步的深度数据
                self._process_with_bev_affordance(frame, file_path, image_id, depth_data=depth_frame)
            elif grasp_policy == 'bf':
                # BF策略：使用Affordance4PVGraspBag  
                self._process_with_pv_affordance(frame, file_path, image_id)
            else:
                print(f"未知的grasp_policy: {grasp_policy}")
                
        except Exception as e:
            print(f"处理帧 {file_path} 时发生错误: {e}")
    
    def _load_depth_data(self, rgb_file_path, frame_count):
        """从已打开的深度数据集中加载对应帧的深度数据"""
        if self.depth_dataset is None:
            return None
            
        try:
            # 检查帧索引是否有效
            if frame_count < self.depth_dataset.shape[0]:
                depth_data = self.depth_dataset[frame_count]
                return depth_data
            else:
                if frame_count == 0:  # 只在第一次警告
                    print(f"Warning: 帧索引 {frame_count} 超出深度数据范围 {self.depth_dataset.shape[0]}")
                return None
                    
        except Exception as e:
            print(f"加载深度数据失败: {e}")
            return None
    
    def _process_with_bev_affordance(self, frame, file_path, image_id, depth_data=None):
        """使用BEV Affordance处理帧"""
        if self.bev_affordance is None:
            return
            
        # 如果没有深度数据，直接跳过BEV处理
        if depth_data is None:
            if self.verbose:
                print("Warning: 没有深度数据，跳过BEV affordance处理")
            return
            
        try:
            # 保存临时RGB和深度文件
            temp_rgb_path = file_path
            temp_depth_path = file_path.replace('.jpg', '_depth.npy')
            np.save(temp_depth_path, depth_data)
            
            # 调用BEV Affordance处理
            bev_result = self.bev_affordance._process_single(Path(temp_rgb_path), Path(temp_depth_path))
            
            # 从BEV结果中提取信息并添加到COCO
            self._extract_bev_results_to_coco(bev_result, image_id)
            
            # 清理临时深度文件
            if os.path.exists(temp_depth_path):
                os.remove(temp_depth_path)
                
        except Exception as e:
            print(f"BEV affordance处理失败: {e}")
    
    def _process_with_pv_affordance(self, frame, file_path, image_id):
        """使用PV Affordance处理帧"""
        if self.pv_affordance is None:
            return
            
        try:
            # 直接使用已保存的crop后图像文件路径
            pv_result = self.pv_affordance._process_single(Path(file_path))
            
            # 从PV结果中提取信息并添加到COCO
            self._extract_pv_results_to_coco(pv_result, image_id)
                
        except Exception as e:
            print(f"PV affordance处理失败: {e}")
    
    def _extract_bev_results_to_coco(self, bev_result, image_id):
        """从BEV结果中提取信息并添加到COCO格式"""
        if not bev_result or 'detections' not in bev_result or 'affordances' not in bev_result:
            return
            
        detections = bev_result.get('detections', [])
        affordances = bev_result.get('affordances', [])
        
        # 匹配detection和affordance
        for i, det in enumerate(detections):
            bbox = det.get('bbox', [])
            mask_rle = det.get('mask', None)
            category = det.get('label', 'bag').lower()
            score = det.get('confidence', 0.0)
            
            # 确定类别ID
            if 'bag' in category:
                category_id = 0
            else:
                continue  # 只处理bag类别
            
            if len(bbox) >= 4:
                # bbox格式已经是[x, y, width, height]
                bbox_coco = [float(b) for b in bbox]
                area = float(bbox[2] * bbox[3])
                
                # 找到对应的affordance
                affordance_data = None
                if i < len(affordances) and affordances[i].get('aff_rot_bbox'):
                    aff_rot_bbox_list = affordances[i].get('aff_rot_bbox', [])
                    affordance_data = aff_rot_bbox_list 
                top_rot_bbox = affordances[i].get('rot_bbox') if i < len(affordances) else None
                # 验证affordance格式
                if not affordance_data:
                    continue  # 跳过无效的affordance
                
                annotation = {
                    'id': int(self.annotation_id),
                    'image_id': int(image_id),
                    'category_id': int(category_id),
                    'cat': int(category_id),
                    'bbox': bbox_coco,
                    'area': area,
                    'iscrowd': 0,
                    'score': float(score),
                    'affordance': affordance_data,
                    'grasp_policy': 'td',
                    'handle_hole': None,
                    'top_rot_bbox':  top_rot_bbox
                }
                
                # 添加分割掩码（如果存在）
                if mask_rle:
                    try:
                        if isinstance(mask_rle, dict) and 'counts' in mask_rle:
                            annotation['segmentation'] = mask_rle
                            annotation['area'] = float(coco_mask.area(mask_rle))
                    except Exception as e:
                        print(f"处理掩码时发生错误: {e}")
                
                self.coco_data['annotations'].append(annotation)
                self.annotation_id += 1
    
    def _extract_pv_results_to_coco(self, pv_result, image_id):
        """从PV结果中提取信息并添加到COCO格式"""
        if not pv_result or 'detections' not in pv_result or 'affordances' not in pv_result:
            return
            
        detections = pv_result.get('detections', [])
        affordances = pv_result.get('affordances', [])
        
        # 匹配detection和affordance
        for i, det in enumerate(detections):
            bbox = det.get('bbox', [])
            mask_rle = det.get('mask', None)
            category = det.get('label', 'bag').lower()
            score = det.get('confidence', 0.0)
            
            # 确定类别ID
            if 'bag' in category:
                category_id = 0
            else:
                continue  # 只处理bag类别
            
            if len(bbox) >= 4:
                # bbox格式已经是[x, y, width, height]
                bbox_coco = [float(b) for b in bbox]
                area = float(bbox[2] * bbox[3])
                
                # 找到对应的affordance
                affordance_data = None
                handle_hole = None
                rot_bbox_bag = None
                
                if i < len(affordances):
                    aff_result = affordances[i]
                    affordance_data = aff_result.get('affordance', [])
                    handle_hole = aff_result.get('handle_hole')
                    rot_bbox_bag = aff_result.get('rot_bbox_bag')
                
                # 验证affordance格式：对于PV，affordance可能是[cx, cy, radius]或[]
                if not affordance_data:
                    continue  # 跳过无效的affordance
                
                # 将affordance包装成列表格式
                if affordance_data and not isinstance(affordance_data[0], list):
                    affordance_data = [affordance_data]
                
                annotation = {
                    'id': int(self.annotation_id),
                    'image_id': int(image_id),
                    'category_id': int(category_id),
                    'cat': int(category_id),
                    'bbox': bbox_coco,
                    'area': area,
                    'iscrowd': 0,
                    'score': float(score),
                    'affordance': affordance_data,
                    'grasp_policy': 'bf',
                    'handle_hole': handle_hole if handle_hole else None,
                    'rot_bbox_bag': rot_bbox_bag if rot_bbox_bag else None
                }
                
                # 添加分割掩码（如果存在）
                if mask_rle:
                    try:
                        if isinstance(mask_rle, dict) and 'counts' in mask_rle:
                            annotation['segmentation'] = mask_rle
                            annotation['area'] = float(coco_mask.area(mask_rle))
                    except Exception as e:
                        print(f"处理掩码时发生错误: {e}")
                
                self.coco_data['annotations'].append(annotation)
                self.annotation_id += 1

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



    def _get_grasp_policy_from_filename(self):
        """根据视频文件名确定grasp_policy"""
        video_name_lower = self.video_name.lower()
        
        if 'topdown' in video_name_lower:
            return 'td'
        elif 'backfront' in video_name_lower:
            return 'bf'
        else:
            return 'no'



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
                       default=None,
                       help='输出根目录，默认使用视频文件所在目录')
    parser.add_argument('--fps', '-f', type=float, default=2.0,
                       help='目标帧率（帧/秒），默认2.0（每0.5秒1帧）')
    parser.add_argument('--no-crop', action='store_true',
                       help='不裁剪图像，保留完整画面')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='最大处理帧数，None表示不限制')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='开始处理的帧数，默认从第0帧开始')
    parser.add_argument('--depth-file', type=str, default=None,
                       help='深度数据文件路径（H5文件），用于TD策略（可选）')
    
    args = parser.parse_args()
    
    # 检查视频文件是否存在
    if not os.path.exists(args.video):
        print(f"错误: 视频文件不存在 - {args.video}")
        return
    
    # 如果output参数为None，使用视频文件所在目录
    output_dir = args.output
    if output_dir is None:
        output_dir = os.path.dirname(args.video)
        print(f"使用视频文件所在目录作为输出目录: {output_dir}")
    
    try:
        # 创建视频处理器
        processor = VideoProcessor(
            video_path=args.video,
            output_dir=output_dir,
            frame_rate=args.fps,
            crop_top_half=not args.no_crop,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            depth_file=args.depth_file
        )
        
        # 显示视频信息
        info = processor.get_frame_info()
        print(f"\n处理设置:")
        print(f"  输入视频: {info['video_path']}")
        print(f"  输出目录: {info['output_dir']}")
        print(f"  是否裁剪上半部分: {info['crop_top_half']}")
        if info['start_frame'] > 0:
            print(f"  开始帧: {info['start_frame']}")
        if info['max_frames'] is not None:
            print(f"  最大处理帧数: {info['max_frames']}")
        
        # 处理视频和深度数据
        result = processor.process_data()
        
        if result['saved_images'] > 0:
            print(f"\n✓ 处理完成!")
            print(f"  成功保存图像: {result['saved_images']}")
            print(f"  检测处理帧数: {result['processed_frames']}")
            print(f"  总标注数量: {result['total_annotations']}")
            print(f"  处理时间: {result['processing_time']:.2f} 秒")
            print(f"  图像保存到: {processor.img_dir}")
            print(f"  标注保存到: {processor.ann_dir}")
            # 可视化保存到策略特定目录（pv_vis或bev_vis）
        else:
            print(f"\n✗ 没有成功提取任何图像")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()