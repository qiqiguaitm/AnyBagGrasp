#!/usr/bin/env python3
"""
集成CDM缓存功能的处理器
用于在实际应用中使用带缓存的CDM深度增强
"""

import torch
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import time

# Add CDM path
CDM_DIR = Path(__file__).parent
sys.path.insert(0, str(CDM_DIR))

from cdm_cache import CachedCDMProcessor, CDMCache

class CDMProcessorWithCache:
    """带缓存的CDM深度处理器完整实现"""
    
    def __init__(self, model_path: str = None, cache_dir: str = ".cache/cdm", 
                 enable_cache: bool = True, device: str = None):
        """
        初始化CDM处理器
        
        Args:
            model_path: 模型文件路径，默认为 cdm/cdm_d435.ckpt
            cache_dir: 缓存目录
            enable_cache: 是否启用缓存
            device: 设备类型 ('cuda', 'cpu' 或 None自动选择)
        """
        # 设置设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        
        # 设置模型路径
        if model_path is None:
            model_path = CDM_DIR / "cdm_d435.ckpt"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        # 创建带缓存的处理器
        self.processor = CachedCDMProcessor(
            self.model, 
            cache_dir=cache_dir, 
            enable_cache=enable_cache
        )
        
        self.enable_cache = enable_cache
        print(f"缓存状态: {'启用' if enable_cache else '禁用'}")
        if enable_cache:
            print(f"缓存目录: {cache_dir}")
    
    def _load_model(self, model_path: Path):
        """加载CDM模型"""
        try:
            from rgbddepth.dpt import RGBDDepth
            
            print(f"加载模型: {model_path}")
            print(f"模型大小: {model_path.stat().st_size / (1024**3):.2f} GB")
            
            # 创建模型
            model = RGBDDepth(encoder='vitl', features=256)
            
            # 加载权重
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 处理不同的checkpoint格式
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # 移除module前缀（如果有）
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    cleaned_state_dict[key[7:]] = value
                else:
                    cleaned_state_dict[key] = value
            
            # 加载状态
            model.load_state_dict(cleaned_state_dict, strict=False)
            
            # 设置为评估模式
            model.to(self.device)
            model.eval()
            
            print("✓ 模型加载成功")
            return model
            
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            raise
    
    def process_depth(self, rgb: np.ndarray, depth: np.ndarray, 
                     target_size: int = 518) -> np.ndarray:
        """
        处理深度图像
        
        Args:
            rgb: RGB图像 (H, W, 3) uint8 或 BGR格式
            depth: 深度图像 (H, W) float32，单位：米
            target_size: 模型输入尺寸
            
        Returns:
            np.ndarray: 增强后的深度图像 (H, W) float32，单位：米
        """
        # 确保RGB是正确格式
        if rgb.shape[-1] == 3 and rgb.dtype == np.uint8:
            # 假设输入是BGR（OpenCV格式），转换为RGB
            if len(rgb.shape) == 3:
                rgb_input = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            else:
                rgb_input = rgb
        else:
            rgb_input = rgb
        
        # 确保深度是float32
        if depth.dtype != np.float32:
            depth_input = depth.astype(np.float32)
        else:
            depth_input = depth
        
        # 使用缓存处理器
        start_time = time.time()
        enhanced_depth = self.processor.process(
            rgb_input, 
            depth_input, 
            target_size=target_size
        )
        process_time = time.time() - start_time
        
        print(f"处理耗时: {process_time:.2f} 秒")
        
        return enhanced_depth
    
    def process_rgbd_file(self, rgb_path: str, depth_path: str, 
                         save_path: str = None) -> np.ndarray:
        """
        处理RGB和深度文件
        
        Args:
            rgb_path: RGB图像文件路径
            depth_path: 深度文件路径（.npy或图像文件）
            save_path: 保存增强深度的路径（可选）
            
        Returns:
            np.ndarray: 增强后的深度图像
        """
        # 读取RGB图像
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            raise ValueError(f"无法读取RGB图像: {rgb_path}")
        
        # 读取深度数据
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        else:
            # 假设是深度图像文件
            depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth_img is None:
                raise ValueError(f"无法读取深度图像: {depth_path}")
            # 转换为米（假设原始单位是毫米）
            depth = depth_img.astype(np.float32) / 1000.0
        
        print(f"输入RGB: {rgb.shape}, 深度: {depth.shape}")
        print(f"深度范围: {depth[depth>0].min():.3f} - {depth[depth>0].max():.3f} 米")
        
        # 处理深度
        enhanced_depth = self.process_depth(rgb, depth)
        
        print(f"增强深度范围: {enhanced_depth[enhanced_depth>0].min():.3f} - "
              f"{enhanced_depth[enhanced_depth>0].max():.3f} 米")
        
        # 保存结果（如果指定）
        if save_path:
            if save_path.endswith('.npy'):
                np.save(save_path, enhanced_depth)
            else:
                # 保存为深度图像（转换为毫米）
                depth_mm = (enhanced_depth * 1000).astype(np.uint16)
                cv2.imwrite(save_path, depth_mm)
            print(f"✓ 增强深度已保存到: {save_path}")
        
        return enhanced_depth
    
    def batch_process(self, rgb_list, depth_list, show_progress: bool = True):
        """
        批量处理多个RGB-D对
        
        Args:
            rgb_list: RGB图像列表
            depth_list: 深度图像列表
            show_progress: 是否显示进度
            
        Returns:
            list: 增强深度列表
        """
        results = []
        total = len(rgb_list)
        
        print(f"批量处理 {total} 个RGB-D对...")
        
        for i, (rgb, depth) in enumerate(zip(rgb_list, depth_list)):
            if show_progress:
                print(f"\n处理 [{i+1}/{total}]...")
            
            enhanced = self.process_depth(rgb, depth)
            results.append(enhanced)
        
        print(f"\n✓ 批量处理完成")
        
        # 显示缓存统计
        if self.enable_cache:
            cache_info = self.processor.get_cache_info()
            print(f"\n缓存统计:")
            print(f"  缓存文件数: {cache_info['total_files']}")
            print(f"  缓存大小: {cache_info['total_size_mb']:.2f} MB")
            print(f"  总访问次数: {cache_info['total_accesses']}")
        
        return results
    
    def clear_cache(self):
        """清空缓存"""
        if self.processor:
            self.processor.clear_cache()
            print("✓ 缓存已清空")
    
    def get_cache_info(self):
        """获取缓存信息"""
        if self.processor:
            return self.processor.get_cache_info()
        return None


def demo_cached_cdm():
    """演示带缓存的CDM处理"""
    print("="*60)
    print("CDM缓存处理演示")
    print("="*60)
    
    # 创建处理器（启用缓存）
    processor = CDMProcessorWithCache(
        model_path=None,  # 使用默认路径
        cache_dir=".cache/cdm_demo",
        enable_cache=True,
        device=None  # 自动选择
    )
    
    # 创建测试数据
    print("\n创建测试数据...")
    rgb1 = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    depth1 = np.random.uniform(0.2, 1.5, (720, 1280)).astype(np.float32)
    
    rgb2 = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    depth2 = np.random.uniform(0.2, 1.5, (720, 1280)).astype(np.float32)
    
    # 第一次处理（会进行计算并缓存）
    print("\n第一次处理RGB-D对1（应该进行计算）:")
    enhanced1_first = processor.process_depth(rgb1, depth1)
    
    # 第二次处理相同数据（应该使用缓存）
    print("\n第二次处理RGB-D对1（应该使用缓存）:")
    enhanced1_second = processor.process_depth(rgb1, depth1)
    
    # 验证结果一致
    if np.allclose(enhanced1_first, enhanced1_second):
        print("✓ 缓存结果与计算结果一致")
    
    # 处理不同的数据
    print("\n处理RGB-D对2（新数据，应该进行计算）:")
    enhanced2 = processor.process_depth(rgb2, depth2)
    
    # 批量处理（混合新旧数据）
    print("\n批量处理测试:")
    rgb_list = [rgb1, rgb2, rgb1]  # 第3个是重复的
    depth_list = [depth1, depth2, depth1]
    results = processor.batch_process(rgb_list, depth_list)
    
    # 显示缓存信息
    print("\n缓存信息:")
    cache_info = processor.get_cache_info()
    for key, value in cache_info.items():
        print(f"  {key}: {value}")
    
    # 询问是否清空缓存
    response = input("\n是否清空缓存？(y/n): ")
    if response.lower() == 'y':
        processor.clear_cache()


def process_video_with_cache(video_path: str, output_dir: str = "cdm_output"):
    """
    处理视频文件中的深度，使用缓存加速
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
    """
    print("="*60)
    print("视频深度增强处理（带缓存）")
    print("="*60)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建CDM处理器
    processor = CDMProcessorWithCache(
        enable_cache=True,
        cache_dir=".cache/cdm_video"
    )
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"视频信息:")
    print(f"  总帧数: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    
    frame_count = 0
    cache_hits = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # 每30帧处理一次
            print(f"\n处理帧 {frame_count}/{total_frames}")
            
            # 模拟深度（实际应用中应该从深度相机或估计获得）
            depth = np.random.uniform(0.2, 1.5, (frame.shape[0], frame.shape[1])).astype(np.float32)
            
            # 处理深度
            enhanced = processor.process_depth(frame, depth)
            
            # 保存结果
            output_file = output_path / f"enhanced_depth_{frame_count:06d}.npy"
            np.save(output_file, enhanced)
    
    cap.release()
    
    # 显示统计
    print(f"\n处理完成!")
    cache_info = processor.get_cache_info()
    print(f"缓存统计:")
    print(f"  总处理帧数: {frame_count // 30}")
    print(f"  缓存命中次数: {cache_info.get('total_accesses', 0) - (frame_count // 30)}")
    print(f"  缓存大小: {cache_info.get('total_size_mb', 0):.2f} MB")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CDM缓存处理器")
    parser.add_argument("--demo", action="store_true", help="运行演示")
    parser.add_argument("--rgb", type=str, help="RGB图像路径")
    parser.add_argument("--depth", type=str, help="深度图像路径")
    parser.add_argument("--output", type=str, help="输出路径")
    parser.add_argument("--video", type=str, help="视频文件路径")
    parser.add_argument("--clear-cache", action="store_true", help="清空缓存")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_cached_cdm()
    elif args.rgb and args.depth:
        # 处理单个RGB-D对
        processor = CDMProcessorWithCache(enable_cache=not args.no_cache)
        enhanced = processor.process_rgbd_file(args.rgb, args.depth, args.output)
        print("✓ 处理完成")
    elif args.video:
        # 处理视频
        process_video_with_cache(args.video)
    elif args.clear_cache:
        # 清空缓存
        processor = CDMProcessorWithCache()
        processor.clear_cache()
    else:
        # 默认运行演示
        demo_cached_cdm()