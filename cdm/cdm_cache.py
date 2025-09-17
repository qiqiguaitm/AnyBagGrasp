#!/usr/bin/env python3
"""
CDM (Camera Depth Model) 缓存功能
用于缓存CDM深度增强结果，避免重复计算
"""

import torch
import numpy as np
import hashlib
import pickle
import os
from pathlib import Path
import time
from typing import Optional, Tuple, Dict, Any
import json

class CDMCache:
    """CDM深度增强结果缓存管理器"""
    
    def __init__(self, cache_dir: str = ".cache/cdm_results"):
        """
        初始化CDM缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 缓存索引文件
        self.index_file = self.cache_dir / "cache_index.json"
        self.load_index()
        
    def load_index(self):
        """加载缓存索引"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            except:
                self.index = {}
        else:
            self.index = {}
    
    def save_index(self):
        """保存缓存索引"""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _generate_cache_key(self, rgb: np.ndarray, depth: np.ndarray, **kwargs) -> str:
        """
        生成缓存键
        
        Args:
            rgb: RGB图像数组
            depth: 深度图像数组
            **kwargs: 其他参数（如模型参数等）
            
        Returns:
            str: 缓存键（哈希值）
        """
        hasher = hashlib.sha256()
        
        # RGB数据哈希
        hasher.update(rgb.tobytes())
        hasher.update(str(rgb.shape).encode())
        hasher.update(str(rgb.dtype).encode())
        
        # 深度数据哈希
        hasher.update(depth.tobytes())
        hasher.update(str(depth.shape).encode())
        hasher.update(str(depth.dtype).encode())
        
        # 添加额外参数
        params = {
            'model': kwargs.get('model_name', 'cdm_d435'),
            'target_size': kwargs.get('target_size', 518),
            'version': kwargs.get('version', '1.0')
        }
        params_str = json.dumps(params, sort_keys=True)
        hasher.update(params_str.encode())
        
        return hasher.hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """
        获取缓存文件路径
        
        Args:
            cache_key: 缓存键
            
        Returns:
            Path: 缓存文件路径
        """
        # 使用前两个字符作为子目录
        subdir = cache_key[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{cache_key}.npz"
    
    def get(self, rgb: np.ndarray, depth: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        """
        获取缓存的CDM增强深度结果
        
        Args:
            rgb: RGB图像数组
            depth: 深度图像数组
            **kwargs: 其他参数
            
        Returns:
            Optional[np.ndarray]: 缓存的增强深度结果，如果不存在则返回None
        """
        cache_key = self._generate_cache_key(rgb, depth, **kwargs)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                # 加载缓存数据
                data = np.load(cache_path)
                enhanced_depth = data['enhanced_depth']
                metadata = data['metadata'].item()  # .item() 将0维数组转为字典
                
                # 检查缓存是否过期（默认7天）
                cache_age = time.time() - metadata.get('timestamp', 0)
                max_age = kwargs.get('max_cache_age', 7 * 24 * 3600)
                
                if cache_age < max_age:
                    print(f"✓ 使用CDM缓存结果 (缓存键: {cache_key[:8]}...)")
                    
                    # 更新索引中的访问时间
                    if cache_key in self.index:
                        self.index[cache_key]['last_accessed'] = time.time()
                        self.index[cache_key]['access_count'] = self.index[cache_key].get('access_count', 0) + 1
                        self.save_index()
                    
                    return enhanced_depth
                else:
                    print(f"⚠ CDM缓存已过期，重新计算")
                    cache_path.unlink()
                    if cache_key in self.index:
                        del self.index[cache_key]
                        self.save_index()
                    
            except Exception as e:
                print(f"⚠ 读取CDM缓存失败: {e}")
                return None
        
        return None
    
    def set(self, rgb: np.ndarray, depth: np.ndarray, enhanced_depth: np.ndarray, **kwargs) -> None:
        """
        保存CDM增强深度结果到缓存
        
        Args:
            rgb: RGB图像数组
            depth: 原始深度图像数组
            enhanced_depth: CDM增强后的深度图像数组
            **kwargs: 其他参数
        """
        cache_key = self._generate_cache_key(rgb, depth, **kwargs)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # 准备元数据
            metadata = {
                'timestamp': time.time(),
                'rgb_shape': rgb.shape,
                'depth_shape': depth.shape,
                'enhanced_shape': enhanced_depth.shape,
                'model_name': kwargs.get('model_name', 'cdm_d435'),
                'target_size': kwargs.get('target_size', 518),
            }
            
            # 保存到npz文件（压缩格式）
            np.savez_compressed(
                cache_path,
                enhanced_depth=enhanced_depth,
                metadata=metadata
            )
            
            # 更新索引
            self.index[cache_key] = {
                'path': str(cache_path),
                'created': time.time(),
                'last_accessed': time.time(),
                'access_count': 1,
                'size_bytes': cache_path.stat().st_size,
                'rgb_shape': list(rgb.shape),
                'depth_shape': list(depth.shape)
            }
            self.save_index()
            
            print(f"✓ CDM结果已缓存 (缓存键: {cache_key[:8]}...)")
            
        except Exception as e:
            print(f"⚠ 保存CDM缓存失败: {e}")
    
    def clear(self) -> None:
        """清空所有缓存"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.index = {}
            self.save_index()
            print(f"✓ 已清空CDM缓存目录: {self.cache_dir}")
    
    def get_cache_size(self) -> int:
        """
        获取缓存大小（字节）
        
        Returns:
            int: 缓存总大小（字节）
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size
    
    def get_cache_info(self) -> Dict:
        """
        获取缓存信息
        
        Returns:
            Dict: 缓存统计信息
        """
        cache_files = list(self.cache_dir.glob("**/*.npz"))
        total_size = self.get_cache_size()
        
        # 计算访问统计
        total_accesses = sum(info.get('access_count', 0) for info in self.index.values())
        
        return {
            'cache_dir': str(self.cache_dir),
            'total_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'total_accesses': total_accesses,
            'index_entries': len(self.index),
            'oldest_cache': min([info['created'] for info in self.index.values()]) if self.index else None,
            'newest_cache': max([info['created'] for info in self.index.values()]) if self.index else None
        }
    
    def cleanup_old_caches(self, max_age_days: int = 7, max_size_mb: float = 1000):
        """
        清理旧的缓存文件
        
        Args:
            max_age_days: 最大缓存天数
            max_size_mb: 最大缓存大小（MB）
        """
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        max_size_bytes = max_size_mb * 1024 * 1024
        
        # 按访问时间排序
        sorted_caches = sorted(
            self.index.items(),
            key=lambda x: x[1].get('last_accessed', 0)
        )
        
        removed_count = 0
        current_size = self.get_cache_size()
        
        for cache_key, info in sorted_caches:
            # 检查是否过期
            age = current_time - info.get('created', 0)
            if age > max_age_seconds or current_size > max_size_bytes:
                cache_path = Path(info['path'])
                if cache_path.exists():
                    file_size = cache_path.stat().st_size
                    cache_path.unlink()
                    current_size -= file_size
                    removed_count += 1
                del self.index[cache_key]
        
        if removed_count > 0:
            self.save_index()
            print(f"✓ 清理了 {removed_count} 个旧缓存文件")


class CachedCDMProcessor:
    """带缓存功能的CDM处理器"""
    
    def __init__(self, model, cache_dir: str = ".cache/cdm_results", enable_cache: bool = True):
        """
        初始化带缓存的CDM处理器
        
        Args:
            model: CDM模型实例
            cache_dir: 缓存目录
            enable_cache: 是否启用缓存
        """
        self.model = model
        self.cache = CDMCache(cache_dir) if enable_cache else None
        self.enable_cache = enable_cache
        
    def process(self, rgb: np.ndarray, depth: np.ndarray, 
                target_size: int = 518, **kwargs) -> np.ndarray:
        """
        处理深度图像（带缓存）
        
        Args:
            rgb: RGB图像数组
            depth: 深度图像数组
            target_size: 模型输入尺寸
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 增强后的深度图像
        """
        # 如果启用缓存，先尝试从缓存获取
        if self.enable_cache and self.cache:
            cached_result = self.cache.get(rgb, depth, target_size=target_size, **kwargs)
            if cached_result is not None:
                return cached_result
        
        # 缓存未命中，执行实际处理
        print("○ 执行CDM深度增强...")
        
        # 准备输入张量
        rgb_tensor = torch.from_numpy(rgb.copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # 创建相似度深度图
        simi_depth = np.zeros_like(depth)
        valid_mask = depth > 0
        simi_depth[valid_mask] = 1.0 / depth[valid_mask]
        depth_tensor = torch.from_numpy(simi_depth).float().unsqueeze(0).unsqueeze(0)
        
        # 处理
        with torch.no_grad():
            # 调整尺寸
            rgb_resized = torch.nn.functional.interpolate(
                rgb_tensor, size=(target_size, target_size), 
                mode='bilinear', align_corners=False
            )
            depth_resized = torch.nn.functional.interpolate(
                depth_tensor, size=(target_size, target_size), 
                mode='bilinear', align_corners=False
            )
            
            # 合并输入
            rgbd_input = torch.cat([rgb_resized, depth_resized], dim=1)
            
            # 模型推理
            enhanced_depth = self.model(rgbd_input)
            
            # 添加通道维度（如果需要）
            if len(enhanced_depth.shape) == 3:
                enhanced_depth = enhanced_depth.unsqueeze(1)
            
            # 调整回原始尺寸
            enhanced_depth = torch.nn.functional.interpolate(
                enhanced_depth, size=(depth.shape[0], depth.shape[1]), 
                mode='bilinear', align_corners=False
            )
            
            # 转换回numpy
            enhanced_np = enhanced_depth.squeeze().cpu().numpy()
            
            # 从相似度转换回深度
            enhanced_meters = np.where(enhanced_np > 0, 1.0 / enhanced_np, 0)
        
        # 保存到缓存
        if self.enable_cache and self.cache:
            self.cache.set(rgb, depth, enhanced_meters, target_size=target_size, **kwargs)
        
        return enhanced_meters
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache:
            self.cache.clear()
    
    def get_cache_info(self):
        """获取缓存信息"""
        if self.cache:
            return self.cache.get_cache_info()
        return {"message": "缓存未启用"}


def test_cdm_cache():
    """测试CDM缓存功能"""
    print("="*60)
    print("测试CDM缓存功能")
    print("="*60)
    
    # 创建缓存管理器
    cache = CDMCache(".cache/cdm_test")
    
    # 创建测试数据
    rgb_test = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    depth_test = np.random.uniform(0.2, 1.5, (720, 1280)).astype(np.float32)
    enhanced_test = depth_test * 1.1  # 模拟增强后的深度
    
    print("\n1. 测试缓存保存...")
    cache.set(rgb_test, depth_test, enhanced_test, model_name="test_model")
    
    print("\n2. 测试缓存读取...")
    cached = cache.get(rgb_test, depth_test, model_name="test_model")
    if cached is not None:
        print(f"   ✓ 成功读取缓存，shape: {cached.shape}")
        print(f"   ✓ 数据匹配: {np.allclose(cached, enhanced_test)}")
    else:
        print("   ✗ 缓存读取失败")
    
    print("\n3. 缓存信息...")
    info = cache.get_cache_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("\n4. 测试缓存清理...")
    cache.cleanup_old_caches(max_age_days=0.00001)  # 立即过期
    
    print("\n5. 清空缓存...")
    cache.clear()
    
    print("\n✓ CDM缓存测试完成")


if __name__ == "__main__":
    test_cdm_cache()