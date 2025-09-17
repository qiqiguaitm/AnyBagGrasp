import json
import hashlib
import os
from pathlib import Path
import pickle
import numpy as np
from typing import Any, Dict, Optional
import time

class DetectionCache:
    """检测结果缓存管理器"""
    
    def __init__(self, cache_dir: str = ".cache/detection_results"):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _generate_cache_key(self, image_data: Any, params: Dict) -> str:
        """
        生成缓存键
        
        Args:
            image_data: 图像数据（numpy array或路径）
            params: API参数
            
        Returns:
            str: 缓存键（哈希值）
        """
        # 创建哈希对象
        hasher = hashlib.sha256()
        
        # 处理图像数据
        if isinstance(image_data, str):
            # 如果是文件路径，使用文件内容和修改时间
            if os.path.exists(image_data):
                mtime = os.path.getmtime(image_data)
                hasher.update(f"file:{image_data}:{mtime}".encode())
            else:
                hasher.update(f"file:{image_data}".encode())
        elif isinstance(image_data, np.ndarray):
            # 如果是numpy数组，使用数组的哈希
            hasher.update(image_data.tobytes())
            hasher.update(str(image_data.shape).encode())
            hasher.update(str(image_data.dtype).encode())
        else:
            # 其他类型转换为字符串
            hasher.update(str(image_data).encode())
        
        # 添加参数到哈希
        # 排序参数以确保一致性
        sorted_params = sorted(params.items())
        params_str = json.dumps(sorted_params, sort_keys=True)
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
        # 使用前两个字符作为子目录，避免单个目录文件过多
        subdir = cache_key[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{cache_key}.pkl"
    
    def get(self, image_data: Any, params: Dict) -> Optional[Dict]:
        """
        获取缓存的检测结果
        
        Args:
            image_data: 图像数据
            params: API参数
            
        Returns:
            Optional[Dict]: 缓存的结果，如果不存在则返回None
        """
        cache_key = self._generate_cache_key(image_data, params)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # 检查缓存是否过期（可选，这里设置为7天）
                cache_age = time.time() - cached_data.get('timestamp', 0)
                max_age = 7 * 24 * 3600  # 7天
                
                if cache_age < max_age:
                    print(f"✓ 使用缓存结果 (缓存键: {cache_key[:8]}...)")
                    return cached_data.get('result')
                else:
                    print(f"⚠ 缓存已过期，重新调用API")
                    cache_path.unlink()  # 删除过期缓存
                    
            except Exception as e:
                print(f"⚠ 读取缓存失败: {e}")
                return None
        
        return None
    
    def set(self, image_data: Any, params: Dict, result: Dict) -> None:
        """
        保存检测结果到缓存
        
        Args:
            image_data: 图像数据
            params: API参数
            result: 检测结果
        """
        cache_key = self._generate_cache_key(image_data, params)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cached_data = {
                'result': result,
                'timestamp': time.time(),
                'params': params
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            
            print(f"✓ 检测结果已缓存 (缓存键: {cache_key[:8]}...)")
            
        except Exception as e:
            print(f"⚠ 保存缓存失败: {e}")
    
    def clear(self) -> None:
        """清空所有缓存"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ 已清空缓存目录: {self.cache_dir}")
    
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
        cache_files = list(self.cache_dir.glob("**/*.pkl"))
        total_size = self.get_cache_size()
        
        return {
            'cache_dir': str(self.cache_dir),
            'total_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'oldest_cache': min([f.stat().st_mtime for f in cache_files]) if cache_files else None,
            'newest_cache': max([f.stat().st_mtime for f in cache_files]) if cache_files else None
        }


class CachedDetectionAPI:
    """带缓存功能的检测API包装器"""
    
    def __init__(self, api_instance, cache_dir: str = ".cache/detection_results", enable_cache: bool = True):
        """
        初始化带缓存的API包装器
        
        Args:
            api_instance: 原始API实例（ReferAPI, DetectionAPI或GraspAnythingAPI）
            cache_dir: 缓存目录
            enable_cache: 是否启用缓存
        """
        self.api = api_instance
        self.cache = DetectionCache(cache_dir) if enable_cache else None
        self.enable_cache = enable_cache
        
    def forward(self, *args, **kwargs):
        """
        带缓存的forward方法
        """
        if not self.enable_cache:
            return self.api.forward(*args, **kwargs)
        
        # 提取图像数据和参数
        if 'rgb' in kwargs:
            image_data = kwargs['rgb']
        elif len(args) > 0:
            # 对于ReferAPI，第一个参数是text，第二个是rgb
            if hasattr(self.api, '__class__') and self.api.__class__.__name__ == 'ReferAPI':
                if len(args) >= 2:
                    image_data = args[1]
                    kwargs['text'] = args[0]
                else:
                    image_data = kwargs.get('rgb')
            else:
                image_data = args[0]
        else:
            image_data = None
            
        if image_data is None:
            # 无法确定图像数据，直接调用API
            return self.api.forward(*args, **kwargs)
        
        # 构建缓存参数
        cache_params = {
            'api_class': self.api.__class__.__name__,
            'model_name': getattr(self.api, 'model_name', None),
        }
        
        # 添加API特定参数
        if hasattr(self.api, '__class__'):
            if self.api.__class__.__name__ == 'ReferAPI':
                cache_params['text'] = kwargs.get('text', args[0] if args else '')
            elif self.api.__class__.__name__ == 'DetectionAPI':
                cache_params['prompt_text'] = kwargs.get('prompt_text', '')
                cache_params['bbox_threshold'] = kwargs.get('bbox_threshold', 0.25)
                cache_params['iou_threshold'] = kwargs.get('iou_threshold', 0.8)
            elif self.api.__class__.__name__ == 'GraspAnythingAPI':
                cache_params['bag'] = kwargs.get('bag', False)
                cache_params['use_touching_points'] = kwargs.get('use_touching_points', True)
        
        # 尝试从缓存获取结果
        cached_result = self.cache.get(image_data, cache_params)
        if cached_result is not None:
            # 如果是任务对象，需要重新构建
            if hasattr(self.api, '__class__') and self.api.__class__.__name__ in ['ReferAPI', 'DetectionAPI']:
                # 创建一个假的任务对象，只包含result属性
                class CachedTask:
                    def __init__(self, result):
                        self.result = result
                return CachedTask(cached_result)
            else:
                return cached_result
        
        # 缓存未命中，调用实际API
        print(f"○ 调用API进行检测...")
        result = self.api.forward(*args, **kwargs)
        
        # 保存到缓存
        if hasattr(result, 'result'):
            # 对于返回任务对象的API
            self.cache.set(image_data, cache_params, result.result)
        elif isinstance(result, tuple) and len(result) == 2:
            # 对于GraspAnythingAPI返回的元组
            self.cache.set(image_data, cache_params, result)
        else:
            self.cache.set(image_data, cache_params, result)
        
        return result
    
    def detect_objects(self, *args, **kwargs):
        """
        带缓存的detect_objects方法（仅DetectionAPI）
        """
        if hasattr(self.api, 'detect_objects'):
            if not self.enable_cache:
                return self.api.detect_objects(*args, **kwargs)
            
            # 调用forward方法，它会处理缓存
            task = self.forward(*args, **kwargs)
            return task.result if hasattr(task, 'result') else {}
        else:
            raise AttributeError(f"{self.api.__class__.__name__} 没有 detect_objects 方法")
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache:
            self.cache.clear()
    
    def get_cache_info(self):
        """获取缓存信息"""
        if self.cache:
            return self.cache.get_cache_info()
        return {"message": "缓存未启用"}
    
    def __getattr__(self, name):
        """代理其他方法到原始API"""
        return getattr(self.api, name)