#!/usr/bin/env python3
"""
DA2相对深度到绝对深度的转换方法
"""

import numpy as np
import cv2
import os

def convert_relative_to_absolute_depth(relative_depth, method='linear', **params):
    """
    将DA2的相对深度转换为绝对深度
    
    Args:
        relative_depth: DA2输出的归一化相对深度 (0-1)
        method: 转换方法
        params: 方法相关的参数
    
    Returns:
        absolute_depth: 估计的绝对深度值（米）
    """
    
    if method == 'linear':
        # 方法1：线性映射（最简单，但可能不准确）
        # 需要提供场景的最小和最大深度
        min_depth = params.get('min_depth', 0.25)  # 默认25cm
        max_depth = params.get('max_depth', 0.8)   # 默认80cm
        
        absolute_depth = relative_depth * (max_depth - min_depth) + min_depth
        return absolute_depth
    
    elif method == 'scale_shift':
        # 方法2：Scale-Shift对齐（需要参考深度）
        # 基于最小二乘法对齐到参考深度
        reference_depth = params.get('reference_depth')
        if reference_depth is None:
            raise ValueError("需要提供reference_depth参数")
        
        # 只使用有效深度点进行对齐
        valid_mask = (reference_depth > 0) & (relative_depth > 0)
        if not np.any(valid_mask):
            return relative_depth  # 无法对齐，返回原值
        
        # 解决: absolute_depth = scale * relative_depth + shift
        # 使用最小二乘法
        rel_valid = relative_depth[valid_mask].flatten()
        ref_valid = reference_depth[valid_mask].flatten()
        
        # 构建方程 Ax = b, 其中 x = [scale, shift]
        A = np.vstack([rel_valid, np.ones(len(rel_valid))]).T
        scale, shift = np.linalg.lstsq(A, ref_valid, rcond=None)[0]
        
        absolute_depth = scale * relative_depth + shift
        return absolute_depth
    
    elif method == 'median_scaling':
        # 方法3：中值缩放（需要参考深度）
        # 基于中值比例进行缩放
        reference_depth = params.get('reference_depth')
        if reference_depth is None:
            raise ValueError("需要提供reference_depth参数")
        
        valid_mask = (reference_depth > 0) & (relative_depth > 0)
        if not np.any(valid_mask):
            return relative_depth
        
        # 计算中值比例
        rel_median = np.median(relative_depth[valid_mask])
        ref_median = np.median(reference_depth[valid_mask])
        scale = ref_median / (rel_median + 1e-8)
        
        absolute_depth = relative_depth * scale
        return absolute_depth
    
    elif method == 'histogram_matching':
        # 方法4：直方图匹配（需要参考深度）
        # 匹配深度分布
        reference_depth = params.get('reference_depth')
        if reference_depth is None:
            raise ValueError("需要提供reference_depth参数")
        
        # 归一化到0-255进行直方图匹配
        rel_uint8 = (relative_depth * 255).astype(np.uint8)
        ref_valid = reference_depth[reference_depth > 0]
        ref_normalized = (ref_valid - ref_valid.min()) / (ref_valid.max() - ref_valid.min())
        ref_uint8 = (ref_normalized * 255).astype(np.uint8)
        
        # 计算累积分布函数
        hist_rel, _ = np.histogram(rel_uint8.flatten(), 256, [0, 256])
        hist_ref, _ = np.histogram(ref_uint8.flatten(), 256, [0, 256])
        cdf_rel = hist_rel.cumsum()
        cdf_ref = hist_ref.cumsum()
        
        # 归一化CDF
        cdf_rel = cdf_rel / cdf_rel[-1]
        cdf_ref = cdf_ref / cdf_ref[-1]
        
        # 构建映射表
        mapping = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            idx = np.argmin(np.abs(cdf_ref - cdf_rel[i]))
            mapping[i] = idx
        
        # 应用映射
        matched = mapping[rel_uint8]
        
        # 转换回深度范围
        matched_norm = matched.astype(np.float32) / 255.0
        absolute_depth = matched_norm * (ref_valid.max() - ref_valid.min()) + ref_valid.min()
        
        return absolute_depth
    
    elif method == 'known_object':
        # 方法5：已知物体尺寸（需要物体检测和尺寸信息）
        # 例如：已知桌面高度或某个物体的实际尺寸
        object_real_size = params.get('object_real_size', 0.5)  # 实际尺寸（米）
        object_pixel_size = params.get('object_pixel_size', 100)  # 像素尺寸
        focal_length = params.get('focal_length', 600)  # 相机焦距（像素）
        
        # 基于物体计算缩放因子
        estimated_distance = (object_real_size * focal_length) / object_pixel_size
        
        # 假设物体在相对深度的中值位置
        rel_median = np.median(relative_depth[relative_depth > 0])
        scale = estimated_distance / rel_median
        
        absolute_depth = relative_depth * scale
        return absolute_depth
    
    else:
        raise ValueError(f"Unknown method: {method}")


def demonstrate_conversion():
    """演示不同的转换方法"""
    
    # 加载DA2相对深度
    da2_path = "data/bag/0912_topdownbags/images/da2/da2_depth_000000.npy"
    if not os.path.exists(da2_path):
        print(f"文件不存在: {da2_path}")
        return
    
    # 加载真实深度作为参考（如果有）
    real_depth_path = "data/bag/0912_topdownbags/images/depth/depth_frame_000000.npy"
    
    da2_relative = np.load(da2_path)
    
    print("DA2相对深度转换演示\n" + "="*50)
    print(f"DA2相对深度范围: {da2_relative.min():.3f} - {da2_relative.max():.3f}")
    
    # 方法1：线性映射
    print("\n1. 线性映射方法（假设场景深度0.25-0.8m）:")
    depth_linear = convert_relative_to_absolute_depth(
        da2_relative, 
        method='linear',
        min_depth=0.25,
        max_depth=0.8
    )
    print(f"   转换后范围: {depth_linear.min():.3f} - {depth_linear.max():.3f}m")
    
    # 如果有真实深度，使用更精确的方法
    if os.path.exists(real_depth_path):
        real_depth = np.load(real_depth_path)
        print(f"\n发现真实深度数据，可使用更精确的对齐方法:")
        print(f"真实深度范围: {real_depth[real_depth>0].min():.3f} - {real_depth[real_depth>0].max():.3f}m")
        
        # 方法2：Scale-Shift对齐
        print("\n2. Scale-Shift对齐方法:")
        depth_scale_shift = convert_relative_to_absolute_depth(
            da2_relative,
            method='scale_shift',
            reference_depth=real_depth
        )
        print(f"   转换后范围: {depth_scale_shift.min():.3f} - {depth_scale_shift.max():.3f}m")
        
        # 方法3：中值缩放
        print("\n3. 中值缩放方法:")
        depth_median = convert_relative_to_absolute_depth(
            da2_relative,
            method='median_scaling',
            reference_depth=real_depth
        )
        print(f"   转换后范围: {depth_median.min():.3f} - {depth_median.max():.3f}m")
        
        # 计算误差（如果有真实深度）
        valid_mask = real_depth > 0
        if np.any(valid_mask):
            print("\n误差分析（与真实深度对比）:")
            methods = {
                '线性映射': depth_linear,
                'Scale-Shift': depth_scale_shift,
                '中值缩放': depth_median
            }
            
            for name, depth in methods.items():
                mae = np.mean(np.abs(depth[valid_mask] - real_depth[valid_mask]))
                rmse = np.sqrt(np.mean((depth[valid_mask] - real_depth[valid_mask])**2))
                print(f"   {name:12s}: MAE={mae:.4f}m, RMSE={rmse:.4f}m")
    
    # 可视化对比
    print("\n生成可视化对比图...")
    create_comparison_visualization(da2_relative, depth_linear, real_depth if 'real_depth' in locals() else None)


def create_comparison_visualization(relative_depth, absolute_depth, reference_depth=None):
    """创建对比可视化"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3 if reference_depth is not None else 2, figsize=(15, 5))
    
    # 相对深度
    im1 = axes[0].imshow(relative_depth, cmap='turbo')
    axes[0].set_title('DA2 Relative Depth (0-1)')
    plt.colorbar(im1, ax=axes[0])
    
    # 转换后的绝对深度
    im2 = axes[1].imshow(absolute_depth, cmap='turbo')
    axes[1].set_title('Converted Absolute Depth (m)')
    plt.colorbar(im2, ax=axes[1])
    
    # 真实深度（如果有）
    if reference_depth is not None:
        im3 = axes[2].imshow(reference_depth, cmap='turbo')
        axes[2].set_title('Reference Depth (m)')
        plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('da2_depth_conversion_comparison.png', dpi=150)
    print(f"可视化已保存为: da2_depth_conversion_comparison.png")
    plt.show()


if __name__ == "__main__":
    demonstrate_conversion()