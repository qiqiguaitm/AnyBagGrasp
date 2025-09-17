#!/usr/bin/env python3
"""
基于muggled_dpt的解释，实现DA2/DPT模型输出的正确后处理
参考: https://github.com/heyoeyo/muggled_dpt/blob/main/.readme_assets/results_explainer.md
"""

import numpy as np
import cv2
from typing import Optional, Tuple
import matplotlib.pyplot as plt


def postprocess_inverse_depth(
    inverse_depth: np.ndarray,
    method: str = 'normalization',
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
    percentile: Tuple[float, float] = (2, 98)
) -> np.ndarray:
    """
    对DPT/DA2等模型输出的inverse depth进行后处理
    
    Args:
        inverse_depth: 模型原始输出（inverse depth，值越大越近）
        method: 后处理方法
            - 'normalization': 简单归一化到[0,1]
            - 'reciprocal': 直接取倒数转为深度
            - 'robust_reciprocal': 鲁棒的倒数转换
            - 'metric_scaling': 转换到米制深度（需要depth_min和depth_max）
        depth_min: 场景最小深度（米），用于metric_scaling
        depth_max: 场景最大深度（米），用于metric_scaling
        percentile: 用于鲁棒处理的百分位数
    
    Returns:
        处理后的深度图（大值=远，小值=近）
    """
    
    if method == 'normalization':
        # 方法1：简单归一化（保持相对关系）
        # 由于是inverse depth，需要先反转
        depth = 1.0 / (inverse_depth + 1e-8)
        
        # 归一化到[0,1]
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth_norm
    
    elif method == 'reciprocal':
        # 方法2：直接倒数（简单但可能有噪声）
        depth = 1.0 / (inverse_depth + 1e-8)
        return depth
    
    elif method == 'robust_reciprocal':
        # 方法3：鲁棒的倒数转换（处理异常值）
        # 使用百分位数裁剪异常值
        p_low, p_high = percentile
        inv_low = np.percentile(inverse_depth, p_low)
        inv_high = np.percentile(inverse_depth, p_high)
        
        # 裁剪到合理范围
        inverse_depth_clipped = np.clip(inverse_depth, inv_low, inv_high)
        
        # 转换为深度
        depth = 1.0 / (inverse_depth_clipped + 1e-8)
        
        # 处理无效区域（如天空）
        # 将极小的inverse depth值（远处/天空）设为最大深度
        sky_mask = inverse_depth < inv_low * 0.1
        if np.any(sky_mask):
            depth[sky_mask] = depth[~sky_mask].max() * 1.5
        
        return depth
    
    elif method == 'metric_scaling':
        # 方法4：转换到米制深度（基于muggled_dpt的公式）
        if depth_min is None or depth_max is None:
            raise ValueError("metric_scaling需要提供depth_min和depth_max参数")
        
        # 首先归一化inverse depth到[0,1]
        inv_norm = (inverse_depth - inverse_depth.min()) / (inverse_depth.max() - inverse_depth.min() + 1e-8)
        
        # 应用转换公式：
        # True Depth = [V_norm(1/d_min - 1/d_max) + 1/d_max]^-1
        # 其中V_norm是归一化的inverse depth
        
        # 计算inverse depth的范围
        inv_d_min = 1.0 / depth_min  # 近处的inverse depth（大值）
        inv_d_max = 1.0 / depth_max  # 远处的inverse depth（小值）
        
        # 映射到实际的inverse depth范围
        # 注意：inv_norm中0对应远处，1对应近处（因为是inverse depth）
        scaled_inv = inv_norm * (inv_d_min - inv_d_max) + inv_d_max
        
        # 转换回深度
        depth = 1.0 / (scaled_inv + 1e-8)
        
        # 裁剪到合理范围
        depth = np.clip(depth, depth_min, depth_max)
        
        return depth
    
    else:
        raise ValueError(f"未知的后处理方法: {method}")


def analyze_depth_output(raw_output: np.ndarray, reference_depth: Optional[np.ndarray] = None):
    """
    分析模型输出特性，确定最佳后处理方法
    
    Args:
        raw_output: 模型原始输出
        reference_depth: 参考深度（如果有）
    
    Returns:
        分析结果字典
    """
    
    analysis = {}
    
    # 基本统计
    analysis['range'] = (raw_output.min(), raw_output.max())
    analysis['mean'] = raw_output.mean()
    analysis['std'] = raw_output.std()
    
    # 检查是否是inverse depth（通过值的分布）
    # Inverse depth通常有更大的动态范围
    dynamic_range = raw_output.max() / (raw_output.min() + 1e-8)
    analysis['dynamic_range'] = dynamic_range
    analysis['likely_inverse'] = dynamic_range > 10  # 经验阈值
    
    if reference_depth is not None:
        valid_mask = reference_depth > 0
        if np.any(valid_mask):
            ref_valid = reference_depth[valid_mask].flatten()
            raw_valid = raw_output[valid_mask].flatten()
            
            # 测试不同转换的相关性
            corr_direct = np.corrcoef(ref_valid, raw_valid)[0, 1]
            
            # 测试倒数转换
            depth_reciprocal = 1.0 / (raw_valid + 1e-8)
            corr_reciprocal = np.corrcoef(ref_valid, depth_reciprocal)[0, 1]
            
            analysis['correlation_direct'] = corr_direct
            analysis['correlation_reciprocal'] = corr_reciprocal
            
            # 推荐方法
            if abs(corr_reciprocal) > abs(corr_direct):
                analysis['recommended_method'] = 'reciprocal'
                analysis['is_inverse_depth'] = True
            else:
                analysis['recommended_method'] = 'direct'
                analysis['is_inverse_depth'] = False
    
    return analysis


def visualize_postprocessing_comparison(
    raw_output: np.ndarray,
    reference_depth: Optional[np.ndarray] = None,
    depth_min: float = 0.3,
    depth_max: float = 10.0
):
    """
    可视化不同后处理方法的效果
    """
    
    methods = {
        'Raw Output': raw_output,
        'Normalization': postprocess_inverse_depth(raw_output, 'normalization'),
        'Reciprocal': postprocess_inverse_depth(raw_output, 'reciprocal'),
        'Robust Reciprocal': postprocess_inverse_depth(raw_output, 'robust_reciprocal'),
        'Metric Scaling': postprocess_inverse_depth(
            raw_output, 'metric_scaling', 
            depth_min=depth_min, depth_max=depth_max
        )
    }
    
    n_methods = len(methods)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, depth_map) in enumerate(methods.items()):
        ax = axes[idx]
        im = ax.imshow(depth_map, cmap='turbo')
        ax.set_title(f'{name}\nRange: [{depth_map.min():.2f}, {depth_map.max():.2f}]')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis('off')
    
    # 如果有参考深度，显示在最后一个位置
    if reference_depth is not None:
        ax = axes[-1]
        valid_mask = reference_depth > 0
        depth_show = reference_depth.copy()
        depth_show[~valid_mask] = 0
        im = ax.imshow(depth_show, cmap='turbo')
        ax.set_title(f'Reference Depth\nRange: [{depth_show[valid_mask].min():.2f}, {depth_show[valid_mask].max():.2f}]')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis('off')
    else:
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('da2_postprocessing_comparison.png', dpi=150, bbox_inches='tight')
    print("可视化已保存为: da2_postprocessing_comparison.png")
    plt.show()


def demo_postprocessing():
    """演示DA2输出的后处理"""
    
    print("=" * 70)
    print("DA2/DPT Inverse Depth 后处理演示")
    print("=" * 70)
    
    # 加载DA2输出和参考深度
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "dpt_any2"))
    
    # 加载数据
    rgb_path = "data/bag/0912_topdownbags/images/rgb/rgb_frame_000000.jpg"
    real_depth_path = "data/bag/0912_topdownbags/images/depth/depth_frame_000000.npy"
    
    if not Path(rgb_path).exists():
        print("错误：找不到测试数据")
        return
    
    # 获取DA2原始输出
    from da2_processor import DA2Processor
    
    rgb_img = cv2.imread(rgb_path)
    processor = DA2Processor(encoder='vitb')
    
    if processor.is_available():
        # 获取模型的原始输出（未经过我们之前的1/x处理）
        import torch
        with torch.no_grad():
            rgb_for_model = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            raw_output = processor.model.infer_image(rgb_for_model, 518)
            raw_output = cv2.resize(raw_output, (rgb_img.shape[1], rgb_img.shape[0]))
        
        print(f"\n原始输出统计:")
        print(f"  范围: {raw_output.min():.4f} - {raw_output.max():.4f}")
        print(f"  均值: {raw_output.mean():.4f}")
        print(f"  标准差: {raw_output.std():.4f}")
        
        # 分析输出特性
        reference_depth = None
        if Path(real_depth_path).exists():
            reference_depth = np.load(real_depth_path)
        
        print("\n分析模型输出特性...")
        analysis = analyze_depth_output(raw_output, reference_depth)
        
        print(f"\n分析结果:")
        print(f"  动态范围: {analysis['dynamic_range']:.2f}")
        print(f"  可能是inverse depth: {analysis['likely_inverse']}")
        
        if 'correlation_direct' in analysis:
            print(f"  直接相关性: {analysis['correlation_direct']:.4f}")
            print(f"  倒数相关性: {analysis['correlation_reciprocal']:.4f}")
            print(f"  推荐方法: {analysis['recommended_method']}")
            print(f"  确认是inverse depth: {analysis['is_inverse_depth']}")
        
        # 测试不同的后处理方法
        print("\n测试不同后处理方法...")
        
        methods_results = {}
        
        # 1. 归一化
        depth_norm = postprocess_inverse_depth(raw_output, 'normalization')
        methods_results['normalization'] = depth_norm
        
        # 2. 直接倒数
        depth_recip = postprocess_inverse_depth(raw_output, 'reciprocal')
        methods_results['reciprocal'] = depth_recip
        
        # 3. 鲁棒倒数
        depth_robust = postprocess_inverse_depth(raw_output, 'robust_reciprocal')
        methods_results['robust_reciprocal'] = depth_robust
        
        # 4. 米制缩放（假设场景深度0.25-0.8米）
        depth_metric = postprocess_inverse_depth(
            raw_output, 'metric_scaling',
            depth_min=0.25, depth_max=0.8
        )
        methods_results['metric_scaling'] = depth_metric
        
        # 评估每种方法
        if reference_depth is not None:
            valid_mask = reference_depth > 0
            ref_valid = reference_depth[valid_mask].flatten()
            
            print("\n后处理方法评估:")
            print("方法\t\t\t相关系数\tMAE")
            print("-" * 50)
            
            for method_name, depth_result in methods_results.items():
                depth_valid = depth_result[valid_mask].flatten()
                
                # 归一化到相同范围进行比较
                ref_norm = (ref_valid - ref_valid.min()) / (ref_valid.max() - ref_valid.min())
                depth_norm = (depth_valid - depth_valid.min()) / (depth_valid.max() - depth_valid.min())
                
                corr = np.corrcoef(ref_norm, depth_norm)[0, 1]
                mae = np.mean(np.abs(ref_norm - depth_norm))
                
                print(f"{method_name:20s}\t{corr:.4f}\t\t{mae:.4f}")
        
        # 可视化比较
        print("\n生成可视化比较...")
        visualize_postprocessing_comparison(
            raw_output, 
            reference_depth,
            depth_min=0.25,
            depth_max=0.8
        )
        
    print("\n" + "=" * 70)
    print("结论:")
    print("1. DA2/DPT输出的是inverse depth（视差相关）")
    print("2. 需要进行倒数转换以获得正确的深度方向")
    print("3. 'robust_reciprocal'方法通常效果最好")
    print("4. 对于米制深度，需要知道场景的深度范围")
    print("=" * 70)


if __name__ == "__main__":
    demo_postprocessing()