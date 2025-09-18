#!/usr/bin/env python3
"""
Enhanced Camera Depth Analysis Script
Extracts RGB and depth frames from video and HDF5 files with improved visualization
FPS=2 for full data parsing
"""

import cv2
import h5py
import numpy as np
import os
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import time

# Add CDM directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "cdm"))

try:
    from simple_cdm_fixed import process_depth_with_cdm
    CDM_AVAILABLE = True
    print("CDM processor available")
except ImportError as e:
    print(f"Warning: CDM processor not available: {e}")
    CDM_AVAILABLE = False
    process_depth_with_cdm = lambda rgb, depth: None

# Add DA2 directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dpt_any2"))

try:
    from da2_processor import process_rgb_with_da2
    DA2_AVAILABLE = True
    print("DA2 processor available")
except ImportError as e:
    print(f"Warning: DA2 processor not available: {e}")
    DA2_AVAILABLE = False
    process_rgb_with_da2 = lambda rgb: None

# 桌面抓取场景全局深度范围参数
MIN_DEPTH = 0.25  # 最近深度(米) - 桌面高度
MAX_DEPTH = 0.8   # 最远深度(米) - 桌面上物体的合理范围
NUM_CONTOUR_LEVELS = 8  # 等深度线的数量

def compute_depth_statistics(h5_path, sample_rate=0.1, percentile_min=5, percentile_max=95, inverse_depth=False):
    """
    计算深度数据的统计信息来动态确定MIN_DEPTH和MAX_DEPTH
    
    Args:
        h5_path: HDF5文件路径
        sample_rate: 采样率(0.0-1.0)，默认采样10%的帧
        percentile_min: 用于计算最小深度的百分位数
        percentile_max: 用于计算最大深度的百分位数
        inverse_depth: 如果为True，对深度值取倒数（用于视差数据）
    
    Returns:
        tuple: (computed_min_depth, computed_max_depth, stats_dict)
    """
    print(f"\n=== Computing depth statistics from {h5_path} ===")
    if inverse_depth:
        print("Note: Applying inverse transformation (1/depth) for disparity data")
    
    all_valid_depths = []
    
    with h5py.File(h5_path, 'r') as h5f:
        depth_data = h5f['frames']
        total_frames = depth_data.shape[0]
        
        # 计算采样帧数
        sample_frames = max(1, int(total_frames * sample_rate))
        # 均匀采样帧索引
        frame_indices = np.linspace(0, total_frames-1, sample_frames, dtype=int)
        
        print(f"Sampling {sample_frames} frames out of {total_frames} (rate: {sample_rate:.1%})")
        
        for idx in frame_indices:
            depth_frame = depth_data[idx]
            
            # 如果需要，应用倒数变换
            if inverse_depth:
                # 避免除以0
                valid_disparity = depth_frame > 0
                if np.any(valid_disparity):
                    depth_frame_converted = np.zeros_like(depth_frame)
                    depth_frame_converted[valid_disparity] = 1.0 / depth_frame[valid_disparity]
                    depth_frame = depth_frame_converted
            
            # 只考虑有效深度值（大于0）
            valid_mask = depth_frame > 0
            if np.any(valid_mask):
                valid_depths = depth_frame[valid_mask].flatten()
                # 初步过滤异常值（0-10米范围）
                reasonable_depths = valid_depths[(valid_depths > 0.1) & (valid_depths < 10.0)]
                if len(reasonable_depths) > 0:
                    all_valid_depths.extend(reasonable_depths)
            
            if len(frame_indices) > 10 and (idx - frame_indices[0]) % (len(frame_indices) // 10) == 0:
                print(f"  Processed {idx - frame_indices[0] + 1}/{sample_frames} sampled frames...")
    
    if len(all_valid_depths) == 0:
        print("Warning: No valid depth values found!")
        return MIN_DEPTH, MAX_DEPTH, {}
    
    # 转换为numpy数组进行统计
    all_valid_depths = np.array(all_valid_depths)
    
    # 计算统计信息
    stats = {
        'mean': np.mean(all_valid_depths),
        'median': np.median(all_valid_depths),
        'std': np.std(all_valid_depths),
        'min': np.min(all_valid_depths),
        'max': np.max(all_valid_depths),
        f'p{percentile_min}': np.percentile(all_valid_depths, percentile_min),
        f'p{percentile_max}': np.percentile(all_valid_depths, percentile_max),
        'p25': np.percentile(all_valid_depths, 25),
        'p75': np.percentile(all_valid_depths, 75),
        'sample_count': len(all_valid_depths)
    }
    
    # 使用百分位数作为深度范围，避免极端异常值
    computed_min = stats[f'p{percentile_min}']
    computed_max = stats[f'p{percentile_max}']
    
    # 添加一些边界扩展以确保覆盖大部分有效数据
    margin = (computed_max - computed_min) * 0.1
    computed_min = max(0.1, computed_min - margin)  # 不小于0.1米
    computed_max = min(2.0, computed_max + margin)  # 不大于2米（桌面场景合理范围）
    
    print(f"\n=== Depth Statistics Summary ===")
    print(f"Sample count: {stats['sample_count']:,}")
    print(f"Mean: {stats['mean']:.3f} m")
    print(f"Median: {stats['median']:.3f} m")
    print(f"Std Dev: {stats['std']:.3f} m")
    print(f"Min/Max (raw): {stats['min']:.3f} / {stats['max']:.3f} m")
    print(f"P{percentile_min}/P{percentile_max}: {stats[f'p{percentile_min}']:.3f} / {stats[f'p{percentile_max}']:.3f} m")
    print(f"P25/P75 (IQR): {stats['p25']:.3f} / {stats['p75']:.3f} m")
    print(f"\nComputed depth range: {computed_min:.3f} - {computed_max:.3f} m")
    print(f"(Original fixed range: {MIN_DEPTH:.3f} - {MAX_DEPTH:.3f} m)")
    
    return computed_min, computed_max, stats

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

def extract_frames_with_fps(video_path, h5_path, output_dir, target_fps=2, max_frames=None, use_dynamic_range=True, sample_rate=0.1, inverse_depth=False, interactive=False):
    """Extract RGB and depth frames based on target FPS with frame limit
    
    Args:
        inverse_depth: If True, apply 1/depth transformation (useful when depth is stored as disparity)
    """
    global MIN_DEPTH, MAX_DEPTH
    
    # 动态计算深度范围
    if use_dynamic_range:
        computed_min, computed_max, depth_stats = compute_depth_statistics(h5_path, sample_rate=sample_rate, inverse_depth=inverse_depth)
        MIN_DEPTH = computed_min
        MAX_DEPTH = computed_max
        print(f"\nUsing dynamic depth range: {MIN_DEPTH:.3f} - {MAX_DEPTH:.3f} m")
    else:
        print(f"\nUsing fixed depth range: {MIN_DEPTH:.3f} - {MAX_DEPTH:.3f} m")
    
    if inverse_depth:
        print("Note: Applying inverse transformation (1/depth) to depth values")
    
    if max_frames:
        print(f"\n=== Extracting frames at {target_fps} FPS (max {max_frames} frames) ===")
    else:
        print(f"\n=== Extracting ALL frames at {target_fps} FPS ===")
    
    # Open video to get original FPS
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval for target FPS
    frame_interval = max(1, int(original_fps / target_fps))
    
    print(f"Original FPS: {original_fps}, Target FPS: {target_fps}")
    print(f"Frame interval: {frame_interval} (extracting every {frame_interval}th frame)")
    print(f"Total frames in video: {total_frames}")
    if max_frames:
        print(f"Max frames to process: {max_frames}")
    else:
        print(f"Processing ALL available frames")
    
    # Create directories
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth") 
    vis_dir = os.path.join(output_dir, "vis")
    
    for dir_path in [rgb_dir, depth_dir, vis_dir]:
        create_output_directory(dir_path)
    
    # Open HDF5 file
    with h5py.File(h5_path, 'r') as h5f:
        depth_data = h5f['frames']
        depth_frames_count = depth_data.shape[0]
        
        print(f"Depth frames available: {depth_frames_count}")
        
        # Process frames with limits
        available_frames = min(total_frames, depth_frames_count)
        
        # Calculate how many frames we'll actually extract
        if max_frames:
            estimated_frames = min(available_frames // frame_interval, max_frames)
        else:
            estimated_frames = available_frames // frame_interval
        print(f"Estimated frames to extract: {estimated_frames}")
        
        extracted_count = 0
        
        for i in range(0, available_frames, frame_interval):
            # Stop if we've reached the max frame limit (if set)
            if max_frames and extracted_count >= max_frames:
                break
            # Extract RGB frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, rgb_frame = cap.read()
            if not ret:
                print(f"Warning: Could not read RGB frame {i}")
                continue
            
            # Store original height for coordinate adjustment
            original_height = rgb_frame.shape[0]
            
            # Crop RGB to match depth size (top half)
            rgb_frame = rgb_frame[:720, :, :]
            crop_offset = original_height - 720  # Amount cropped from bottom
            
            # Get depth frame
            depth_frame = depth_data[i]
            #print('RAW_DEPTH',depth_frame)
            
            # 如果需要，应用倒数变换（处理视差数据）
            if inverse_depth:
                valid_disparity = depth_frame > 0
                if np.any(valid_disparity):
                    depth_frame_converted = np.zeros_like(depth_frame)
                    depth_frame_converted[valid_disparity] = 1.0 / depth_frame[valid_disparity]
                    depth_frame = depth_frame_converted
            
            # 过滤异常值：将超出合理范围的深度值设为0（无效）
            depth_frame_filtered = depth_frame.copy()
            # 将小于最小值或大于最大值两倍的值设为无效
            invalid_mask = (depth_frame_filtered < MIN_DEPTH * 0.8) | (depth_frame_filtered > MAX_DEPTH * 2)
            
            # 打印异常值统计
            if i < 5 or i % 100 == 0:  # 只打印前5帧和每100帧
                valid_before = np.sum(depth_frame > 0)
                invalid_count = np.sum(invalid_mask & (depth_frame > 0))
                if invalid_count > 0:
                    print(f"  Frame {i}: Filtered {invalid_count}/{valid_before} abnormal depth values")
                    abnormal_values = depth_frame[invalid_mask & (depth_frame > 0)]
                    if len(abnormal_values) > 0:
                        print(f"    Abnormal range: {abnormal_values.min():.3f} - {abnormal_values.max():.3f}m")
            
            depth_frame_filtered[invalid_mask] = 0
            depth_frame = depth_frame_filtered
            #print('FILTED_DEPTH',depth_frame)
            
            # Save RGB frame
            rgb_filename = f"rgb_frame_{extracted_count:06d}.jpg"
            rgb_path = os.path.join(rgb_dir, rgb_filename)
            cv2.imwrite(rgb_path, rgb_frame)
            
            # Save depth frame (raw data)
            depth_raw_filename = f"depth_frame_{extracted_count:06d}.npy"
            depth_raw_path = os.path.join(depth_dir, depth_raw_filename)
            np.save(depth_raw_path, depth_frame)
            
            # Save depth frame as JPG
            depth_jpg_filename = f"depth_frame_{extracted_count:06d}.jpg"
            depth_jpg_path = os.path.join(depth_dir, depth_jpg_filename)
            create_depth_jpg(depth_frame, depth_jpg_path)
            #print(f"    raw depth saved: {depth_jpg_path}",depth_frame)
            
            
            # Create enhanced visualization
            create_enhanced_visualization(rgb_frame, depth_frame, vis_dir, extracted_count)
            
            # Show interactive display if enabled
            if interactive and extracted_count == 0:  # Show only for first frame
                print("\n  Showing interactive depth visualization for first frame...")
                print("  Hover mouse to see RGB and depth values. Press any key to continue.")
                display_img = create_interactive_depth_display(depth_frame, 'Raw Depth (Turbo)')
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            extracted_count += 1
            
            if extracted_count % 50 == 0:
                print(f"  Processed {extracted_count} frames...")
    
    cap.release()
    print(f"Extraction complete: {extracted_count} frames saved")
    return extracted_count

def create_colorbar(height, width=30, cmap_name='turbo', min_val=MIN_DEPTH, max_val=MAX_DEPTH):
    """创建垂直颜色条"""
    # 创建渐变数组 - 从0到1（顶部是远/大值，底部是近/小值）
    gradient = np.linspace(0, 1, height).reshape(-1, 1)  # 修正：从0到1而不是1到0
    gradient = np.repeat(gradient, width, axis=1)
    
    # 应用颜色映射
    cmap = plt.get_cmap(cmap_name)
    colorbar = cmap(gradient)
    colorbar_uint8 = (colorbar[:, :, :3] * 255).astype(np.uint8)
    
    # 添加刻度和标签
    colorbar_with_labels = np.ones((height, width + 60, 3), dtype=np.uint8) * 255
    colorbar_with_labels[:, :width, :] = colorbar_uint8
    
    # 添加深度值标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    
    # 添加最小值、中间值和最大值标签
    # 注意：顶部显示最大值（远），底部显示最小值（近）
    values = [max_val, (min_val + max_val) / 2, min_val]  # 修正：顺序从大到小
    positions = [10, height // 2, height - 10]  # 位置从上到下
    
    for val, pos in zip(values, positions):
        text = f'{val:.1f}m'
        cv2.putText(colorbar_with_labels, text, (width + 5, pos), 
                   font, font_scale, (0, 0, 0), 1)
        # 添加刻度线
        cv2.line(colorbar_with_labels, (width - 5, pos), (width, pos), (0, 0, 0), 1)
    
    return colorbar_with_labels

def create_depth_value_visualization(depth_frame):
    """创建带深度值标注的深度图可视化（不绘制等深度线）"""
    # 创建基础彩色深度图
    depth_colored = create_desktop_depth_colormap(depth_frame)
    
    # 准备深度数据
    valid_mask = depth_frame > 0
    if not np.any(valid_mask):
        return depth_colored
    
    # 创建副本用于添加深度值标注
    annotated_image = depth_colored.copy()
    
    h, w = depth_frame.shape
    
    # 在网格位置标注深度值
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    
    # 创建网格采样点（6x4网格）
    grid_x = 6  # 横向6个点
    grid_y = 4  # 纵向4个点
    margin = 50  # 边缘留白
    
    step_x = (w - 2 * margin) // (grid_x - 1)
    step_y = (h - 2 * margin) // (grid_y - 1)
    
    for i in range(grid_y):
        for j in range(grid_x):
            x = margin + j * step_x
            y = margin + i * step_y
            
            # 获取该位置的深度值（使用小窗口平均值避免噪点）
            window_size = 5
            x_start = max(0, x - window_size // 2)
            x_end = min(w, x + window_size // 2 + 1)
            y_start = max(0, y - window_size // 2)
            y_end = min(h, y + window_size // 2 + 1)
            
            depth_window = depth_frame[y_start:y_end, x_start:x_end]
            valid_depths = depth_window[depth_window > 0]
            
            if len(valid_depths) > 0:
                depth_val = np.median(valid_depths)  # 使用中值避免异常值
                text = f'{depth_val:.2f}'
                
                # 添加半透明背景框使文字更清晰
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                
                # 绘制背景框
                overlay = annotated_image.copy()
                cv2.rectangle(overlay, 
                             (x - text_size[0]//2 - 3, y - text_size[1]//2 - 3),
                             (x + text_size[0]//2 + 3, y + text_size[1]//2 + 3),
                             (0, 0, 0), -1)
                # 半透明混合
                cv2.addWeighted(overlay, 0.7, annotated_image, 0.3, 0, annotated_image)
                
                # 绘制文字（黄色）
                cv2.putText(annotated_image, text, 
                           (x - text_size[0]//2, y + text_size[1]//2), 
                           font, font_scale, (0, 255, 255), font_thickness)
                
                # 在标注点画个小圆圈
                cv2.circle(annotated_image, (x, y), 2, (255, 255, 255), -1)
    
    return annotated_image

def create_da2_depth_colormap(depth_frame, mask_zeros=True):
    """为DA2相对深度创建专门的可视化
    
    Args:
        depth_frame: DA2输出的深度数据（已经映射到米单位，但本质是相对的）
        mask_zeros: 是否将零值区域设为黑色
    """
    # 处理无效值
    valid_mask = depth_frame > 0 if mask_zeros else np.ones_like(depth_frame, dtype=bool)
    
    if not np.any(valid_mask):
        return np.zeros((*depth_frame.shape, 3), dtype=np.uint8)
    
    # 对于DA2，使用实际的最小最大值进行归一化（因为是相对深度）
    depth_min = depth_frame[valid_mask].min()
    depth_max = depth_frame[valid_mask].max()
    depth_normalized = (depth_frame - depth_min) / (depth_max - depth_min + 1e-8)
    
    # 使用TURBO colormap，与真实深度保持一致的视觉效果
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    colored_uint8 = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)  # 使用TURBO colormap
    
    # 无效区域设为黑色
    if mask_zeros:
        invalid_mask = depth_frame == 0
        colored_uint8[invalid_mask] = [0, 0, 0]
    
    return colored_uint8

def create_desktop_depth_colormap(depth_frame, mask_zeros=True, use_cv2_turbo=False):
    """简化的桌面抓取深度可视化，使用全局深度范围参数
    
    Args:
        depth_frame: 深度数据
        mask_zeros: 是否将零值区域设为黑色
        use_cv2_turbo: 使用cv2的turbo colormap (更快)
    """
    
    # 处理无效值
    valid_mask = depth_frame > 0 if mask_zeros else np.ones_like(depth_frame, dtype=bool)
    
    if not np.any(valid_mask):
        return np.zeros((*depth_frame.shape, 3), dtype=np.uint8)
    
    # 固定范围归一化
    depth_clamped = np.clip(depth_frame, MIN_DEPTH, MAX_DEPTH)
    depth_normalized = (depth_clamped - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
    
    if use_cv2_turbo:
        # 使用cv2的turbo colormap (更快)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        colored_uint8 = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    else:
        # 使用matplotlib的turbo配色方案
        cmap = plt.get_cmap('turbo')
        colored = cmap(depth_normalized)
        colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
    
    # 无效区域设为黑色
    if mask_zeros:
        invalid_mask = depth_frame == 0
        colored_uint8[invalid_mask] = [0, 0, 0]
    
    return colored_uint8

def create_depth_jpg(depth_frame, output_path, is_relative_depth=False):
    """将深度数据转换为灰度图像并保存为JPG
    
    Args:
        depth_frame: 深度数据
        output_path: 输出路径
        is_relative_depth: 是否为相对深度（如DA2输出），如果是则直接使用0-1归一化
    """
    valid_mask = depth_frame > 0
    
    if not np.any(valid_mask):
        depth_gray = np.zeros_like(depth_frame, dtype=np.uint8)
    else:
        if is_relative_depth:
            # DA2等相对深度：已经在合理范围内，直接归一化
            depth_min = depth_frame[valid_mask].min()
            depth_max = depth_frame[valid_mask].max()
            depth_normalized = (depth_frame - depth_min) / (depth_max - depth_min + 1e-8)
        else:
            # 真实深度：固定范围归一化到0-255
            depth_clamped = np.clip(depth_frame, MIN_DEPTH, MAX_DEPTH)
            depth_normalized = (depth_clamped - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
        
        depth_gray = (depth_normalized * 255).astype(np.uint8)
        
        # 无效区域设为0
        depth_gray[~valid_mask] = 0
    
    # 保存为JPG
    cv2.imwrite(output_path, depth_gray)
    return depth_gray

def create_interactive_depth_display(depth_frame, window_name='Depth Visualization'):
    """Create interactive depth visualization with mouse hover RGB display"""
    # Apply turbo colormap
    depth_normalized = np.clip(depth_frame * 255 / MAX_DEPTH, 0, 255).astype(np.uint8)
    vis_img = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
    display_img = vis_img.copy()
    
    # Mouse callback function to display RGB values on image
    def mouse_callback(event, x, y, flags, param):
        nonlocal display_img
        if event == cv2.EVENT_MOUSEMOVE:
            display_img = vis_img.copy()
            rgb = vis_img[y, x]
            depth_value = depth_frame[y, x]
            # Create text to display
            text = f"({x}, {y}) RGB: ({rgb[2]}, {rgb[1]}, {rgb[0]}) Depth: {depth_value:.3f}"
            # Add text to image with background for better visibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.6, 1)[0]
            # Draw background rectangle
            cv2.rectangle(display_img, (10, 10), (20 + text_size[0], 35), (0, 0, 0), -1)
            # Draw text
            cv2.putText(display_img, text, (15, 30), font, 0.6, (255, 255, 255), 1)
            cv2.imshow(window_name, display_img)
    
    # Display the image
    cv2.imshow(window_name, display_img)
    cv2.setMouseCallback(window_name, mouse_callback)
    return display_img

def create_enhanced_visualization(rgb_frame, depth_frame, vis_dir, frame_idx):
    """桌面抓取可视化方案，包含CDM处理"""
    
    # 创建CDM目录
    cdm_dir = os.path.join(vis_dir, "..", "cdm")
    os.makedirs(cdm_dir, exist_ok=True)
    
    # 创建DA2目录
    da2_dir = os.path.join(vis_dir, "..", "da2")
    os.makedirs(da2_dir, exist_ok=True)
    
    # 处理CDM深度增强
    cdm_depth = None
    if CDM_AVAILABLE:
        print(f"  Processing frame {frame_idx} with CDM...")
        try:
            import time
            # Convert BGR to RGB for CDM processing
            rgb_for_cdm = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            
            # 统计CDM推理时间
            cdm_start_time = time.time()
            cdm_depth = process_depth_with_cdm(rgb_for_cdm, depth_frame)
            cdm_inference_time = (time.time() - cdm_start_time) * 1000  # 转换为毫秒
            
            if cdm_depth is not None:
                # Print CDM stats
                valid_cdm = cdm_depth > 0
                if np.any(valid_cdm):
                    print(f"    CDM depth range: {cdm_depth[valid_cdm].min():.3f} - {cdm_depth[valid_cdm].max():.3f} m")
                    print(f"    CDM inference time: {cdm_inference_time:.2f} ms")
                
                # Save CDM processed depth
                cdm_depth_filename = f"cdm_depth_{frame_idx:06d}.npy"
                cdm_depth_path = os.path.join(cdm_dir, cdm_depth_filename)
                np.save(cdm_depth_path, cdm_depth)
                
                # Save CDM depth as JPG
                cdm_jpg_filename = f"cdm_depth_{frame_idx:06d}.jpg"
                cdm_jpg_path = os.path.join(cdm_dir, cdm_jpg_filename)
                create_depth_jpg(cdm_depth, cdm_jpg_path)
                #print(f"    CDM depth saved: {cdm_jpg_path}",cdm_depth)
            else:
                print(f"    CDM returned None for frame {frame_idx}")
                
        except Exception as e:
            print(f"  Warning: CDM processing failed for frame {frame_idx}: {e}")
            cdm_depth = None
    else:
        print(f"  CDM not available for frame {frame_idx}")
    
    # 创建深度彩色图 (使用cv2的turbo colormap)
    depth_colored = create_desktop_depth_colormap(depth_frame, use_cv2_turbo=True)
    
    # 确保尺寸一致
    h, w = depth_frame.shape
    if rgb_frame.shape[:2] != (h, w):
        rgb_frame_resized = cv2.resize(rgb_frame, (w, h))
    else:
        rgb_frame_resized = rgb_frame
    
    # 添加标签和信息的辅助函数
    def add_panel_label(image, label, stats_text=None, frame_info=None, add_colorbar=False):
        """在图像面板上添加标签和统计信息，可选添加颜色条"""
        result = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 如果需要添加颜色条
        if add_colorbar:
            # 创建颜色条
            colorbar = create_colorbar(image.shape[0], width=30)
            # 在右侧添加颜色条
            result = np.hstack([result, colorbar])
        
        # 添加主标签背景和文字
        cv2.rectangle(result, (10, 10), (220, 40), (0, 0, 0), -1)
        cv2.putText(result, label, (15, 30), font, 0.7, (255, 255, 255), 2)
        
        # 添加统计信息（如果提供）
        if stats_text:
            cv2.rectangle(result, (10, 45), (280, 70), (0, 0, 0), -1)
            cv2.putText(result, stats_text, (15, 62), font, 0.5, (255, 255, 255), 1)
        
        # 添加帧信息（右上角，考虑颜色条的宽度）
        if frame_info:
            text_size = cv2.getTextSize(frame_info, font, 0.5, 1)[0]
            # 如果有颜色条，将帧信息放在颜色条左侧
            if add_colorbar:
                x = image.shape[1] - text_size[0] - 15  # 原始图像宽度
            else:
                x = result.shape[1] - text_size[0] - 15
            cv2.rectangle(result, (x - 5, 10), (x + text_size[0] + 5, 35), (0, 0, 0), -1)
            cv2.putText(result, frame_info, (x, 27), font, 0.5, (0, 255, 0), 1)
        
        return result
    
    # 为每个面板添加标签
    frame_info = f'Frame #{frame_idx:04d}'
    rgb_labeled = add_panel_label(rgb_frame_resized, 'RGB', frame_info=frame_info)
    
    # 计算原始深度统计
    valid_depths = depth_frame[(depth_frame > 0) & (depth_frame <= MAX_DEPTH)]
    if len(valid_depths) > 0:
        depth_min, depth_max = np.min(valid_depths), np.max(valid_depths)
        depth_stats = f'Range: {depth_min:.2f}-{depth_max:.2f}m'
    else:
        depth_stats = 'No valid depth'
    # 为原始深度添加颜色条
    depth_labeled = add_panel_label(depth_colored, 'Raw Depth', depth_stats, add_colorbar=True)
    
    # 确保RGB面板和深度面板宽度一致（深度面板多了颜色条）
    if depth_labeled.shape[1] > rgb_labeled.shape[1]:
        padding_width = depth_labeled.shape[1] - rgb_labeled.shape[1]
        # 在RGB右侧添加白色填充
        rgb_padding = np.ones((rgb_labeled.shape[0], padding_width, 3), dtype=np.uint8) * 255
        rgb_labeled = np.hstack([rgb_labeled, rgb_padding])
    
    # 先处理DA2（第三幅图）
    da2_labeled = None
    if DA2_AVAILABLE:
        print("  Processing with DA2 (vitg)...")
        import time
        # DA2只需要RGB图像
        da2_start_time = time.time()
        da2_depth_normalized = process_rgb_with_da2(rgb_frame_resized)
        da2_inference_time = (time.time() - da2_start_time) * 1000  # 转换为毫秒
        
        if da2_depth_normalized is not None:
            # 将归一化深度转换到实际深度范围（假设场景深度）
            da2_depth = da2_depth_normalized * (MAX_DEPTH - MIN_DEPTH) + MIN_DEPTH
            
            # 创建DA2深度彩色可视化（使用专门的相对深度可视化）
            da2_colored = create_da2_depth_colormap(da2_depth)
            
            # 保存DA2深度数据
            da2_depth_filename = f"da2_depth_{frame_idx:06d}.npy"
            da2_depth_path = os.path.join(da2_dir, da2_depth_filename)
            np.save(da2_depth_path, da2_depth)
            
            # 保存DA2深度为JPG（标记为相对深度）
            da2_jpg_filename = f"da2_depth_{frame_idx:06d}.jpg"
            da2_jpg_path = os.path.join(da2_dir, da2_jpg_filename)
            create_depth_jpg(da2_depth, da2_jpg_path, is_relative_depth=True)
            
            # 统计信息
            da2_valid = da2_depth > 0
            if np.any(da2_valid):
                da2_min, da2_max = da2_depth[da2_valid].min(), da2_depth[da2_valid].max()
                da2_stats = f'Monocular (RGB only): {da2_min:.2f}-{da2_max:.2f}m'
            else:
                da2_stats = 'No valid depth'
            
            da2_labeled = add_panel_label(da2_colored, 'DA2 Relative Depth (vitg)', da2_stats + ' [Estimated]', add_colorbar=True)
            print(f"    DA2 depth range: {da2_min:.3f} - {da2_max:.3f} m")
            print(f"    DA2 inference time: {da2_inference_time:.2f} ms")
            print(f"    DA2 depth saved: {da2_jpg_path}")
        else:
            print("  DA2 processing failed")
    
    # 再处理CDM（第四幅图）
    if cdm_depth is not None:
        cdm_colored = create_desktop_depth_colormap(cdm_depth, use_cv2_turbo=True)
        
        # 计算CDM深度统计
        cdm_valid = cdm_depth[(cdm_depth > 0) & (cdm_depth <= MAX_DEPTH)]
        if len(cdm_valid) > 0:
            cdm_min, cdm_max = np.min(cdm_valid), np.max(cdm_valid)
            cdm_stats = f'Range: {cdm_min:.2f}-{cdm_max:.2f}m'
        else:
            cdm_stats = 'No valid depth'
        # 为CDM深度也添加颜色条
        cdm_labeled = add_panel_label(cdm_colored, 'CDM Enhanced', cdm_stats, add_colorbar=True)
        
        # 纵向堆叠四个面板：RGB, Raw Depth, DA2, CDM
        if da2_labeled is not None:
            combined = np.vstack([rgb_labeled, depth_labeled, da2_labeled, cdm_labeled])
        else:
            # 如果DA2失败，只显示三个面板
            combined = np.vstack([rgb_labeled, depth_labeled, cdm_labeled])
    else:
        # 没有CDM，检查是否有DA2
        if da2_labeled is not None:
            # 纵向堆叠三个面板：RGB, Raw Depth, DA2
            combined = np.vstack([rgb_labeled, depth_labeled, da2_labeled])
        else:
            # 纵向堆叠两个面板
            combined = np.vstack([rgb_labeled, depth_labeled])
    
    # 保存合并可视化
    suffix = "with_cdm" if cdm_depth is not None else "raw_only"
    vis_filename = f"combined_{suffix}_{frame_idx:06d}.jpg"
    vis_path = os.path.join(vis_dir, vis_filename)
    cv2.imwrite(vis_path, combined)

def create_depth_histogram(depth_frame, vis_dir, frame_idx):
    """Create depth distribution histogram"""
    valid_depths = depth_frame[depth_frame > 0]
    
    if len(valid_depths) == 0:
        return
    
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    plt.subplot(2, 2, 1)
    plt.hist(valid_depths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'Depth Distribution (Frame {frame_idx})')
    plt.xlabel('Depth Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(2, 2, 2)
    plt.boxplot(valid_depths)
    plt.title('Depth Box Plot')
    plt.ylabel('Depth Value')
    plt.grid(True, alpha=0.3)
    
    # Depth map as heatmap
    plt.subplot(2, 2, 3)
    masked_depth = np.where(depth_frame > 0, depth_frame, np.nan)
    plt.imshow(masked_depth, cmap='viridis', aspect='auto')
    plt.colorbar(label='Depth')
    plt.title('Depth Heatmap')
    
    # Statistics text
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats_text = f"""
    Frame: {frame_idx}
    Valid pixels: {len(valid_depths):,}
    Invalid pixels: {np.sum(depth_frame == 0):,}
    Min depth: {np.min(valid_depths):.3f}
    Max depth: {np.max(valid_depths):.3f}
    Mean depth: {np.mean(valid_depths):.3f}
    Std depth: {np.std(valid_depths):.3f}
    """
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save histogram
    hist_filename = f"depth_analysis_{frame_idx:06d}.png"
    hist_path = os.path.join(vis_dir, hist_filename)
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Enhanced RGB and depth frame extraction with improved visualization')
    parser.add_argument('--video', type=str, default='data/bag/0912_topdownbags/topdown_grasp01.mp4', help='Path to video file (default: topdown_grasp01.mp4)')
    parser.add_argument('--h5', type=str, default='data/bag/0912_topdownbags/topdown_grasp01.h5', help='Path to HDF5 file (default: topdown_grasp01.h5)')
    parser.add_argument('--output', type=str, default='data/bag/0912_topdownbags/images/', help='Output directory (default: vis)')
    parser.add_argument('--fps', type=float, default=2.0, help='Target FPS for extraction (default: 2.0)')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum number of frames to process (default: None - process all)')
    parser.add_argument('--dynamic-range', action='store_true', default=False, help='Use dynamic depth range calculation (default: True)')
    parser.add_argument('--no-dynamic-range', dest='dynamic_range', action='store_false', help='Use fixed depth range')
    parser.add_argument('--sample-rate', type=float, default=0.01, help='Sample rate for depth statistics (0.0-1.0, default: 0.01)')
    parser.add_argument('--inverse-depth', action='store_true', default=False, help='Apply inverse (1/depth) to depth values if stored as disparity (default: False)')
    parser.add_argument('--interactive', action='store_true', default=False, help='Show interactive depth visualization with mouse hover (default: False)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")
    if not os.path.exists(args.h5):
        raise FileNotFoundError(f"HDF5 file not found: {args.h5}")
    
    print(f"Input video: {args.video}")
    print(f"Input HDF5: {args.h5}")
    print(f"Output directory: {args.output}")
    print(f"Target FPS: {args.fps}")
    
    # Create main output directory
    create_output_directory(args.output)
    
    # Extract and visualize frames
    frame_count = extract_frames_with_fps(args.video, args.h5, args.output, args.fps, args.max_frames, 
                                         args.dynamic_range, args.sample_rate, args.inverse_depth, args.interactive)
    
    print(f"\n=== Processing Summary ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Output directory structure:")
    print(f"  {args.output}/rgb/              - RGB frames (.jpg)")
    print(f"  {args.output}/depth/            - Raw depth data (.npy)")
    print(f"  {args.output}/cdm/              - CDM enhanced depth data (.npy, .jpg)")
    print(f"  {args.output}/da2/              - DA2 estimated depth data (.npy, .jpg)")
    print(f"  {args.output}/vis/              - Enhanced visualizations (.jpg, .png)")
    print(f"    - combined_[with_cdm|raw_only]_*.jpg   - Multi-panel visualizations")
    print(f"    - depth_analysis_*.png        - Depth statistics and distributions")

if __name__ == "__main__":
    main()