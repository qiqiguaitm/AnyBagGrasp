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

# 桌面抓取场景全局深度范围参数
MIN_DEPTH = 0.2  # 最近深度(米) - 手部/工具区域
MAX_DEPTH = 1.5  # 最远深度(米) - 背景区域

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

def extract_frames_with_fps(video_path, h5_path, output_dir, target_fps=2, max_frames=300):
    """Extract RGB and depth frames based on target FPS with frame limit"""
    print(f"\n=== Extracting frames at {target_fps} FPS (max {max_frames} frames) ===")
    
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
    print(f"Max frames to process: {max_frames}")
    
    # Create directories
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth") 
    vis_dir = os.path.join(output_dir, "visualizations")
    
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
        estimated_frames = min(available_frames // frame_interval, max_frames)
        print(f"Estimated frames to extract: {estimated_frames}")
        
        extracted_count = 0
        
        for i in range(0, available_frames, frame_interval):
            # Stop if we've reached the max frame limit
            if extracted_count >= max_frames:
                break
            # Extract RGB frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, rgb_frame = cap.read()
            if not ret:
                print(f"Warning: Could not read RGB frame {i}")
                continue
            
            # Crop RGB to match depth size (top half)
            rgb_frame = rgb_frame[:720, :, :]
            
            # Get depth frame
            depth_frame = depth_data[i]
            
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
            
            # Create enhanced visualization
            create_enhanced_visualization(rgb_frame, depth_frame, vis_dir, extracted_count)
            
            extracted_count += 1
            
            if extracted_count % 50 == 0:
                print(f"  Processed {extracted_count} frames...")
    
    cap.release()
    print(f"Extraction complete: {extracted_count} frames saved")
    return extracted_count

def create_desktop_depth_colormap(depth_frame, mask_zeros=True):
    """简化的桌面抓取深度可视化，使用全局深度范围参数"""
    
    # 处理无效值
    valid_mask = depth_frame > 0 if mask_zeros else np.ones_like(depth_frame, dtype=bool)
    
    if not np.any(valid_mask):
        return np.zeros((*depth_frame.shape, 3), dtype=np.uint8)
    
    # 固定范围归一化
    depth_clamped = np.clip(depth_frame, MIN_DEPTH, MAX_DEPTH)
    depth_normalized = (depth_clamped - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
    
    # 使用turbo配色方案（适合机器人视觉）
    # 蓝色(近) -> 青色 -> 绿色 -> 黄色 -> 红色(远)
    cmap = plt.get_cmap('turbo')
    colored = cmap(depth_normalized)
    colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
    
    # 无效区域设为黑色
    if mask_zeros:
        invalid_mask = depth_frame == 0
        colored_uint8[invalid_mask] = [0, 0, 0]
    
    return colored_uint8

def create_depth_jpg(depth_frame, output_path):
    """将深度数据转换为灰度图像并保存为JPG"""
    valid_mask = depth_frame > 0
    
    if not np.any(valid_mask):
        depth_gray = np.zeros_like(depth_frame, dtype=np.uint8)
    else:
        # 固定范围归一化到0-255
        depth_clamped = np.clip(depth_frame, MIN_DEPTH, MAX_DEPTH)
        depth_normalized = (depth_clamped - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
        depth_gray = (depth_normalized * 255).astype(np.uint8)
        
        # 无效区域设为0
        depth_gray[~valid_mask] = 0
    
    # 保存为JPG
    cv2.imwrite(output_path, depth_gray)
    return depth_gray

def create_enhanced_visualization(rgb_frame, depth_frame, vis_dir, frame_idx):
    """桌面抓取可视化方案，包含CDM处理"""
    
    # 创建CDM目录
    cdm_dir = os.path.join(vis_dir, "..", "cdm")
    os.makedirs(cdm_dir, exist_ok=True)
    
    # 处理CDM深度增强
    cdm_depth = None
    if CDM_AVAILABLE:
        print(f"  Processing frame {frame_idx} with CDM...")
        try:
            # Convert BGR to RGB for CDM processing
            rgb_for_cdm = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            cdm_depth = process_depth_with_cdm(rgb_for_cdm, depth_frame)
            
            if cdm_depth is not None:
                # Print CDM stats
                valid_cdm = cdm_depth > 0
                if np.any(valid_cdm):
                    print(f"    CDM depth range: {cdm_depth[valid_cdm].min():.3f} - {cdm_depth[valid_cdm].max():.3f} m")
                
                # Save CDM processed depth
                cdm_depth_filename = f"cdm_depth_{frame_idx:06d}.npy"
                cdm_depth_path = os.path.join(cdm_dir, cdm_depth_filename)
                np.save(cdm_depth_path, cdm_depth)
                
                # Save CDM depth as JPG
                cdm_jpg_filename = f"cdm_depth_{frame_idx:06d}.jpg"
                cdm_jpg_path = os.path.join(cdm_dir, cdm_jpg_filename)
                create_depth_jpg(cdm_depth, cdm_jpg_path)
                print(f"    CDM depth saved: {cdm_jpg_path}")
            else:
                print(f"    CDM returned None for frame {frame_idx}")
                
        except Exception as e:
            print(f"  Warning: CDM processing failed for frame {frame_idx}: {e}")
            cdm_depth = None
    else:
        print(f"  CDM not available for frame {frame_idx}")
    
    # 创建深度彩色图
    depth_colored = create_desktop_depth_colormap(depth_frame)
    
    # 确保尺寸一致
    h, w = depth_frame.shape
    if rgb_frame.shape[:2] != (h, w):
        rgb_frame_resized = cv2.resize(rgb_frame, (w, h))
    else:
        rgb_frame_resized = rgb_frame
    
    # 添加标签和信息的辅助函数
    def add_panel_label(image, label, stats_text=None, frame_info=None):
        """在图像面板上添加标签和统计信息"""
        result = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 添加主标签背景和文字
        cv2.rectangle(result, (10, 10), (220, 40), (0, 0, 0), -1)
        cv2.putText(result, label, (15, 30), font, 0.7, (255, 255, 255), 2)
        
        # 添加统计信息（如果提供）
        if stats_text:
            cv2.rectangle(result, (10, 45), (280, 70), (0, 0, 0), -1)
            cv2.putText(result, stats_text, (15, 62), font, 0.5, (255, 255, 255), 1)
        
        # 添加帧信息（右上角）
        if frame_info:
            text_size = cv2.getTextSize(frame_info, font, 0.5, 1)[0]
            x = result.shape[1] - text_size[0] - 15
            cv2.rectangle(result, (x - 5, 10), (result.shape[1] - 10, 35), (0, 0, 0), -1)
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
    depth_labeled = add_panel_label(depth_colored, 'Raw Depth', depth_stats)
    
    # 如果有CDM结果，创建三行显示，否则两行显示
    if cdm_depth is not None:
        cdm_colored = create_desktop_depth_colormap(cdm_depth)
        
        # 计算CDM深度统计
        cdm_valid = cdm_depth[(cdm_depth > 0) & (cdm_depth <= MAX_DEPTH)]
        if len(cdm_valid) > 0:
            cdm_min, cdm_max = np.min(cdm_valid), np.max(cdm_valid)
            cdm_stats = f'Range: {cdm_min:.2f}-{cdm_max:.2f}m'
        else:
            cdm_stats = 'No valid depth'
        cdm_labeled = add_panel_label(cdm_colored, 'CDM Enhanced', cdm_stats)
        
        # 纵向堆叠三个面板
        combined = np.vstack([rgb_labeled, depth_labeled, cdm_labeled])
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
    parser.add_argument('--output', type=str, default='images', help='Output directory (default: images)')
    parser.add_argument('--fps', type=float, default=0.5, help='Target FPS for extraction (default: 2.0)')
    parser.add_argument('--max-frames', type=int, default=3000, help='Maximum number of frames to process (default: 300)')
    
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
    frame_count = extract_frames_with_fps(args.video, args.h5, args.output, args.fps, args.max_frames)
    
    print(f"\n=== Processing Summary ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Output directory structure:")
    print(f"  {args.output}/rgb/              - RGB frames (.jpg)")
    print(f"  {args.output}/depth/            - Raw depth data (.npy)")
    print(f"  {args.output}/visualizations/   - Enhanced visualizations (.jpg, .png)")
    print(f"    - combined_[colormap]_*.jpg   - Side-by-side RGB+Depth with different colormaps")
    print(f"    - depth_analysis_*.png        - Depth statistics and distributions")

if __name__ == "__main__":
    main()